from __future__ import annotations

import json
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from email.utils import parsedate_to_datetime

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_naver_community.tool import NaverNewsSearch

from state import GraphState, NewsSignal, Sentiment, initial_state

load_dotenv()
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

llm = ChatOllama(model="gpt-oss:20b")
MAX_CONTENT_CHARS = 1200
DEBUG_SENT = True 
DEBUG_SNIPPET = 300 

def get_naver_tool() -> NaverNewsSearch:
    if not NAVER_ID or not NAVER_SECRET:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 없음")
    return NaverNewsSearch(naver_client_id=NAVER_ID, naver_client_secret=NAVER_SECRET)

naver_tool = get_naver_tool()


class SummaryAgentState(TypedDict, total=False):
    content: str
    focus: str            
    summary_strategy: str
    summary: str

def node_decide_strategy(state: SummaryAgentState) -> SummaryAgentState:
    prompt = ChatPromptTemplate.from_template(
        """
        아래 기사 본문을 읽고, 타깃 기업({focus})의 주가/기업가치에 직접적인 영향을 주는 정보만
        우선 요약하기 위한 전략을 한두 문장으로 제시해줘.
            - {focus}의 실적/가이던스/수주/규제/경쟁·공급망/재무 이벤트(증자·자사주 등) 중심
            - 단순 업계 일반론, 타사/거시 코멘트만 있는 경우는 "관련성 낮음"으로 판단

            기사 본문:
            {content}

            전략:
        """
    )
    chain = prompt | llm
    strategy = chain.invoke({"content": state["content"], "focus": state["focus"]}).content
    state["summary_strategy"] = strategy

    return state

def node_summarize_with_strategy(state: SummaryAgentState) -> SummaryAgentState:
    prompt = ChatPromptTemplate.from_template(
        """
        지침에 따라 {focus} 관련 핵심만 요약해줘.
            - {focus}에 '직접' 영향 없는 내용은 제외
            - 반드시 3개 불릿 이내: (1) 주가 영향 요인, (2) 촉매·수치, (3) 리스크/유의점
            - 정말 {focus}와 관련이 거의 없으면 정확히 한 단어로만 "UNRELATED" 출력

            지침:
            {strategy}

            기사 본문:
            {content}

            요약:
        """
    )
    chain = prompt | llm
    summary = chain.invoke({
        "strategy": state["summary_strategy"],
        "content": state["content"],
        "focus": state["focus"],
    }).content
    state["summary"] = summary.strip()
    return state

def build_summary_agent_graph():
    g = StateGraph(SummaryAgentState)
    g.add_node("decide_strategy", node_decide_strategy)
    g.add_node("summarize_with_strategy", node_summarize_with_strategy)
    g.add_edge(START, "decide_strategy")
    g.add_edge("decide_strategy", "summarize_with_strategy")
    g.add_edge("summarize_with_strategy", END)
    return g.compile()

summary_agent = build_summary_agent_graph()

_TAG_RE = re.compile(r"</?[^>]+>")

def _strip_tags(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()

def _to_iso8601(s: str) -> str:
    if not s:
        return ""
    try:
        dt = parsedate_to_datetime(s)
        return dt.isoformat()
    except Exception:
        m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", s)
        if m:
            y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
            return f"{y}-{mo}-{d}"
        return ""

def _normalize_naver_payload(raw: Any) -> dict:
    try:
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {"items": []}
        elif isinstance(raw, list):
            data = {"items": raw}
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {"items": []}
    except Exception:
        data = {"items": []}

    if not isinstance(data, dict):
        data = {"items": []}
    if not isinstance(data.get("items"), list):
        data["items"] = []
    return data

def _fingerprint(title: str, link: str) -> str:
    base = (title or "").strip().lower() + "|" + (link or "").strip().lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def _dedup_and_clean(items: List[Dict[str, Any]]) -> Tuple[List[Dict], int]:
    seen, out, dropped = set(), [], 0
    for it in items:
        title = _strip_tags(it.get("title") or it.get("headline") or "")
        link  = (it.get("originallink") or it.get("link") or it.get("url") or "").strip()
        content = _strip_tags(it.get("content") or it.get("description") or it.get("summary") or "")
        pub_raw = it.get("published_at") or it.get("pubDate") or it.get("date") or ""
        pub = _to_iso8601(pub_raw)

        if not title or not link:
            dropped += 1
            continue

        fp = _fingerprint(title, link)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)

        out.append({"title": title, "link": link, "content": content, "published_at": pub})
    return out, dropped

REFINE_LIST = [
    "{q}",
    "{q} 실적 OR 공시",
    "{q} 뉴스 -블로그 -카페",
    "site:news.naver.com {q}",
]
def _refined_query(q: str, attempt: int) -> str:
    pat = REFINE_LIST[min(attempt, len(REFINE_LIST)-1)]
    return pat.replace("{q}", q)

def run_news_graph(
    app_state: GraphState,
    query: str,
    n_items: int = 5,
    k_sentences: int = 3,   # 현재 로직에선 직접 쓰진 않지만, 인터페이스 유지
    max_attempts: int = 3,
) -> Tuple[GraphState, Dict[str, Any]]:

    # 그래프 바깥 공유 컨텍스트(임시 값 저장소) — 상태에 아무 키도 추가하지 않음
    ctx: Dict[str, Any] = dict(
        query=query,
        n_items=n_items,
        k_sentences=k_sentences,
        max_attempts=max_attempts,
        attempts=0,
        collected=[],
        cleaned=[],
        dropped=0,
        items=[],
    )

    def node_search(state: GraphState) -> GraphState:
        q = _refined_query(ctx["query"], ctx["attempts"])
        start = 1 + ctx["attempts"] * ctx["n_items"]
        try:
            raw = naver_tool.run(q, display=ctx["n_items"], start=start, sort="date")
        except TypeError:
            raw = naver_tool.run(q, display=ctx["n_items"])
        payload = _normalize_naver_payload(raw)
        items = payload.get("items", [])
        print(f"[DEBUG] attempt={ctx['attempts']+1}, query='{q}', fetched={len(items)}, start={start}", flush=True)
        ctx["collected"] = items
        ctx["attempts"] += 1

        return state  

    def node_clean_and_check(state: GraphState) -> GraphState:
        collected = ctx.get("collected", [])
        print(f"[DEBUG] collected_total={len(collected)} before clean", flush=True)
        cleaned, dropped = _dedup_and_clean(collected)
        print(f"[DEBUG] cleaned={len(cleaned)}, dropped={dropped}", flush=True)
        ctx["cleaned"] = cleaned
        ctx["dropped"] = dropped

        return state

    def route_retry(state: GraphState) -> str:
        if len(ctx.get("cleaned", [])) >= ctx["n_items"]:
            return "enough"
        if ctx["attempts"] >= ctx["max_attempts"]:
            return "exhausted"
        return "retry"

    def node_summarize(state: GraphState) -> GraphState:
        cleaned = ctx.get("cleaned", [])[: ctx["n_items"]]
        final_items: List[Dict[str, Any]] = []
        for i, it in enumerate(cleaned):
            content = str(it.get("content", ""))[:MAX_CONTENT_CHARS]
            if not content:
                continue

            # ★ 타깃 기업 = ctx["query"]를 focus로 전달
            res: SummaryAgentState = summary_agent.invoke({
                "content": content,
                "focus": ctx["query"],
            })

            summary = (res.get("summary") or "").strip()
            if summary == "UNRELATED":
                # ★ 타깃과 관련성 낮으면 스킵(원하면 포함해도 됨)
                continue

            final_items.append({
                "title": it["title"],
                "link": it["link"],
                "published_at": it.get("published_at", ""),
                "summary": summary,
            })

        ctx["items"] = final_items
        return state
    
    # 주가 전망 분석
    def node_analyze_sentiment(state: GraphState) -> GraphState:
        """
        요약된 기사별 감성을 LLM으로 분류해 ctx['items'][i]['sentiment']에 기록.
        긍정/부정만 판별 (POSITIVE / NEGATIVE)
        """
        items = ctx.get("items", [])
        if not items:
            return state

        prompt = ChatPromptTemplate.from_template(
            """너는 한국 주식 시장 뉴스 요약을 읽고 주가 방향을 판단하는 분석기다.

            반드시 다음 중 하나만 출력하세요 (대문자, 한 단어):
            POSITIVE
            NEGATIVE

            요약:
            {summary}

            정답:"""
        )

        clf = prompt | llm

        labeled = []
        for it in items:
            summary = (it.get("summary") or "").strip()
            if not summary:
                it["sentiment"] = "NEGATIVE"  # 여기 보수적으로 잡으려고 Negative로 했음
                labeled.append(it)
                continue

            try:
                out = clf.invoke({"summary": summary}).content.strip().upper()
            except Exception as e:
                out = "NEGATIVE"  
            token = out.split()[0] if out else ""
            if token not in ("POSITIVE", "NEGATIVE"):
                token = "NEGATIVE"  

            it["sentiment"] = token
            labeled.append(it)

        ctx["items"] = labeled
        return state
    
    def node_finalize(state: GraphState) -> GraphState:
        items = ctx.get("items", [])
        signals = [
            NewsSignal(
                ticker=str(ctx["query"]),
                summary=it.get("summary", "") or "",
                # 여기서 숫자로 매핑
                sentiment=(1 if it.get("sentiment") == "POSITIVE" else 0),
                decision=None,
            )
            for it in items
        ]
        existing = list(state.get("news_signals") or [])
        existing.extend(signals)
        state["news_signals"] = existing
        print(f"[DEBUG] news_signals updated (+{len(signals)})", flush=True)
        
        return state

    g = StateGraph(GraphState)
    g.add_node("search", node_search)
    g.add_node("clean", node_clean_and_check)
    g.add_node("summarize", node_summarize)
    g.add_node("analyze_sentiment", node_analyze_sentiment)  # ★ 추가
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "search")
    g.add_edge("search", "clean")
    g.add_conditional_edges(
        "clean", route_retry,
        {"retry": "search", "enough": "summarize", "exhausted": "summarize"}
    )
    g.add_edge("summarize", "analyze_sentiment") 
    g.add_edge("analyze_sentiment", "finalize")    
    g.add_edge("finalize", END)
    graph = g.compile()

    # 초기 상태는 기존 GraphState 그대로 전달 (새 키 주입 없음)
    out_state: GraphState = graph.invoke(app_state)

    # 호출자에게 보여줄 결과 JSON (ctx에서 구성)
    result_json = {
        "items": ctx.get("items", []),
        "attempts": ctx.get("attempts", 0),
        "dropped": ctx.get("dropped", 0),
    }
    return out_state, result_json

if __name__ == "__main__":
    st = initial_state()
    st, result = run_news_graph(
        app_state=st,
        query="2025년 삼성전자 전망",
        n_items=1,
        max_attempts=5,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for sig in st["news_signals"]:
        print(f"{sig.ticker} → {sig.sentiment}: {sig.summary[:60]}...")