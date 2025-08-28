# python >=3.10
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# ===== Enums =====

class UIStep(str, Enum):
    INPUT_GOAL = "INPUT_GOAL"                     # 목표 금액/기간 입력
    INPUT_PERCENT = "INPUT_PERCENT"               # % 입력 (투자 가능 금액 안내)
    ASK_HEART = "ASK_HEART"                       # 마음 물어보기
    SHOW_PORTFOLIOS = "SHOW_PORTFOLIOS"           # 30/50/70% 시나리오 제시
    CONFIRM_PORTFOLIO = "CONFIRM_PORTFOLIO"       # 사용자 선택 대기
    REBALANCING = "REBALANCING"                   # 리밸런싱 단계
    CHATBOT = "CHATBOT"                           # 챗봇 상호작용
    DONE = "DONE"                                 # 종료

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class NewsDecision(str, Enum):
    HOLD = "hold"          # 그대로 보유
    SELL = "sell"          # 매도
    REVIEW = "review"      # 재검토(추가 분석 필요)
    RECOMMEND_REFRESH = "recommend_refresh"  # 추천 알고리즘 다시 시작

class RiskLevel(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"

class SourceType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    FUND = "fund"
    BOND = "bond"
    CASH = "cash"
    OTHER = "other"


# ===== Dataclasses (payload structures) =====

@dataclass
class Goal:
    target_amount: float                   # 목표 금액
    target_horizon_months: int             # 목표 기간(개월)
    created_ts: str | None = None          # ISO8601

@dataclass
class IncomeExpense:
    monthly_income: float                  # 월급(세후 등 기준 통일)
    fixed_expense: float                   # 고정지출(월)
    variable_expense_hint: float | None = None  # 선택: 변동지출 힌트

@dataclass
class Investable:
    percent_input: float                   # 사용자가 입력한 % (0~100)
    amount: float                          # 실제 투자 가능 금액(원)
    rationale: str | None = None

@dataclass
class Product:
    ticker: str
    name: str
    source_type: SourceType
    risk: RiskLevel
    metadata: Dict[str, str] = field(default_factory=dict)  # ex) 섹터, 지역 등

@dataclass
class AllocationItem:
    product: Product
    weight: float                          # 포트폴리오 내 비중 (0~1)
    expected_return_yr: float | None = None
    expected_vol_yr: float | None = None
    notes: str | None = None

@dataclass
class Portfolio:
    id: str                                # "alloc_30", "alloc_50", "alloc_70" 등
    items: List[AllocationItem]
    investable_amount: float               # 금액 기준
    expected_return_yr: float | None = None
    expected_vol_yr: float | None = None
    risk_level: RiskLevel | None = None
    explanation: str | None = None

@dataclass
class RebalanceAction:
    from_ticker: str
    to_ticker: str
    amount: float                          # 원화 금액(+/-)
    reason: str

@dataclass
class RebalancePlan:
    before: Portfolio
    after: Portfolio
    actions: List[RebalanceAction]
    trigger: str                           # 트리거 설명 (밴드 이탈, 드로우다운 등)

@dataclass
class NewsSignal:
    ticker: str
    summary: str
    sentiment: Sentiment
    decision: NewsDecision | None = None   # 판단 결과

@dataclass
class RAGTrace:
    query: str
    topk: int
    ids: List[str]                         # 검색된 상품/문서 id
    notes: str | None = None


# ===== LangGraph State (TypedDict) =====
# - messages는 LangGraph의 add_messages 어노테이션을 써서 자동 머지되도록 설정
# - 나머지 키는 노드에서 부분 업데이트(얕은 머지)로 다루는 전제

class GraphState(TypedDict, total=False):
    # 대화/프론트 단계
    ui_step: UIStep

    # 입력/프로필
    goal: Goal
    income_expense: IncomeExpense
    investable: Investable
    risk_preference: RiskLevel | None             # 마음 물어보기 결과를 리스크로 맵핑

    # RAG/추천
    rag_trace: RAGTrace
    candidate_portfolios: List[Portfolio]         # 30/50/70% 후보
    selected_portfolio: Portfolio | None          # 최종 선택안

    # 리밸런싱
    rebalance_plan: RebalancePlan | None

    # 뉴스 파이프라인
    news_signals: List[NewsSignal]                # 수집/요약/감성 결과
    news_last_cursor: str | None                  # 크롤링 커서(옵션)

    # 감사·실행 추적
    events: List[Dict]                            # 작은 이벤트 로그(노드 진입/결정 등)

    # 대화 메시지 (LangChain messages)
    messages: Annotated[List[AnyMessage], add_messages]


# ===== 초기 상태 헬퍼 =====

def initial_state() -> GraphState:
    return GraphState(
        ui_step=UIStep.INPUT_GOAL,
        messages=[],
        events=[],
        news_signals=[],
    )


# ===== 유틸: 상태 전이/업데이트 헬퍼 =====

def update_goal(state: GraphState, target_amount: float, months: int, ts: Optional[str] = None) -> GraphState:
    state["goal"] = Goal(target_amount=target_amount, target_horizon_months=months, created_ts=ts)
    state["ui_step"] = UIStep.INPUT_PERCENT
    state.setdefault("events", []).append({"type": "goal_set", "amount": target_amount, "months": months})
    return state

def set_income_expense(state: GraphState, income: float, fixed: float, variable_hint: Optional[float] = None) -> GraphState:
    state["income_expense"] = IncomeExpense(monthly_income=income, fixed_expense=fixed, variable_expense_hint=variable_hint)
    state.setdefault("events", []).append({"type": "profile_set", "income": income, "fixed": fixed})
    return state

def compute_investable(state: GraphState, percent: float) -> GraphState:
    ie: IncomeExpense = state.get("income_expense")  # type: ignore
    if not ie:
        raise ValueError("Income/Expense not set")
    base = max(ie.monthly_income - ie.fixed_expense - (ie.variable_expense_hint or 0.0), 0.0)
    amount = round(base * (percent / 100.0), 2)
    state["investable"] = Investable(percent_input=percent, amount=amount, rationale="auto-derived from profile")
    state["ui_step"] = UIStep.ASK_HEART
    state.setdefault("events", []).append({"type": "investable_computed", "percent": percent, "amount": amount})
    return state

def set_risk_preference(state: GraphState, risk: RiskLevel) -> GraphState:
    state["risk_preference"] = risk
    state["ui_step"] = UIStep.SHOW_PORTFOLIOS
    state.setdefault("events", []).append({"type": "risk_set", "risk": risk})
    return state

def set_candidates(state: GraphState, portfolios: List[Portfolio], rag_trace: Optional[RAGTrace] = None) -> GraphState:
    state["candidate_portfolios"] = portfolios
    if rag_trace:
        state["rag_trace"] = rag_trace
    state["ui_step"] = UIStep.SHOW_PORTFOLIOS
    state.setdefault("events", []).append({"type": "candidates_ready", "count": len(portfolios)})
    return state

def select_portfolio(state: GraphState, portfolio_id: str) -> GraphState:
    cands: List[Portfolio] = state.get("candidate_portfolios", [])
    found = next((p for p in cands if p.id == portfolio_id), None)
    if not found:
        raise ValueError(f"portfolio {portfolio_id} not found")
    state["selected_portfolio"] = found
    state["ui_step"] = UIStep.CONFIRM_PORTFOLIO
    state.setdefault("events", []).append({"type": "portfolio_selected", "id": portfolio_id})
    return state

def set_rebalance_plan(state: GraphState, plan: RebalancePlan) -> GraphState:
    state["rebalance_plan"] = plan
    state["ui_step"] = UIStep.REBALANCING
    state.setdefault("events", []).append({"type": "rebalance_ready", "actions": len(plan.actions)})
    return state

def append_news_signals(state: GraphState, signals: List[NewsSignal], cursor: Optional[str] = None) -> GraphState:
    state.setdefault("news_signals", []).extend(signals)
    if cursor:
        state["news_last_cursor"] = cursor
    # 의사결정 요약
    neg = [s for s in signals if s.sentiment == Sentiment.NEGATIVE]
    state.setdefault("events", []).append({"type": "news_ingested", "count": len(signals), "negatives": len(neg)})
    return state

def decide_from_news(state: GraphState) -> GraphState:
    # 간단한 규칙: 음성 기사 비중이 높으면 추천 재시작
    signals: List[NewsSignal] = state.get("news_signals", [])
    if not signals:
        return state
    neg_ratio = sum(s.sentiment == Sentiment.NEGATIVE for s in signals) / len(signals)
    decision = NewsDecision.RECOMMEND_REFRESH if neg_ratio >= 0.4 else NewsDecision.REVIEW
    state.setdefault("events", []).append({"type": "news_decision", "neg_ratio": round(neg_ratio, 2), "decision": decision})
    return state
