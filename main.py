from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import json

llm = ChatOllama(model='gpt-oss:20b')

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

@dataclass
class Goal:
    target_amount: int
    target_months: int         

@dataclass
class IncomeExpense:
    monthly_income: float                  # 월급(세후 등 기준 통일)
    fixed_expense: float                   # 고정지출(월)

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
    headline: str
    url: str
    published_at: str                      # ISO8601
    summary: str
    sentiment: Sentiment
    decision: NewsDecision | None = None   # 판단 결과

@dataclass
class RAGTrace:
    query: str
    topk: int
    ids: List[str]                         # 검색된 상품/문서 id
    notes: str | None = None

class GraphState(TypedDict):
    user_id: int
    created_ts: str
    question:str
    answer:str
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
    messages: Annotated[List, add_messages]

def chatbot(state:GraphState):
    return GraphState()

def get_goal(state:GraphState) -> GraphState:
    question = state['question']

    prompt = f"""
        Extract the user's savings goal.
        Return JSON with keys:
        - target_amount (int)
        - target_months (int)
        User input: {question}
    """

    response = llm.invoke(prompt)
    data = json.loads(response.content)
    
    return GraphState(
        goal=Goal(
            target_amount=int(data["target_amount"]), 
            target_months=int(data["target_months"])
        )
    )

def load_profile(state:GraphState):
    user_id = state['user_id']
    # tmp
    with open(f"./data/user_example_data{user_id}.json", "r", encoding="utf-8") as f:
        user_profile = json.load(f)

    incomes = [m["monthly_income"] for m in user_profile["user_profile"]]
    avg_income = sum(incomes) / len(incomes)

    # 고정지출 합계 리스트
    expenses = [sum(m["fixed_expenses"].values()) for m in user_profile["user_profile"]]
    avg_expense = sum(expenses) / len(expenses)

    return GraphState(
        income_expense=IncomeExpense(
            monthly_income=avg_income,
            fixed_expense=avg_expense
        )
    )

def calc_investable(state:GraphState):
    return GraphState()

def get_percent(state:GraphState):
    return GraphState()

def retrieve_products(state:GraphState):
    return GraphState()

def select_products(state:GraphState):
    return GraphState()

def build_indicates(state:GraphState):
    return GraphState()

def build_portfolios(state:GraphState):
    return GraphState()

def crawl_news(state:GraphState):
    return GraphState()

def summarize_news(state:GraphState):
    return GraphState()

def analyze_sentiment(state:GraphState):
    return GraphState()

def evaluate_rebalance(state:GraphState):
    return GraphState()
def is_goal_reached(state:GraphState):
    return "yes"

def is_rebalance_needed(state:GraphState):
    return "yes"
from langchain_teddynote.graphs import visualize_graph

graph = StateGraph(GraphState)

graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("calc_investable", calc_investable)
graph.add_node("get_percent", get_percent)
graph.add_node("retrieve_products", retrieve_products)
graph.add_node("select_products", select_products)
graph.add_node("build_indicates", build_indicates)
graph.add_node("build_portfolios", build_portfolios)
graph.add_node("crawl_news", crawl_news)
graph.add_node("summarize_news", summarize_news)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("evaluate_rebalance", evaluate_rebalance)

graph.set_entry_point("chatbot")
graph.add_edge("chatbot", "get_goal")
graph.add_edge("get_goal", END)
# graph.add_edge("get_goal", "load_profile")
# graph.add_edge("load_profile", "calc_investable")
# graph.add_edge("calc_investable", "chatbot")
# graph.add_edge("chatbot", "get_percent")
# graph.add_edge("get_percent", "retrieve_products")
# graph.add_edge("retrieve_products", "select_products")
# graph.add_edge("select_products", "build_indicates")
# graph.add_edge("build_indicates", "chatbot")
# graph.add_edge("chatbot", "build_portfolios")
# graph.add_edge("build_portfolios", "chatbot")
# graph.add_edge("chatbot", "crawl_news")
# graph.add_edge("crawl_news", "summarize_news")
# graph.add_edge("summarize_news", "analyze_sentiment")
# graph.add_edge("analyze_sentiment", "evaluate_rebalance")
# graph.add_conditional_edges(
#     "evaluate_rebalance",
#     is_rebalance_needed,
#     {
#         "yes":"crawl_news",
#         "no":"chatbot"
#     }
# )
# graph.add_conditional_edges(
#     "chatbot",
#     is_goal_reached,
#     {
#         "yes":END,
#         "no":"crawl_news"
#     }
# )
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
# visualize_graph(app)

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

KST = timezone(timedelta(hours=9))

target_amount = 1000000
target_months = 6

config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

inputs = GraphState(user_id=1, created_ts=datetime.now(KST).isoformat(), 
                    question=f"내 목표 금액은 {target_amount}이고, {target_months}개월 동안 모을거야.")

result = app.invoke(inputs, config)
print(result)