from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import interrupt, Command
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from langchain_ollama import ChatOllama
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
    fixed_income: int                  # 월급(세후 등 기준 통일)
    fixed_expense: int                   # 고정지출(월)

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
    investable_amount: int
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

def start(state:GraphState) -> GraphState:
    return GraphState()

def chatbot(state:GraphState) -> GraphState:
    print("챗봇 시작")
    question = state['question']
    response = llm.invoke(question)
    print(f"챗봇 종료: {response.content}")
    return GraphState(answer=response.content)

def get_goal(state:GraphState) -> GraphState:
    print("목표 금액, 기간 추출 시작")
    question = state['question']

    prompt = f"""
        Extract the user's savings goal.
        Return ONLY valid JSON, no markdown, no code block.
        Keys:
        - target_amount (int)
        - target_months (int)
        User input: {question}
    """

    response = llm.invoke(prompt)
    data = json.loads(response.content)
    
    print(f"목표 금액, 기간 추출 종료: {data}")
    return GraphState(
        goal=Goal(
            target_amount=int(data["target_amount"]), 
            target_months=int(data["target_months"])
        )
    )

def load_profile(state:GraphState) -> GraphState:
    print("사용자 수입 및 지출 계산 시작")
    # 추후 api로 대체
    user_id = state['user_id']
    with open(f"./data/user_example_data{user_id}.json", "r", encoding="utf-8") as f:
        user_profile = json.load(f)

    ts = datetime.fromisoformat(state["created_ts"])
    current_year = ts.year
    current_month = ts.month

    recent_months = []
    for i in range(3):
        month = (current_month - i - 1) % 12 + 1
        year = current_year if current_month - i > 0 else current_year - 1
        recent_months.append(f"{year}-{month:02d}")

    recent_months = set(recent_months)

    filtered_months = [
        m for m in user_profile["months"] if m["month"] in recent_months
    ]

    template = '''
        You are an assistant that analyzes personal finance transaction data.

        Tasks:
        1) Identify the average fixed income per month.
        2) Identify the average fixed expenses per month.
        3) Identify the average variable expenses per month.
        4) Compute the average investable amount per month = fixed_income - (fixed_expenses + variable_expenses).

        Return ONLY valid JSON, no markdown, no code block. 
        Keys:
        - fixed_income (int)
        - fixed_expenses (int)
        - variable_expenses (int)
        - investable_amount (int)

        User Input:
        {}
    '''
    prompt = template.format(filtered_months)

    response = llm.invoke(prompt)
    print(response)
    data = json.loads(response.content)
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"사용자 수입 및 지출 계산 종료: {data}")
    return GraphState(
        investable_amount=data["investable_amount"]
    )

def hitl_confirm_input(state:GraphState) -> GraphState:
    print("사용자 입력 검증 시작")
    proposed = {
        "target_amount": state["goal"].target_amount,
        "target_months": state["goal"].target_months,
        "investable_amount": state["investable_amount"]
    }
    
    decision = interrupt({
        "step": "confirm_input",
        "message": "목표 금액/기간, 투자 가능 금액을 확인 및 수정해주세요.",
        "proposed": proposed,
        "fields": [
            {"name": "target_amount", "type": "number", "label": "목표 금액(원)"},
            {"name": "target_months", "type": "number", "label": "목표 기간(개월)"},
            {"name": "investable_amount", "type": "number", "label": "투자 가능 금액(원)"},
        ],
        "buttons": ["submit"]
    })
    
    target_amount = int(decision.get("target_amount", proposed["target_amount"]))
    target_months = int(decision.get("target_months", proposed["target_months"]))
    investable_amount = int(decision.get("investable_amount", proposed["investable_amount"]))

    if target_amount < 0 or target_months < 0 or investable_amount < 0:
        raise ValueError("입력 값이 유효하지 않습니다.")

    print(f"사용자 입력 검증 종료: {target_amount, target_months, investable_amount}")
    return GraphState(
        goal=Goal(
            target_amount=int(decision["target_amount"]),
            target_months=int(decision["target_months"]),
        ),
        investable_amount=decision["investable_amount"]
    )

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

def is_our_service(state:GraphState) -> str:
    print("서비스 판단 시작")
    question = state['question']

    prompt = f'''
        You are a classifier. Decide whether the user's message expresses intent to use a savings or investment guidance service.
        Answer only yes or no.

        User Input: {question}
    '''
    response = llm.invoke(prompt)
    print(f"서비스 판단 종료: {response.content}")
    return response.content

def is_goal_reached(state:GraphState):
    return "yes"

def is_rebalance_needed(state:GraphState):
    return "yes"

from langchain_teddynote.graphs import visualize_graph

graph = StateGraph(GraphState)

graph.add_node("start", start)
graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("hitl_confirm_input", hitl_confirm_input)
graph.add_node("get_percent", get_percent)
graph.add_node("retrieve_products", retrieve_products)
graph.add_node("select_products", select_products)
graph.add_node("build_indicates", build_indicates)
graph.add_node("build_portfolios", build_portfolios)
graph.add_node("crawl_news", crawl_news)
graph.add_node("summarize_news", summarize_news)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("evaluate_rebalance", evaluate_rebalance)

graph.set_entry_point("start")
graph.add_conditional_edges(
    "start",
    is_our_service,
    {
        "yes":"get_goal",
        "no":"chatbot"
    }
)
graph.add_edge("get_goal", "load_profile")
graph.add_edge("load_profile", 'hitl_confirm_input')
graph.add_edge("hitl_confirm_input", END)
graph.add_edge("chatbot", END)

# graph.set_entry_point("start")
# graph.add_conditional_edges(
#     "start",
#     is_our_service,
#     {
#         "yes":"get_goal",
#         "no":"chatbot"
#     }
# )
# graph.add_edge("get_goal", "load_profile")
# graph.add_edge("load_profile", "hitl_confirm_input")
# graph.add_edge("hitl_confirm_input", "get_percent")
# graph.add_edge("get_percent", "retrieve_products")
# graph.add_edge("retrieve_products", "select_products")
# graph.add_edge("select_products", "build_indicates")
# graph.add_edge("build_indicates", "build_portfolios")
# graph.add_edge("build_portfolios", "crawl_news")
# graph.add_edge("crawl_news", "summarize_news")
# graph.add_edge("summarize_news", "analyze_sentiment")
# graph.add_edge("analyze_sentiment", "evaluate_rebalance")
# graph.add_conditional_edges(
#     "evaluate_rebalance",
#     is_rebalance_needed,
#     {
#         "yes":"crawl_news",
#         "no":"start"
#     }
# )
# graph.add_edge("chatbot", "start")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
# visualize_graph(app)

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

KST = timezone(timedelta(hours=9))

user_id = 1
target_amount = 1000000
target_months = 6
created_ts = datetime.now(KST).isoformat()

config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

inputs = GraphState(user_id=user_id, created_ts=created_ts, 
                    question=f"내 목표 금액은 {target_amount}이고, {target_months}개월 동안 모을거야.")
# inputs = GraphState(user_id=user_id, created_ts=created_ts, 
#                     question=f"예금과 주식의 차이가 뭐야.")

result = app.invoke(inputs, config)
print(result)

target_amount = result['goal'].target_amount
target_months = result['goal'].target_months
investable_amount = result['investable_amount'] - 100000

result = app.invoke(Command(resume={"target_amount":target_amount, "target_months":target_months, "investable_amount":investable_amount}), config)
print(result)