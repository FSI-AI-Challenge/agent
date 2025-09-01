from datetime import datetime, timedelta, timezone
from utils.state import *
import json

from main import llm, interrupt
from utils.state import *
from utils.tools import *

from langchain_core.messages import SystemMessage, HumanMessage

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

def retrieve_deposit_products(state:GraphState):
    financial_products = pd.read_csv('../data/financial_products.csv')

    top_5_products = financial_products[financial_products["save_trm"]<=12].sort_values("intr_rate", ascending=False).head(5)

    top_5_products = top_5_products[["kor_co_nm", "fin_prdt_nm", "max_limit", "intr_rate_type_nm", "save_trm", "intr_rate"]].to_dict(orient='records')

    system = SystemMessage(content=(
        "너는 예금/적금 상품 전문가야. "
        "아래 후보 중 '금리가 높고' 그리고 'etc_notes에 우대/혜택이 있는' 상품을 1개 고른다. "
        "같은 금리라면 복리 우선, 우대조건(우대금리, 자동이체, 급여이체, 비대면/모바일, 주거래, 청년, 마이데이터, 세금우대 등) 있으면 가점. "
        "최종 출력은 오직 JSON 한 개 객체만. 다른 텍스트 금지."
    ))
    user = HumanMessage(content=(
        "후보 리스트는 다음과 같아:\n"
        f"{json.dumps(top_5_products, ensure_ascii=False, indent=2)}\n\n"
        "아래 JSON 스키마에 정확히 맞춰 1개만 반환해줘.\n"
        "스키마: {\n"
        '  "kor_co_nm": str,\n'
        '  "fin_prdt_nm": str,\n'
        '  "max_limit": int,\n'
        '  "intr_rate_type_nm": "단리" | "복리",\n'
        '  "save_trm": int,\n'
        '  "intr_rate": float,\n'
        '  "etc_notes": str | null\n'
        "}\n"
        "반드시 키 이름/타입을 정확히 지켜줘."
    ))

    resp = llm.invoke([system, user])
    picked_raw = extract_json(resp.content)

    def _to_int(v): 
        return int(v) if v is not None and str(v).strip() != "" else 0
    def _to_float(v):
        return float(v) if v is not None and str(v).strip() != "" else 0.0
    def _to_str(v):
        return None if v is None else str(v)

    selected = SelectedFinPrdt(
        kor_co_nm=_to_str(picked_raw.get("kor_co_nm", "")) or "",
        fin_prdt_nm=_to_str(picked_raw.get("fin_prdt_nm", "")) or "",
        max_limit=_to_int(picked_raw.get("max_limit", 0)),
        intr_rate_type_nm=_to_str(picked_raw.get("intr_rate_type_nm", "")) or "",
        save_trm=_to_int(picked_raw.get("save_trm", 0)),
        intr_rate=_to_float(picked_raw.get("intr_rate", 0.0)),
        etc_notes=_to_str(picked_raw.get("etc_notes", None)),
    )

    return {**state, "selected_fin_prdt": selected}

def select_deposit_products(state:GraphState):
    return GraphState()

def retrieve_stock_products(state:GraphState):
    return GraphState()

def select_stock_products(state:GraphState):
    return GraphState()

def build_indicators(state:GraphState):
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