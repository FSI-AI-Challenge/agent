from datetime import datetime, timedelta, timezone
from utils.state import GraphState, Goal
import json
from main import llm, interrupt

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