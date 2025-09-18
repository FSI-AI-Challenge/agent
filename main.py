from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import interrupt, Command
from langchain_ollama import ChatOllama
from langgraph.types import Command
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from utils.state import *
from utils.node import *
from langgraph.types import Command

llm = ChatOllama(model='gpt-oss:20b')

graph = StateGraph(GraphState)

# 승현
graph.add_node("planner", planner)
graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("hitl_confirm_input", hitl_confirm_input)
# 주엽
graph.add_node("select_fin_prdt", select_fin_prdt)
graph.add_node("select_stock_products", select_stock_products)
graph.add_node("build_indicators", build_indicators)
graph.add_node("build_portfolios", build_portfolios)
# 지수
graph.add_node("crawl_news", crawl_news)
graph.add_node("summarize_news", summarize_news)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("evaluate_rebalance", evaluate_rebalance)

graph.set_entry_point("planner")
graph.add_conditional_edges(
    "planner",
    lambda s: s.get("route", "chatbot"),
    {
        "get_goal":"get_goal",
        "chatbot":"chatbot"
    }
)
graph.add_edge("get_goal", "load_profile")
graph.add_edge("load_profile", 'hitl_confirm_input')
graph.add_edge("hitl_confirm_input", "select_fin_prdt")
graph.add_edge("select_fin_prdt", "select_stock_products")
graph.add_edge("select_stock_products", "build_indicators")
graph.add_edge("build_indicators", "build_portfolios")
graph.add_edge("build_portfolios", END)
graph.add_edge("chatbot", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    KST = timezone(timedelta(hours=9))

    user_id = 1
    target_amount = 6000000
    target_months = 12
    created_ts = datetime.now(KST).isoformat()
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

    events = app.stream(GraphState(
        user_id=user_id,
        created_ts=created_ts,
        question=f"나는 {target_months}개월 안에 {target_amount}원을 모으고 싶어. 어떻게 해야 할까?",
        months_passed=0,
        investable_amount=500000,
    ), config=config)

    for event in events:
        print(event)

    # 사용자에게 input 받아야 하는 부분
    user_decision = {"target_amount": target_amount, "target_months": target_months, "investable_amount": 500000}
    for ev in app.stream(Command(resume=user_decision), config=config):
        print(ev)

    # 사용자에게 input 받아야 하는 부분
    user_decision = {"stock_allocation_pct": 30}
    for ev in app.stream(Command(resume=user_decision), config=config):
        print(ev)
        
    print(app.get_state(config))