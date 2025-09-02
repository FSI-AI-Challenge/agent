from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

from datetime import datetime, timezone, timedelta

from utils.node import *
from utils.state import *
from utils.tools import *

llm = ChatOllama(model='gpt-oss:20b', streaming=True)

graph = StateGraph(GraphState)

graph.add_node("planner", planner)
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


memory = MemorySaver()
app = graph.compile(checkpointer=memory)


KST = timezone(timedelta(hours=9))

user_id = 1
target_amount = 1000000
target_months = 6
created_ts = datetime.now(KST).isoformat()

config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

inputs = GraphState(user_id=user_id, created_ts=created_ts, 
                    question=f"내 목표 금액은 {target_amount}이고, {target_months}개월 동안 모을거야.")
