import os
from typing import Annotated, TypedDict
from pathlib import Path

from google import genai as gemini
from google.genai.errors import APIError
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import sqlite3
import requests


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-flash-latest",
    ).split(",")
    if model.strip()
]

#tools
search_tool= DuckDuckGoSearchRun(region='us-en')
@tool
def calculator(first_num:float,second_num:float ,operation:str)->dict:
    """Perform a basic arithmetic operation on two numbers and return a result payload."""
    try:
        if operation=='add':
            result=first_num+second_num
        elif operation=='sub':
            result=first_num-second_num
        elif operation=='mul':
            result=first_num*second_num
        elif operation=='div':
            return{'error':'division by zero is not allowed'}
            result=first_num/second_num if second_num!=0 else None
        else: 
            return{'error':'invalid operation'}
        return {'first_num': first_num,'second_num':second_num,'operation':operation,'result':result}
    except Exception as e:
        return {'error': str(e)}
    
@tool
def get_stock_price(symbol:str)->dict:
    """Fetch the latest stock quote payload for a given ticker symbol from Alpha Vantage."""
    url='https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=' + symbol + '&apikey=3W9II91AT919POF6'
    r=requests.get(url)
    return r.json()
tools=[search_tool, get_stock_price, calculator]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_client() -> gemini.Client | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return gemini.Client(api_key=api_key)


def _generate_with_fallback(client: gemini.Client, prompt: str):
    candidates = [CHAT_MODEL, *FALLBACK_MODELS]
    seen: set[str] = set()
    last_exc: APIError | None = None

    for model_name in candidates:
        if model_name in seen:
            continue
        seen.add(model_name)

        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        except APIError as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("No Gemini model candidates configured.")


def chat_node(state: ChatState):
    client = _build_client()
    if client is None:
        return {
            "messages": [
                AIMessage(content="Set GEMINI_API_KEY to enable chatbot responses.")
            ]
        }

    messages = state["messages"]
    prompt = "\n".join(str(m.content) for m in messages)

    try:
        response = _generate_with_fallback(client, prompt)
    except APIError as exc:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Gemini API error with model '{CHAT_MODEL}': {exc}. "
                        "Try again, or set GEMINI_MODEL/GEMINI_FALLBACK_MODELS to supported models."
                    )
                )
            ]
        }

    return {"messages": [AIMessage(content=response.text)]}

conn=sqlite3.connect("chat_history.db",check_same_thread=False)


checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    while True:
        user_message = input("type here..")
        print("user:", user_message)

        if user_message.strip().lower() in ["exit", "quit", "bye"]:
            break

        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": "1"}},
        )["messages"][-1].content
        print("chatbot:", response)


def retrieve_all_threads() -> list[str]:
    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)