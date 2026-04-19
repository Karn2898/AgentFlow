import os
from typing import Annotated, TypedDict

from google import genai as gemini
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_client() -> gemini.Client | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return gemini.Client(api_key=api_key)


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

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return {"messages": [AIMessage(content=response.text)]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpointer = MemorySaver()
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

