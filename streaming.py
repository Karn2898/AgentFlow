import uuid
import re
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend import chatbot, ingest_pdf, retrieve_all_threads


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list[dict[str, str]]:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    values = getattr(state, "values", {}) or {}
    messages = values.get("messages", [])

    history: list[dict[str, str]] = []
    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        history.append({"role": role, "content": str(message.content)})
    return history


def reset_chat() -> None:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def stream_word_by_word(text: str, delay: float = 0.03):
    for token in re.findall(r"\S+\s*", text):
        yield token
        time.sleep(delay)

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()



add_thread(st.session_state["thread_id"])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New chat"):
    reset_chat()
    st.rerun()

if "thread_docs" not in st.session_state:
    st.session_state["thread_docs"] = {}

thread_key = st.session_state["thread_id"]
if thread_key not in st.session_state["thread_docs"]:
    st.session_state["thread_docs"][thread_key] = {}

thread_docs = st.session_state["thread_docs"][thread_key]

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using '{latest_doc.get('filename')}' "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


user_input = st.chat_input("Type here")
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata":{
            "thread_id":st.session_state["thread_id"]
             },
        "run_name":"chat iter "
        }

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, _metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": str(ai_message)}
    )