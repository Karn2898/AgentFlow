import uuid
import re
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend import chatbot, get_image, ingest_image, ingest_pdf, retrieve_all_threads


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


st.sidebar.title(" Random Chatbot")

st.sidebar.subheader("Thread ID")
st.sidebar.code(st.session_state["thread_id"])

if st.session_state.get("chat_threads"):
    with st.sidebar.expander("Saved thread IDs", expanded=False):
        for thread_id in st.session_state["chat_threads"]:
            st.write(thread_id)

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

st.sidebar.divider()
st.sidebar.subheader("Photo editor")

uploaded_image = st.sidebar.file_uploader(
    "Upload a photo for this chat",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_image:
    image_summary = ingest_image(
        uploaded_image.getvalue(),
        thread_id=thread_key,
        filename=uploaded_image.name,
    )
    st.sidebar.caption(f"Loaded {image_summary['filename']} for editing.")
    st.sidebar.info("Ask in chat to edit the uploaded photo.")

backend_image = get_image(thread_key, uploaded_image.name) if uploaded_image else None
if backend_image and backend_image.get("edited_bytes"):
    original_bytes = backend_image.get("bytes") or uploaded_image.getvalue()
    edited_bytes = backend_image["edited_bytes"]

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_bytes, caption="Original", use_container_width=True)
    with col2:
        st.image(
            edited_bytes,
            caption=f"Edited ({backend_image.get('model', 'vertex-image-edit')})",
            use_container_width=True,
        )

    st.download_button(
        "Download edited photo",
        data=edited_bytes,
        file_name=backend_image.get("output_filename", "edited-image.png"),
        mime=backend_image.get("output_mime_type", "image/png"),
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


user_input = st.chat_input("Type here")
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config = {
        "thread_id": st.session_state["thread_id"],
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata":{
            "thread_id":st.session_state["thread_id"]
             },
        "run_name":"chat iter "
        }

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, _metadata in chatbot.stream(
                {
                    "thread_id": st.session_state["thread_id"],
                    "messages": [HumanMessage(content=user_input)],
                },
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": str(ai_message)}
    )
