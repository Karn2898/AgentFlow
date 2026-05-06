import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from pathlib import Path

from google import genai as gemini
from google.genai.errors import APIError
from google.genai import types
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

IMAGE_EDIT_MODEL = os.getenv("VERTEX_IMAGE_EDIT_MODEL", "imagen-3.0-capability-001")
VERTEX_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    or os.getenv("GOOGLE_VERTEX_PROJECT")
)
VERTEX_LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("GOOGLE_CLOUD_REGION")
    or os.getenv("GOOGLE_VERTEX_LOCATION")
    or "us-central1"
)

#pdf retiriever store
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_THREAD_IMAGES: Dict[str, Dict[str, dict[str, Any]]] = {}

def get_retriever(thread_id: str):
    """Get the PDF retriever for a given thread ID."""
    return _THREAD_RETRIEVERS.get(thread_id)
    
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """Ingest a PDF file and register metadata for it, associated with the given thread ID."""
    if not file_bytes:
        raise ValueError("No file bytes provided for ingestion.")

    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        _THREAD_RETRIEVERS[thread_id] = {"pdf_path": temp_path}
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "bytes": len(file_bytes),
        }

        return {
            "filename": _THREAD_METADATA[thread_id]["filename"],
            "bytes": _THREAD_METADATA[thread_id]["bytes"],
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def ingest_image(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """Ingest an uploaded image for a chat thread."""
    if not file_bytes:
        raise ValueError("No image bytes provided for ingestion.")

    thread_images = _THREAD_IMAGES.setdefault(thread_id, {})
    key = filename or "uploaded-image"
    existing_image = thread_images.get(key, {})
    thread_images[key] = {
        **existing_image,
        "filename": key,
        "bytes": file_bytes,
        "mime_type": _guess_mime_type(filename),
    }

    return {
        "filename": key,
        "bytes": len(file_bytes),
        "mime_type": thread_images[key]["mime_type"],
    }


def _guess_mime_type(filename: Optional[str]) -> str:
    if not filename:
        return "image/png"

    lower_name = filename.lower()
    if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
        return "image/jpeg"
    if lower_name.endswith(".webp"):
        return "image/webp"
    return "image/png"


def get_image(thread_id: str, filename: Optional[str] = None):
    """Get an uploaded image for a thread."""
    thread_images = _THREAD_IMAGES.get(thread_id, {})
    if not thread_images:
        return None

    if filename and filename in thread_images:
        return thread_images[filename]

    latest_key = list(thread_images.keys())[-1]
    return thread_images[latest_key]


def _get_image_edit_source(thread_id: str, filename: Optional[str] = None) -> tuple[dict[str, Any] | None, bytes | None]:
    image_asset = get_image(thread_id, filename=filename)
    if image_asset is None:
        return None, None

    source_bytes = image_asset.get("edited_bytes") or image_asset.get("current_bytes") or image_asset.get("bytes")
    if source_bytes is None:
        return image_asset, None

    return image_asset, source_bytes


def _build_vertex_client() -> gemini.Client | None:
    if not VERTEX_PROJECT:
        return None
    return gemini.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)


def edit_image(thread_id: str, instruction: str, filename: Optional[str] = None) -> dict[str, Any]:
    """Send the uploaded image and instruction to Vertex Imagen for editing."""
    image_asset, source_bytes = _get_image_edit_source(thread_id, filename=filename)
    if image_asset is None:
        return {
            "error": "No image indexed for this chat. Upload a photo first.",
            "instruction": instruction,
        }
    if source_bytes is None:
        return {
            "error": "The selected image has no bytes to edit.",
            "instruction": instruction,
        }

    client = _build_vertex_client()
    if client is None:
        return {
            "error": (
                "Vertex AI is not configured. Set GOOGLE_CLOUD_PROJECT and a Vertex location "
                "such as GOOGLE_CLOUD_LOCATION=us-central1, plus application default credentials."
            ),
            "instruction": instruction,
        }

    raw_ref_image = types.RawReferenceImage(
        reference_id=1,
        reference_image=types.Image(
            image_bytes=source_bytes,
            mime_type=image_asset["mime_type"],
        ),
    )

    try:
        response = client.models.edit_image(
            model=IMAGE_EDIT_MODEL,
            prompt=instruction,
            reference_images=[raw_ref_image],
            config=types.EditImageConfig(
                edit_mode=types.EditMode.EDIT_MODE_DEFAULT,
                number_of_images=1,
                output_mime_type="image/png",
                include_rai_reason=True,
            ),
        )
    except APIError as exc:
        return {
            "error": f"Vertex image editing failed: {exc}",
            "instruction": instruction,
            "model": IMAGE_EDIT_MODEL,
        }
    except Exception as exc:
        return {
            "error": f"Unexpected image editing failure: {exc}",
            "instruction": instruction,
            "model": IMAGE_EDIT_MODEL,
        }

    if not response.generated_images:
        return {
            "error": "The image editor returned no output.",
            "instruction": instruction,
        }

    generated_image = response.generated_images[0].image
    edited_bytes = generated_image.image_bytes or b""
    output_mime_type = generated_image.mime_type or "image/png"

    thread_images = _THREAD_IMAGES.setdefault(thread_id, {})
    stored_image = thread_images.setdefault(image_asset["filename"], {})
    stored_image.update(
        {
            "filename": image_asset["filename"],
            "bytes": image_asset.get("bytes", source_bytes),
            "mime_type": image_asset["mime_type"],
            "edited_bytes": edited_bytes,
            "current_bytes": edited_bytes,
            "current_mime_type": output_mime_type,
            "output_filename": "edited-image.png",
            "output_mime_type": output_mime_type,
            "output_bytes": len(edited_bytes),
            "model": IMAGE_EDIT_MODEL,
        }
    )

    return {
        "filename": image_asset["filename"],
        "instruction": instruction,
        "output_filename": "edited-image.png",
        "output_mime_type": output_mime_type,
        "output_bytes": len(edited_bytes),
        "edited_bytes": edited_bytes,
        "model": IMAGE_EDIT_MODEL,
    }


@tool
def edit_uploaded_image(thread_id: str, instruction: str, filename: Optional[str] = None) -> dict:
    """Edit an uploaded image for a chat thread and return the edited file payload."""
    return edit_image(thread_id=thread_id, instruction=instruction, filename=filename)


IMAGE_EDIT_KEYWORDS = {
    "edit",
    "photo",
    "image",
    "picture",
    "brighten",
    "darken",
    "grayscale",
    "gray",
    "black and white",
    "invert",
    "negative",
    "contrast",
    "sharpen",
    "blur",
    "rotate",
    "flip",
    "mirror",
    "background",
    "remove",
    "replace",
    "style",
    "make it",
    "change the",
}


def _looks_like_image_edit_request(prompt: str) -> bool:
    normalized = prompt.lower().strip()
    return any(keyword in normalized for keyword in IMAGE_EDIT_KEYWORDS)

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

@tool
def rag_tool(query:str,thread_id:Optional[str]=None)->dict:
    """
    retireve relevant information from the upoaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever=get_retriever(thread_id)
    if retriever is None:
        return {
            "error":"No document indexed for this chat . Upload a PDF first",
            "query":query,
        }

    result=retriever.invoke(query)
    context=[doc.page_content for doc in result]
    metadata=[doc.metadata for doc in result]

    return {
        "query":query,
        "context":context,
        "metadata":metadata,
    }

tools=[search_tool, get_stock_price, calculator,rag_tool]


class ImageEditState(TypedDict):
    thread_id: str
    instruction: str
    filename: Optional[str]
    result: dict[str, Any]


def image_edit_node(state: ImageEditState):
    result = edit_uploaded_image.invoke(
        {
            "thread_id": state["thread_id"],
            "instruction": state["instruction"],
            "filename": state.get("filename"),
        }
    )
    return {"result": result}


image_edit_graph = StateGraph(ImageEditState)
image_edit_graph.add_node("image_edit_node", image_edit_node)
image_edit_graph.add_edge(START, "image_edit_node")
image_edit_graph.add_edge("image_edit_node", END)
photo_editor = image_edit_graph.compile()


class ChatState(TypedDict):
    thread_id: Optional[str]
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
    thread_id = state.get("thread_id") or "1"
    messages = state["messages"]
    user_prompt = str(messages[-1].content) if messages else ""

    if _looks_like_image_edit_request(user_prompt):
        image_asset = get_image(thread_id)
        if image_asset is None:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Upload a photo first, then send your edit instruction again."
                        )
                    )
                ]
            }

        edit_result = photo_editor.invoke(
            {
                "thread_id": thread_id,
                "instruction": user_prompt,
                "filename": image_asset.get("filename"),
                "result": {},
            }
        )["result"]

        if edit_result.get("error"):
            return {"messages": [AIMessage(content=edit_result["error"])]}

        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Edited '{edit_result.get('filename')}' with {edit_result.get('model')}. "
                        f"Download the updated image from the sidebar."
                    )
                )
            ]
        }

    if client is None:
        return {
            "messages": [
                AIMessage(content="Set GEMINI_API_KEY to enable chatbot responses.")
            ]
        }

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
            {"thread_id": "1", "messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": "1"}},
        )["messages"][-1].content
        print("chatbot:", response)


def retrieve_all_threads() -> list[str]:
    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)