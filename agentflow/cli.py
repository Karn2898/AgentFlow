from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

from langchain_core.messages import HumanMessage


_BACKEND_MODULE = None


def _load_backend_module():
    global _BACKEND_MODULE

    if _BACKEND_MODULE is not None:
        return _BACKEND_MODULE

    backend_path = Path.cwd() / "backend.py"
    if backend_path.exists():
        spec = importlib.util.spec_from_file_location(
            "agentflow_runtime_backend", backend_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load backend module from {backend_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BACKEND_MODULE = module
        return module

    import backend as module  # type: ignore[import-not-found]

    _BACKEND_MODULE = module
    return module


def _invoke_chatbot(prompt: str, thread_id: str) -> str:
    chatbot = _load_backend_module().chatbot
    response = chatbot.invoke(
        {"thread_id": thread_id, "messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return str(response["messages"][-1].content)


def _load_flow_config(flow_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for `agentflow run`. Install dependencies first."
        )

    if not flow_path.exists():
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    loaded = yaml.safe_load(flow_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Flow file must contain a YAML mapping at the top level.")
    return loaded


def command_run(flow_file: str, prompt: str | None, thread_id: str | None) -> int:
    flow_path = Path(flow_file)
    flow_config = _load_flow_config(flow_path)

    final_prompt = prompt or flow_config.get("input") or flow_config.get("prompt")
    if not final_prompt:
        raise ValueError("Provide --input or set `input`/`prompt` in the flow file.")

    final_thread_id = thread_id or str(flow_config.get("thread_id", "1"))
    print(_invoke_chatbot(str(final_prompt), final_thread_id))
    return 0


def command_ask(prompt: str, thread_id: str) -> int:
    print(_invoke_chatbot(prompt, thread_id))
    return 0


def command_chat(thread_id: str) -> int:
    while True:
        user_message = input("type here.. ")

        if user_message.strip().lower() in {"exit", "quit", "bye"}:
            break

        print("user:", user_message)
        print("chatbot:", _invoke_chatbot(user_message, thread_id))

    return 0


def command_ui() -> int:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streaming.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.headless=true",
        ],
        check=False,
    )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentflow")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a flow from a YAML file.")
    run_parser.add_argument("flow_file", help="Path to the YAML flow definition.")
    run_parser.add_argument(
        "--input",
        dest="prompt",
        help="Override the prompt from the flow file.",
    )
    run_parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread ID to use for checkpointing.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a single question.")
    ask_parser.add_argument("prompt", help="Prompt to send to the chatbot.")
    ask_parser.add_argument(
        "--thread-id",
        default="1",
        help="Conversation thread ID to use for checkpointing.",
    )

    chat_parser = subparsers.add_parser("chat", help="Open an interactive CLI chat.")
    chat_parser.add_argument(
        "--thread-id",
        default="1",
        help="Conversation thread ID to use for checkpointing.",
    )

    subparsers.add_parser("ui", help="Start the Streamlit UI.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return command_run(args.flow_file, args.prompt, args.thread_id)
    if args.command == "ask":
        return command_ask(args.prompt, args.thread_id)
    if args.command == "chat":
        return command_chat(args.thread_id)
    if args.command == "ui":
        return command_ui()

    return command_chat("1")


if __name__ == "__main__":
    raise SystemExit(main())
