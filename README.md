# AgentFlow

A chatbot with agentic features.

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Set `GEMINI_API_KEY` in your environment or a local `.env` file.
4. Start the app with `bash run-local.sh`.
5. If you want terminal chat instead of the UI, run `bash run-local.sh --cli`.
6. If the package is installed, you can also use `agentflow chat`, `agentflow ask "Hello"`, or `agentflow run flow.yaml --input "Hello"`.

## How It Works

The launcher is a small shell wrapper around the Python CLI:

1. Without flags, it runs `python -m agentflow.cli ui`.
2. With `--cli`, it runs `python -m agentflow.cli chat`.
3. Both paths share the same chat logic in [backend.py](backend.py).

## Features

1. Streamlit web chat UI.
2. Interactive CLI chat mode.
3. PDF upload and thread-specific retrieval.
4. SQLite-backed conversation history (`chat_history.db`).
5. Tool-powered actions including search, calculator, stock lookup, and photo editing.

## Photo Editing Setup

1. Set `GOOGLE_CLOUD_PROJECT` to your Google Cloud project.
2. Set `GOOGLE_CLOUD_LOCATION` to a Vertex region such as `us-central1`.
3. Ensure application default credentials are available for Vertex AI.

## Notes

Keep `chat_history.db` in place if you want chat history to persist locally.
