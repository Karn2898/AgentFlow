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

### Using Stability AI (optional fallback)

You can use Stability AI for image edits instead of Vertex AI. When a `STABILITY_API_KEY` is present in your environment (or in `.env`), the app will prefer Stability for image editing and fall back to Vertex when no key is available.

1. Obtain a Stability API key from https://platform.stability.ai/ and set `STABILITY_API_KEY` in your `.env` or shell.
2. Optionally set `STABILITY_MODEL` to your preferred model (default: `stable-diffusion-512-v2-1`).
3. Upload an image in the UI and ask an edit (e.g. "make it brighter").

Notes:
- The project currently loads local env vars from `.env` at startup. Do not commit `.env` to source control — rotate any keys that were accidentally exposed.
- If you want Vertex only, unset `STABILITY_API_KEY` and ensure `GOOGLE_CLOUD_PROJECT` and credentials are configured.

## Docker

Use Docker when you want a consistent environment without setting up Python locally.

1. Build and run the Streamlit app:
	`docker compose up --build`
2. Open the app at `http://localhost:8501`.
3. Stop it with `docker compose down`.
4. Pass `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` through your shell or a compose environment file before starting.
5. The container keeps `chat_history.db` mounted so chat history survives restarts.
6. The image starts the Streamlit UI by default through the `agentflow ui` command.

## Notes

Keep `chat_history.db` in place if you want chat history to persist locally.
