# Agent Workbench

Agent Workbench is a Streamlit application that lets you design reusable AI agents for automation
projects, organize their playbooks, and execute them in an interactive workspace. The goal of the
app is to provide a lightweight command center for an AI automation agency—you can capture the
persona, goals, operating procedures, and tooling expectations for each agent, then collaborate with
them in a single dashboard.

## Features

- **Agent designer** – Create, edit, and delete agent profiles that include mission, goals,
  operating procedures, and toolbelt descriptions.
- **Persistent library** – Agent definitions are stored locally in `agents.json` so you can build a
  catalog that stays with the repo.
- **Mission control** – Run live conversations with any agent and keep the dialog history scoped to
  that agent.
- **Action planning** – Ask an agent for a structured automation playbook before starting execution
  to translate business objectives into concrete steps.
- **Flexible model backends** – Switch between hosted OpenAI/Azure endpoints and local Ollama
  models for cost-effective experimentation.

## Getting started

1. **Install dependencies**

   ```bash
   cd agent-workbench
   pip install -r requirements.txt
   ```

2. **Configure model access**

   Create a `.env` file based on `.env.example` in this directory and provide the relevant
   credentials or local settings.

   ```bash
   cp .env.example .env
   ```

   - **Hosted APIs (OpenAI compatible)** – Set `OPENAI_API_KEY`. Optionally override
     `OPENAI_BASE_URL` to point at another compatible provider.
   - **Azure OpenAI** – Fill in `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
     `AZURE_OPENAI_DEPLOYMENT`, and (optionally) `AZURE_OPENAI_API_VERSION`.
   - **Local models** – Install and run [Ollama](https://ollama.com), ensure the
     [`ollama`](https://pypi.org/project/ollama/) Python package is installed (it is included in
     `requirements.txt`), and optionally set `OLLAMA_HOST` if the server is not running on the
     default `http://127.0.0.1:11434`.

3. **Launch the workbench**

   ```bash
   streamlit run app.py
   ```

   Once the app is running you can open the provided local URL in your browser, design agents, and
   run automation sessions with them.

### Using local models

When editing or creating an agent, choose **Local (Ollama / LM Studio)** as the provider in the
sidebar. Provide the local model name (for example `llama3:8b`) and ensure the Ollama server is
running with that model pulled. Conversations and action plans will then use your local inference
endpoint instead of a paid API. If you prefer LM Studio or another OpenAI-compatible local server,
leave the provider set to **OpenAI (hosted)** and point `OPENAI_BASE_URL` at the local endpoint.

## Configuration files

- `agents.template.json` contains an example automation strategist agent. When you run the app for
  the first time it will create `agents.json` by copying this template. Subsequent edits are written
  back to `agents.json` so you can version control your agent catalog.
- `.env.example` documents the environment variables consumed by the app.

## Development notes

- You can extend the provider/model suggestions in `PROVIDER_SETTINGS` within `app.py` if your
  automation stack relies on additional APIs or self-hosted models.
- When running conversations, you can reset the session history with the **Reset chat** button to
  start a fresh mission for the selected agent.
- The application keeps per-agent chat history in memory only. If you refresh the page the chat
  resets but the agent definitions remain on disk.

Happy building!
