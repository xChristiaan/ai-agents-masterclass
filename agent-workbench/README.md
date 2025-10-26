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

## Getting started

1. **Install dependencies**

   ```bash
   cd agent-workbench
   pip install -r requirements.txt
   ```

2. **Configure API keys**

   The app uses OpenAI compatible chat models. Create a `.env` file based on `.env.example` in this
   directory and provide the relevant credentials.

   ```bash
   cp .env.example .env
   ```

   At a minimum you must set `OPENAI_API_KEY`. To use a fully managed Azure/OpenAI-compatible
   endpoint you can also fill out the optional settings in the file.

3. **Launch the workbench**

   ```bash
   streamlit run app.py
   ```

   Once the app is running you can open the provided local URL in your browser, design agents, and
   run automation sessions with them.

## Configuration files

- `agents.template.json` contains an example automation strategist agent. When you run the app for
  the first time it will create `agents.json` by copying this template. Subsequent edits are written
  back to `agents.json` so you can version control your agent catalog.
- `.env.example` documents the environment variables consumed by the app.

## Development notes

- The default OpenAI model list includes `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, and
  `gpt-3.5-turbo`. You can extend the list or adapt the API client inside `app.py` if your
  automation stack uses a different provider.
- When running conversations, you can reset the session history with the **Reset chat** button to
  start a fresh mission for the selected agent.
- The application keeps per-agent chat history in memory only. If you refresh the page the chat
  resets but the agent definitions remain on disk.

Happy building!
