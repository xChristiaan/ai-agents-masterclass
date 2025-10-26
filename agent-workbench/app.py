"""Streamlit workbench for designing and collaborating with automation agents."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
AGENT_STORE_PATH = APP_DIR / "agents.json"
AGENT_TEMPLATE_PATH = APP_DIR / "agents.template.json"

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
]


class AgentConfig(BaseModel):
    """Definition of an automation agent."""

    id: str
    name: str
    mission: str
    model: str = Field(default=MODEL_OPTIONS[0])
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: str
    core_objectives: List[str] = Field(default_factory=list)
    operating_procedures: List[str] = Field(default_factory=list)
    tool_belt: List[str] = Field(default_factory=list)
    default_playbook: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    def display_name(self) -> str:
        return self.name


def ensure_agent_store() -> None:
    """Create the agent store on first run."""

    if AGENT_STORE_PATH.exists():
        return

    if AGENT_TEMPLATE_PATH.exists():
        AGENT_STORE_PATH.write_text(AGENT_TEMPLATE_PATH.read_text())
    else:
        AGENT_STORE_PATH.write_text("[]\n")


def load_agents() -> List[AgentConfig]:
    ensure_agent_store()
    try:
        data = json.loads(AGENT_STORE_PATH.read_text())
    except json.JSONDecodeError:
        st.warning("Agent catalog could not be parsed; starting with an empty list.")
        data = []
    agents: List[AgentConfig] = []
    for raw_agent in data:
        try:
            agents.append(AgentConfig.model_validate(raw_agent))
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to load agent entry: {exc}")
    return agents


def save_agents(agents: List[AgentConfig]) -> None:
    payload = [agent.model_dump(mode="json") for agent in agents]
    AGENT_STORE_PATH.write_text(json.dumps(payload, indent=2))


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_multiline(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def build_system_prompt(agent: AgentConfig) -> str:
    sections: List[str] = [f"You are {agent.name}. {agent.mission.strip()}"]
    sections.append(agent.system_prompt.strip())

    if agent.core_objectives:
        objectives = "\n".join(f"- {objective}" for objective in agent.core_objectives)
        sections.append(f"Core objectives:\n{objectives}")

    if agent.operating_procedures:
        procedures = "\n".join(f"- {procedure}" for procedure in agent.operating_procedures)
        sections.append(f"Operating procedures:\n{procedures}")

    if agent.tool_belt:
        tools = "\n".join(f"- {tool}" for tool in agent.tool_belt)
        sections.append(
            "Available capabilities/tools (describe how you use each before acting):\n"
            f"{tools}"
        )

    if agent.default_playbook:
        steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(agent.default_playbook))
        sections.append(
            "Preferred default playbook when the user does not specify a plan:\n"
            f"{steps}"
        )

    sections.append(
        "Always respond with a consulting tone that prioritizes automation opportunities, next "
        "actions, and measurable outcomes."
    )
    return "\n\n".join(sections)


def get_openai_client(agent: AgentConfig) -> Dict[str, Optional[str]]:
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - Streamlit runtime
        raise RuntimeError(
            "The 'openai' package is required. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if azure_key and azure_endpoint and azure_deployment:
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base = azure_endpoint
        openai.api_key = azure_key
        return {"type": "azure", "deployment": azure_deployment, "client": openai}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    openai.api_key = api_key
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        openai.api_base = base_url

    openai.api_type = "open_ai"
    openai.api_version = None
    return {"type": "openai", "client": openai}


def call_model(agent: AgentConfig, messages: List[Dict[str, str]], max_tokens: int = 800) -> str:
    client_info = get_openai_client(agent)
    client = client_info["client"]

    kwargs = {
        "messages": messages,
        "temperature": agent.temperature,
        "max_tokens": max_tokens,
    }

    if client_info["type"] == "azure":
        kwargs["engine"] = client_info["deployment"]
    else:
        kwargs["model"] = agent.model

    try:
        response = client.ChatCompletion.create(**kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Model call failed: {exc}") from exc

    return response["choices"][0]["message"]["content"].strip()


def generate_action_plan(agent: AgentConfig, objective: str) -> str:
    guidance = (
        "Create a step-by-step automation plan. Include phases, responsible roles or agents, "
        "required integrations, and measurable outcomes. Close with the immediate next three "
        "actions the automation agency team should take."
    )
    messages = [
        {"role": "system", "content": build_system_prompt(agent)},
        {
            "role": "user",
            "content": (
                f"Client objective:\n{objective.strip()}\n\n{guidance}\n"
                "Format the response as Markdown with headings and bullet lists."
            ),
        },
    ]
    return call_model(agent, messages, max_tokens=900)


def render_agent_overview(agent: AgentConfig) -> None:
    st.subheader(agent.name)
    st.markdown(f"**Mission:** {agent.mission}")
    st.markdown(f"**Model:** `{agent.model}` @ temperature `{agent.temperature}`")

    if agent.core_objectives:
        st.markdown("### Core objectives")
        st.markdown("\n".join(f"- {item}" for item in agent.core_objectives))

    if agent.operating_procedures:
        st.markdown("### Operating procedures")
        st.markdown("\n".join(f"- {item}" for item in agent.operating_procedures))

    if agent.tool_belt:
        st.markdown("### Toolbelt")
        st.markdown("\n".join(f"- {item}" for item in agent.tool_belt))

    if agent.default_playbook:
        st.markdown("### Default playbook")
        st.markdown("\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(agent.default_playbook)))

    st.markdown("---")
    st.caption(
        f"Created {agent.created_at.strftime('%Y-%m-%d %H:%M')} | "
        f"Last updated {agent.updated_at.strftime('%Y-%m-%d %H:%M')}"
    )
    st.download_button(
        "Download agent JSON",
        data=json.dumps(agent.model_dump(mode="json"), indent=2),
        file_name=f"{agent.id}.json",
        mime="application/json",
    )


def render_action_planner(agent: AgentConfig) -> None:
    objective = st.text_area(
        "What automation objective should this agent plan for?",
        placeholder="Example: Build a lead-qualification workflow that routes hot leads to our sales reps",
        key=f"planner-objective-{agent.id}",
    )
    plan_key = f"planner-output-{agent.id}"

    if st.button("Generate action plan", key=f"planner-submit-{agent.id}"):
        if not objective.strip():
            st.warning("Describe the client's objective before generating a plan.")
        else:
            with st.spinner("Synthesizing automation roadmap..."):
                try:
                    plan = generate_action_plan(agent, objective)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[plan_key] = plan

    if plan_key in st.session_state:
        st.markdown(st.session_state[plan_key])


def render_chat(agent: AgentConfig) -> None:
    chat_key = f"chat-history-{agent.id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    if st.button("Reset chat", key=f"reset-chat-{agent.id}"):
        st.session_state.pop(chat_key, None)
        st.rerun()

    for message in st.session_state.get(chat_key, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask the agent to move your automation forward"):
        history = st.session_state.setdefault(chat_key, [])
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking like an automation expert..."):
                try:
                    response = call_model(
                        agent,
                        messages=[
                            {"role": "system", "content": build_system_prompt(agent)},
                            *st.session_state[chat_key],
                        ],
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    response = ""
            if response:
                st.markdown(response)
                history.append({"role": "assistant", "content": response})


def upsert_agent(
    *,
    existing: Optional[AgentConfig],
    name: str,
    mission: str,
    model: str,
    temperature: float,
    system_prompt: str,
    objectives_raw: str,
    procedures_raw: str,
    tools_raw: str,
    playbook_raw: str,
) -> AgentConfig:
    now = datetime.utcnow()
    agent_id = existing.id if existing else slugify(name) or f"agent-{now.strftime('%Y%m%d%H%M%S')}"
    if existing is None:
        created_at = now
    else:
        created_at = existing.created_at

    default_playbook = parse_multiline(playbook_raw) or None

    return AgentConfig(
        id=agent_id,
        name=name,
        mission=mission,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        core_objectives=parse_multiline(objectives_raw),
        operating_procedures=parse_multiline(procedures_raw),
        tool_belt=parse_multiline(tools_raw),
        default_playbook=default_playbook,
        created_at=created_at,
        updated_at=now,
    )


def render_agent_editor(agents: List[AgentConfig], selected_id: Optional[str]) -> Optional[AgentConfig]:
    mode = st.session_state.get("editor_mode")
    editor_agent_id = st.session_state.get("editor_agent_id")
    agent_lookup = {agent.id: agent for agent in agents}

    if mode not in {"create", "edit"}:
        return agent_lookup.get(selected_id)

    agent = agent_lookup.get(editor_agent_id) if mode == "edit" else None
    defaults = agent.model_dump() if agent else {}

    with st.sidebar.form("agent-editor", clear_on_submit=False):
        st.markdown(f"### {'Create' if agent is None else 'Edit'} agent")
        name = st.text_input("Name", value=defaults.get("name", ""))
        mission = st.text_input(
            "Mission",
            value=defaults.get("mission", ""),
        )
        model_default = defaults.get("model", MODEL_OPTIONS[0])
        if model_default not in MODEL_OPTIONS:
            model_default = MODEL_OPTIONS[0]
        model = st.selectbox("Model", options=MODEL_OPTIONS, index=MODEL_OPTIONS.index(model_default))
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults.get("temperature", 0.7)),
            step=0.05,
        )
        system_prompt = st.text_area(
            "System prompt",
            value=defaults.get(
                "system_prompt",
                (
                    "You are an automation specialist supporting a client-facing agency team."
                    " Always explain your reasoning before final recommendations."
                ),
            ),
            height=180,
        )
        objectives_raw = st.text_area(
            "Core objectives (one per line)",
            value="\n".join(defaults.get("core_objectives", [])),
            height=120,
        )
        procedures_raw = st.text_area(
            "Operating procedures (one per line)",
            value="\n".join(defaults.get("operating_procedures", [])),
            height=120,
        )
        tools_raw = st.text_area(
            "Toolbelt / capabilities (one per line)",
            value="\n".join(defaults.get("tool_belt", [])),
            height=120,
        )
        playbook_defaults = defaults.get("default_playbook") or []
        playbook_raw = st.text_area(
            "Default playbook steps (one per line)",
            value="\n".join(playbook_defaults),
            height=120,
        )

        col1, col2 = st.columns(2)
        save_clicked = col1.form_submit_button("Save agent", type="primary")
        cancel_clicked = col2.form_submit_button("Cancel", type="secondary")

    if cancel_clicked:
        st.session_state["editor_mode"] = None
        st.session_state["editor_agent_id"] = None
        st.session_state["status_message"] = ("info", "Agent editing cancelled.")
        st.rerun()

    if save_clicked:
        if not name.strip():
            st.sidebar.error("Name is required.")
            return agent_lookup.get(selected_id)
        if not mission.strip():
            st.sidebar.error("Mission is required.")
            return agent_lookup.get(selected_id)
        if not system_prompt.strip():
            st.sidebar.error("System prompt is required.")
            return agent_lookup.get(selected_id)

        updated_agent = upsert_agent(
            existing=agent,
            name=name.strip(),
            mission=mission.strip(),
            model=model,
            temperature=temperature,
            system_prompt=system_prompt.strip(),
            objectives_raw=objectives_raw,
            procedures_raw=procedures_raw,
            tools_raw=tools_raw,
            playbook_raw=playbook_raw,
        )

        next_agents = [a for a in agents if a.id != updated_agent.id]
        next_agents.append(updated_agent)
        next_agents.sort(key=lambda item: item.name.lower())
        save_agents(next_agents)
        st.session_state["editor_mode"] = None
        st.session_state["editor_agent_id"] = None
        st.session_state["status_message"] = ("success", f"Agent '{updated_agent.name}' saved.")
        st.session_state["selected_agent_id"] = updated_agent.id
        st.rerun()

    return agent_lookup.get(selected_id)


def main() -> None:
    st.set_page_config(page_title="Agent Workbench", layout="wide")
    st.title("Agent Workbench")
    st.caption("Design, organize, and run automation agents for your AI agency.")

    if "status_message" in st.session_state:
        status_type, message = st.session_state.pop("status_message")
        getattr(st, status_type, st.info)(message)

    agents = load_agents()
    agents.sort(key=lambda item: item.name.lower())
    agent_lookup = {agent.id: agent for agent in agents}

    selected_id = st.session_state.get("selected_agent_id")
    sidebar_options = list(agent_lookup.keys())

    selected_id = st.sidebar.selectbox(
        "Agent catalog",
        options=sidebar_options,
        format_func=lambda value: agent_lookup[value].display_name() if value in agent_lookup else value,
        index=sidebar_options.index(selected_id) if selected_id in sidebar_options else 0 if sidebar_options else 0,
    ) if sidebar_options else None

    st.session_state["selected_agent_id"] = selected_id

    st.sidebar.markdown("---")
    if st.sidebar.button("➕ Create new agent", use_container_width=True):
        st.session_state["editor_mode"] = "create"
        st.session_state["editor_agent_id"] = None
        st.rerun()

    if selected_id and st.sidebar.button("✏️ Edit selected", use_container_width=True):
        st.session_state["editor_mode"] = "edit"
        st.session_state["editor_agent_id"] = selected_id
        st.rerun()

    if selected_id and st.sidebar.button("🗑️ Delete selected", use_container_width=True):
        next_agents = [agent for agent in agents if agent.id != selected_id]
        save_agents(next_agents)
        st.session_state.pop(f"chat-history-{selected_id}", None)
        st.session_state["selected_agent_id"] = next_agents[0].id if next_agents else None
        st.session_state["status_message"] = ("warning", "Agent deleted.")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Agents are stored locally in `agents.json`.")

    active_agent = render_agent_editor(agents, selected_id)
    if active_agent is None:
        active_agent = agent_lookup.get(selected_id)

    if not active_agent:
        st.info("Create an agent to get started.")
        return

    overview_tab, planner_tab, chat_tab = st.tabs([
        "Overview",
        "Action planner",
        "Mission control",
    ])

    with overview_tab:
        render_agent_overview(active_agent)

    with planner_tab:
        render_action_planner(active_agent)

    with chat_tab:
        render_chat(active_agent)


if __name__ == "__main__":
    main()
