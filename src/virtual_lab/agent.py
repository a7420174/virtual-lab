
"""The LLM agent class compatible with OpenAI Agents SDK (Agent/Runner)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional


class Agent:
    """An LLM agent (framework-neutral), with helper to convert to Agents SDK Agent."""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str,
    ) -> None:
        """
        Initializes the agent.

        :param title: The title of the agent.
        :param expertise: The expertise of the agent.
        :param goal: The goal of the agent.
        :param role: The role of the agent.
        :param model: The LLM model identifier (e.g., 'gpt-4o-mini').
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model

    # ----------------------------
    # Prompt & system message
    # ----------------------------
    @property
    def prompt(self) -> str:
        """Returns the prompt (system instructions) for the agent."""
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    @property
    def message(self) -> Dict[str, Any]:
        """
        Returns a generic 'system' message dict without binding to OpenAI-specific types.
        This keeps consumers free to use either Chat Completions or Agents SDK.
        """
        return {"role": "system", "content": self.prompt}

    # ----------------------------
    # Agents SDK helper
    # ----------------------------
    def to_agents(
        self,
        *,
        tools: Optional[List[object]] = None,
        mcp_servers: Optional[List[object]] = None,
        handoffs: Optional[List[object]] = None,
        name: Optional[str] = None,
    ):
        """
        Convert to an Agents SDK Agent (from `openai-agents` package).

        Usage:
            from agents import Agent as AgentsAgent
            aa = my_agent.to_agents(tools=[...], mcp_servers=[...])

        :param tools: Agents SDK function tools / hosted MCP tools, etc.
        :param mcp_servers: MCP server transports (stdio/http/sse) to attach.
        :param handoffs: Optional handoffs to other agents.
        :param name: Optional explicit name; defaults to self.title.
        :return: agents.Agent instance.
        """
        try:
            # Lazy import to avoid hard dependency when not using Agents SDK here.
            from agents import Agent as AgentsAgent  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI Agents SDK (`openai-agents`) is required for to_agents(). "
                "Install with: pip install -U openai-agents"
            ) from e

        return AgentsAgent(
            name=name or self.title,
            instructions=self.prompt,
            model=self.model,
            tools=tools or [],
            mcp_servers=mcp_servers or [],
            handoffs=handoffs or [],
        )

    # ----------------------------
    # Identity & representation
    # ----------------------------
    def __hash__(self) -> int:
        """Returns the hash of the agent."""
        return hash((self.title, self.expertise, self.goal, self.role, self.model))

    def __eq__(self, other: object) -> bool:
        """Checks if the agent is equal to another agent (based on all fields)."""
        if not isinstance(other, Agent):
            return False
        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
        )

    def __str__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title

    def __repr__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title
