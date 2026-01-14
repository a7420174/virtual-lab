
"""
Runs a meeting with LLM agents using the OpenAI Agents SDK (Agent, Runner),
and integrates optional BioMCP server(s).

- Requires: openai-agents (Agents SDK)
- Optional: biomcp-python (for stdio mode), or a hosted/HTTP BioMCP endpoint
"""

import os, time, httpx
from pathlib import Path
from typing import Literal, Dict, List, Tuple

# Agents SDK
from agents import Runner, function_tool, RunConfig, ModelSettings
from agents.mcp.server import MCPServerStdio, MCPServerStreamableHttp, MCPServerSse
from agents.tool import HostedMCPTool

from tqdm import trange, tqdm

# Project-local modules
from virtual_lab.agent import Agent  # <- uses new to_agents()
from virtual_lab.constants import CONSISTENT_TEMPERATURE, PUBMED_TOOL_DESCRIPTION
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import (
    count_discussion_tokens,
    count_tokens,
    get_summary,
    print_cost_and_time,
    save_meeting,
)

# ---------------------------------------------------------------------
# Example function tool (PubMed): Agents SDK recommends @function_tool
# If you enable BioMCP, it already provides article/trial/variant tools,
# so this can be turned off.
# ---------------------------------------------------------------------
@function_tool
def pubmed_search(query: str, top_k: int = 5) -> str:
    """
    Simple PubMed search via NCBI E-utilities.
    For production: add robust error handling, rate-limit, parameter validation.
    """
    try:
        esearch = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": str(top_k)},
            timeout=30.0,
        )
        esearch.raise_for_status()
        ids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results for query: {query}"

        esummary = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=30.0,
        )
        esummary.raise_for_status()
        result = esummary.json().get("result", {})
        lines: List[str] = []
        for pid in ids:
            item = result.get(pid, {})
            title = item.get("title") or "(no title)"
            journal = item.get("fulljournalname") or item.get("source") or ""
            pubdate = item.get("pubdate") or ""
            lines.append(f"- {title} ({journal}, {pubdate}) https://pubmed.ncbi.nlm.nih.gov/{pid}/")
        return "\n".join(lines)
    except Exception as e:
        return f"PubMed search failed: {e}"


def _build_agents_tools(pubmed_search_enabled: bool):
    """
    Compose the local function tools list for Agents SDK.
    If BioMCP is used, you may set pubmed_search_enabled=False.
    """
    return [pubmed_search] if pubmed_search_enabled else []


def _build_biomcp_integration(
    mode: Literal["stdio", "http", "hosted"] = "stdio",
    url: str | None = None,
    env: dict | None = None,
):
    """
    Configure BioMCP server(s) or hosted MCP tool.

    mode:
      - "stdio": launch BioMCP locally via uv + biomcp-python (recommended for dev).
      - "http": connect to an externally reachable Streamable HTTP server.
      - "hosted": let OpenAI Responses call a hosted MCP server via HostedMCPTool.

    Returns:
      mcp_servers: list of MCP server transports (stdio/http/sse)
      hosted_tools: list of HostedMCPTool (only for 'hosted' mode)
    """
    mcp_servers = []
    hosted_tools = []

    if mode == "stdio":
        # uv run --with biomcp-python biomcp run
        # Docs & quickstart: https://biomcp.org/
        mcp_servers.append(
            MCPServerStdio(
                command="uv",
                args=["run", "--with", "biomcp-python", "biomcp", "run"],
                env=env or {},
            )
        )
    elif mode == "http":
        if not url:
            raise ValueError("HTTP mode requires biomcp_url")
        mcp_servers.append(MCPServerStreamableHttp(server_url=url))
    elif mode == "hosted":
        if not url:
            raise ValueError("Hosted mode requires biomcp server_url or label")
        hosted_tools.append(
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "biomcp",
                    "server_url": url,
                    "require_approval": "never",
                }
            )
        )
    else:
        raise ValueError(f"Unsupported biomcp mode: {mode}")

    return mcp_servers, hosted_tools


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: Tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: Tuple[str, ...] = (),
    agenda_rules: Tuple[str, ...] = (),
    summaries: Tuple[str, ...] = (),
    contexts: Tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search_enabled: bool = False,  # default off when using BioMCP
    return_summary: bool = False,
    # BioMCP options
    use_biomcp: bool = False,
    biomcp_mode: Literal["stdio", "http", "hosted"] = "stdio",
    biomcp_url: str | None = None,
    biomcp_env: dict | None = None,
) -> str | None:
    """
    Runs a meeting with LLM agents using OpenAI Agents SDK (Agent, Runner),
    with optional BioMCP integration.

    Returns the summary of the meeting if return_summary is True, else None.
    """

    # --- Validate meeting type (same as previous) ---
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError("Individual meeting does not require team lead or team members")
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # --- Start timing ---
    start_time = time.time()

    # --- Team setup ---
    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        team: List[Agent] = [team_lead] + list(team_members)
        primary_model = team_lead.model
    else:
        assert team_member is not None
        team = [team_member, SCIENTIFIC_CRITIC]
        primary_model = team_member.model

    # --- Tools & BioMCP ---
    tools_for_agents = _build_agents_tools(pubmed_search_enabled=pubmed_search_enabled)
    mcp_servers: List[object] = []
    hosted_mcp_tools: List[object] = []
    if use_biomcp:
        mcp_servers, hosted_mcp_tools = _build_biomcp_integration(
            mode=biomcp_mode, url=biomcp_url, env=biomcp_env
        )
        tools_for_agents = tools_for_agents + hosted_mcp_tools

    # --- Cache: virtual_lab.Agent -> agents.Agent via to_agents() ---
    agents_cache: Dict[Agent, object] = {}

    def get_agents_agent(v_agent: Agent):
        if v_agent not in agents_cache:
            # ✅ NEW: use Agent.to_agents(...) from the updated agent.py
            agents_cache[v_agent] = v_agent.to_agents(
                tools=tools_for_agents,
                mcp_servers=mcp_servers,
                name=getattr(v_agent, "title", None) or "Agent",
            )
        return agents_cache[v_agent]

    # --- Discussion logs ---
    tool_token_count = 0
    discussion: List[dict[str, str]] = []  # [{"agent": "...", "message": "..."}]

    # --- Initial team prompt (once) ---
    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        initial_content = team_meeting_start_prompt(
            team_lead=team_lead,
            team_members=team_members,
            agenda=agenda,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
        )
        discussion.append({"agent": "User", "message": initial_content})

    # --- Rounds ---
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        for v_agent in tqdm(team, desc="Team"):
            # 1) Round-specific prompt selection
            if meeting_type == "team":
                assert team_lead is not None
                if v_agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=v_agent, round_num=round_num, num_rounds=num_rounds
                    )
            else:
                assert team_member is not None
                if v_agent == SCIENTIFIC_CRITIC:
                    prompt = individual_meeting_critic_prompt(critic=SCIENTIFIC_CRITIC, agent=team_member)
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(critic=SCIENTIFIC_CRITIC, agent=team_member)

            # 2) Log user turn
            discussion.append({"agent": "User", "message": prompt})

            # 3) Run this agent via Runner
            a_agent = get_agents_agent(v_agent)
            run_config = RunConfig(
                model_settings=ModelSettings(temperature=temperature),
            )
            result = Runner.run_sync(
                starting_agent=a_agent,
                input=prompt,
                run_config=run_config,
                max_turns=10,  # adjust as needed
            )

            # 4) Extract final response text
            response_text = str(result.final_output or "")

            # 5) Surface tool outputs (optional) & token accounting
            for item in getattr(result, "new_items", []) or []:
                t = getattr(item, "type", "")
                if t == "tool_call_output_item":
                    output = getattr(item, "output", "")
                    if output:
                        discussion.append({"agent": "Tool", "message": str(output)})
                        tool_token_count += count_tokens(str(output))

            # 6) Append assistant response
            discussion.append({"agent": getattr(v_agent, "title", "Assistant"), "message": response_text})

            # 7) Final round: only team lead/member responds
            if round_index == num_rounds:
                break

    # --- Token/cost/time ---
    token_counts = count_discussion_tokens(discussion=discussion)
    token_counts["tool"] = tool_token_count
    print_cost_and_time(
        token_counts=token_counts,
        model=primary_model,
        elapsed_time=time.time() - start_time,
    )

    # --- Save ---
    save_meeting(save_dir=save_dir, save_name=save_name, discussion=discussion)

    # --- Optional summary ---
    if return_summary:
        return get_summary(discussion)
    return None
