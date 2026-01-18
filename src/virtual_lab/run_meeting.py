
# --- NEW: asyncio import ---
import asyncio
import os, time, httpx
from pathlib import Path
from typing import Literal, Dict, List, Tuple

from agents import Runner, function_tool, RunConfig, ModelSettings
from agents.mcp.server import MCPServerStdio, MCPServerStreamableHttp, MCPServerSse
from agents.tool import HostedMCPTool
from tqdm import trange, tqdm

from virtual_lab.agent import Agent  # <- uses new to_agents()
from virtual_lab.constants import CONSISTENT_TEMPERATURE
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
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
    return [pubmed_search] if pubmed_search_enabled else []

# run_meeting.py 상단 어딘가에
async def _connect_mcp_servers(servers: list[object]) -> None:
    for s in servers:
        # 모든 MCPServer*는 async connect()를 제공합니다.
        await s.connect()

async def _cleanup_mcp_servers(servers: list[object]) -> None:
    for s in servers:
        try:
            # 서버에 따라 cleanup()/close()가 구현되어 있습니다.
            await s.cleanup()
        except Exception:
            pass


def _build_biomcp_integration(
    mode: Literal["stdio", "http", "hosted"] = "stdio",
    url: str | None = None,
    env: dict | None = None,
):
    mcp_servers = []
    hosted_tools = []
    if mode == "stdio":
        mcp_servers.append(
            MCPServerStdio(
                params={
                    "command": "uv",
                    "args": ["run", "--with", "biomcp-python", "biomcp", "run"],
                    "env": env or {},
                    "timeout": 30,
                },
                cache_tools_list=True,
                max_retry_attempts=3,
            )
        )
    elif mode == "http":
        if not url:
            raise ValueError("HTTP mode requires biomcp_url")
        mcp_servers.append(
            MCPServerStreamableHttp(
                params={"url": url, "timeout": 30},
                cache_tools_list=True,
                max_retry_attempts=5,
                client_session_timeout_seconds=30,
            )
        )
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

# ---------------------------------------------------------------------
# ASYNC 버전: 노트북/Colab에서 바로 `await run_meeting_async(...)`
# ---------------------------------------------------------------------
async def run_meeting_async(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: Tuple[Agent, ...] | List[Agent] | None = None,
    team_member: Agent | None = None,
    critic: Agent | None = None,
    agenda_questions: Tuple[str, ...] = (),
    agenda_rules: Tuple[str, ...] = (),
    summaries: Tuple[str, ...] = (),
    contexts: Tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search_enabled: bool = False,
    return_summary: bool = False,
    use_biomcp: bool = False,
    biomcp_mode: Literal["stdio", "http", "hosted"] = "stdio",
    biomcp_url: str | None = None,
    biomcp_env: dict | None = None,
) -> str | None:

    # (기존 유효성 검증/팀 구성 동일)
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
        if critic is None:
            raise ValueError("Individual meeting requires a critic")
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    start_time = time.time()

    if meeting_type == "team":
        assert team_lead is not None and team_members is not None
        team: List[Agent] = [team_lead] + list(team_members)
        primary_model = team_lead.model
    else:
        assert team_member is not None
        team = [team_member, critic]
        primary_model = team_member.model

    tools_for_agents = _build_agents_tools(pubmed_search_enabled=pubmed_search_enabled)
    
    mcp_servers: List[object] = []
    hosted_mcp_tools: List[object] = []
    if use_biomcp:
        mcp_servers, hosted_mcp_tools = _build_biomcp_integration(
            mode=biomcp_mode, url=biomcp_url, env=biomcp_env
        )
        tools_for_agents = tools_for_agents + hosted_mcp_tools

    # 여기가 핵심: MCP 서버 연결
    # HostedMCPTool만 쓰는 경우(connect 불필요)에는 mcp_servers가 빈 리스트일 수 있음
    await _connect_mcp_servers(mcp_servers)

    try:
        # --- 기존 라운드/러너 실행 로직 그대로 ---
        agents_cache: Dict[Agent, object] = {}
        def get_agents_agent(v_agent: Agent):
            if v_agent not in agents_cache:
                agents_cache[v_agent] = v_agent.to_agents(
                    tools=tools_for_agents,
                    mcp_servers=mcp_servers,     # 이미 connect() 완료된 서버
                    name=getattr(v_agent, "title", None) or "Agent",
                )
            return agents_cache[v_agent]

        tool_token_count = 0
        discussion: List[dict[str, str]] = []

        if meeting_type == "team":
            assert team_lead is not None and team_members is not None
            initial_content = team_meeting_start_prompt(
                team_lead=team_lead,
                team_members=tuple(team_members),
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
                num_rounds=num_rounds,
            )
            discussion.append({"agent": "User", "message": initial_content})

        # --- Rounds (Async: Runner.run) ---
        for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
            round_num = round_index + 1
            for v_agent in tqdm(team, desc="Team"):
                # 프롬프트 선택 로직 (동일)
                if meeting_type == "team":
                    assert team_lead is not None
                    if v_agent == team_lead:
                        if round_index == 0:
                            prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                        elif round_index == num_rounds:
                            prompt = team_meeting_team_lead_final_prompt(
                                team_lead=team_lead, agenda=agenda,
                                agenda_questions=agenda_questions, agenda_rules=agenda_rules,
                            )
                        else:
                            prompt = team_meeting_team_lead_intermediate_prompt(
                                team_lead=team_lead, round_num=round_num - 1, num_rounds=num_rounds,
                            )
                    else:
                        prompt = team_meeting_team_member_prompt(
                            team_member=v_agent, round_num=round_num, num_rounds=num_rounds
                        )
                else:
                    assert team_member is not None
                    if v_agent == critic:
                        prompt = individual_meeting_critic_prompt(critic=critic, agent=team_member)
                    else:
                        if round_index == 0:
                            prompt = individual_meeting_start_prompt(
                                team_member=team_member, agenda=agenda,
                                agenda_questions=agenda_questions, agenda_rules=agenda_rules,
                                summaries=summaries, contexts=contexts,
                            )
                        else:
                            prompt = individual_meeting_agent_prompt(critic=critic, agent=team_member)

                discussion.append({"agent": "User", "message": prompt})

                a_agent = get_agents_agent(v_agent)
                run_config = RunConfig(model_settings=ModelSettings(temperature=temperature))

                # 비동기 실행
                result = await Runner.run(
                    starting_agent=a_agent,
                    input=prompt,
                    run_config=run_config,
                    max_turns=10,
                )

                response_text = str(result.final_output or "")
                for item in getattr(result, "new_items", []) or []:
                    t = getattr(item, "type", "")
                    if t == "tool_call_output_item":
                        output = getattr(item, "output", "")
                        if output:
                            discussion.append({"agent": "Tool", "message": str(output)})
                            tool_token_count += count_tokens(str(output))
                    if not response_text and t == "message_output_item":
                        content = getattr(item, "content", None)
                        if isinstance(content, list):
                            texts = [getattr(p, "text", "") for p in content if hasattr(p, "text")]
                            if any(texts):
                                response_text = "".join(texts).strip()
                                break
                        elif isinstance(content, str) and content.strip():
                            response_text = content.strip()
                            break

                discussion.append({"agent": getattr(v_agent, "title", "Assistant"), "message": response_text})
                if round_index == num_rounds:
                    break

        token_counts = count_discussion_tokens(discussion=discussion)
        token_counts["tool"] = tool_token_count
        print_cost_and_time(
            token_counts=token_counts,
            model=primary_model.model,
            elapsed_time=time.time() - start_time,
        )

        save_meeting(save_dir=save_dir, save_name=save_name, discussion=discussion)
        if return_summary:
            return get_summary(discussion)
        return None

    finally:
        # 종료 시 MCP 서버 정리
        await _cleanup_mcp_servers(mcp_servers)


# ---------------------------------------------------------------------
# SYNC 래퍼: 스크립트/터미널에서 바로 사용 (노트북 루프가 없을 때만)
# ---------------------------------------------------------------------
def run_meeting(*args, **kwargs):
    """
    Synchronous wrapper. If an event loop is already running (e.g. Jupyter),
    guides the user to call `await run_meeting_async(...)` instead.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError(
                "An event loop is already running. In notebooks, please call:\n"
                "    await run_meeting_async(...)\n"
            )
    except RuntimeError:
        # No running loop: safe to run
        pass
    return asyncio.run(run_meeting_async(*args, **kwargs))
