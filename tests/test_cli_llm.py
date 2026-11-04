import json
import subprocess

import pytest

from poster_agent.llm import LLMError
from poster_agent.llm.cli_llm import NEUTRAL_CWD, CLIAgentLLM


class FakeRun:
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def envelope(result, ok=True):
    return json.dumps({"is_error": not ok, "subtype": "success" if ok else "error", "result": result})


def test_each_call_is_a_fresh_toolless_session_in_an_empty_folder(monkeypatch):
    monkeypatch.setattr("poster_agent.llm.cli_llm.cli_path", lambda: "planner")
    run = FakeRun(envelope('```json\n{"format": "story"}\n```'))
    out = CLIAgentLLM(model="sonnet", run=run).generate_json("SYSTEM RULES", "the brief")
    assert out == {"format": "story"}
    cmd, kw = run.calls[0]
    assert cmd[:2] == ["planner", "-p"] and cmd[cmd.index("--tools") + 1] == "" and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == "" and "--strict-mcp-config" in cmd and kw["cwd"] == NEUTRAL_CWD
    assert kw["input"] == "the brief" and "--resume" not in cmd and "--continue" not in cmd


def test_retries_bad_replies_then_reports_the_failure(monkeypatch):
    monkeypatch.setattr("poster_agent.llm.cli_llm.cli_path", lambda: "planner")
    ok = CLIAgentLLM(run=FakeRun("not json", envelope('{"a": 1}'))).generate_json("s", "u")
    assert ok == {"a": 1}
    with pytest.raises(LLMError, match="failed"):
        CLIAgentLLM(run=FakeRun(envelope("x", ok=False), subprocess.TimeoutExpired("c", 1))).generate_json("s", "u")


def test_images_are_refused_so_the_fallback_chain_moves_on(monkeypatch):
    monkeypatch.setattr("poster_agent.llm.cli_llm.cli_path", lambda: "planner")
    with pytest.raises(LLMError, match="does not take images"):
        CLIAgentLLM(run=FakeRun()).generate_json("s", "u", images=["a.png"])
