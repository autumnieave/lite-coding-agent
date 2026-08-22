"""`cli/main.py` 的单元测试。LLM 用假 Provider 替换，不联网。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.cli import main as cli
from agent.cli.main import _persist_session
from agent.core.compaction import TIER1, CompactionEvent, Compactor
from agent.core.constraints import ConstraintStore
from agent.core.llm import (
    BaseProvider,
    LLMError,
    LLMResponse,
    ToolCall,
    assistant_message,
    system_message,
    user_message,
)
from agent.core.loop import LoopResult
from agent.memory.session import SessionState, SessionStore


class _ScriptedProvider(BaseProvider):
    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": [dict(item) for item in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("Provider 被调用次数超出脚本预期")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _StubFactory:
    """替换 OpenAICompatProvider，让 from_env 直接返回脚本化 Provider。"""

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def from_env(self, environ: Mapping[str, str] | None = None) -> BaseProvider:
        return self._provider


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("第一行\n第二行\n", encoding="utf-8")
    return tmp_path


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: BaseProvider) -> None:
    monkeypatch.setattr(cli, "OpenAICompatProvider", _StubFactory(provider))


# ---------- 参数解析 ----------


def test_parser_reads_chat_task() -> None:
    args = cli.build_parser().parse_args(["chat", "列出当前目录"])
    assert args.command == "chat"
    assert args.task == "列出当前目录"
    assert args.max_turns == 10
    assert args.root == "."
    assert args.verbose is False


def test_parser_reads_chat_options() -> None:
    args = cli.build_parser().parse_args(
        ["chat", "任务", "--max-turns", "3", "--root", "sub", "--verbose"]
    )
    assert args.max_turns == 3
    assert args.root == "sub"
    assert args.verbose is True


def test_parser_accepts_chat_without_task() -> None:
    args = cli.build_parser().parse_args(["chat"])
    assert args.task is None


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert cli.__version__ in capsys.readouterr().out


# ---------- 帮助与用参 ----------


def test_main_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "chat" in out


def test_chat_without_task_prints_chat_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["chat"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "lite-agent chat" in out
    assert "--max-turns" in out


def test_invalid_max_turns_returns_usage_error(
    capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    code = cli.main(["chat", "任务", "--max-turns", "0", "--root", str(workspace)])
    assert code == cli.EXIT_USAGE_ERROR
    assert "--max-turns" in capsys.readouterr().err


def test_missing_workspace_returns_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["chat", "任务", "--root", str(tmp_path / "nope")])
    assert code == cli.EXIT_USAGE_ERROR
    assert "工作区目录不存在" in capsys.readouterr().err


def test_missing_api_key_returns_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    # 屏蔽开发机上真实存在的 .env，保证用例与环境无关
    monkeypatch.setattr(cli, "find_env_file", lambda *args, **kwargs: None)

    code = cli.main(["chat", "任务", "--root", str(workspace)])
    assert code == cli.EXIT_USAGE_ERROR
    assert "LLM_API_KEY" in capsys.readouterr().err


# ---------- 正常执行 ----------


def test_successful_run_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="这是答案")]))

    code = cli.main(["chat", "问题", "--root", str(workspace)])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert captured.out == "这是答案\n"
    assert captured.err == ""


def test_run_with_real_tools_reads_workspace_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="read_file", arguments='{"path": "notes.txt"}'),
                ),
            ),
            LLMResponse(content="两行"),
        ]
    )
    _use_provider(monkeypatch, provider)

    code = cli.main(["chat", "notes.txt 有几行", "--root", str(workspace)])

    assert code == cli.EXIT_OK
    assert capsys.readouterr().out == "两行\n"
    tool_message = [m for m in provider.calls[1]["messages"] if m["role"] == "tool"][-1]
    assert "第一行" in tool_message["content"]


def test_verbose_writes_events_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="list_dir", arguments='{"path": "."}'),),
            ),
            LLMResponse(content="目录已列出"),
        ]
    )
    _use_provider(monkeypatch, provider)

    code = cli.main(["chat", "列出目录", "--root", str(workspace), "--verbose"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert captured.out == "目录已列出\n"
    assert "[verbose]" in captured.err
    assert "list_dir" in captured.err
    assert "成功" in captured.err


def test_non_verbose_reports_tools_tersely(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    """工具进度默认就实时上报，但走 stderr，stdout 只留答案。"""
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="list_dir", arguments='{"path": "."}'),),
            ),
            LLMResponse(content="完成"),
        ]
    )
    _use_provider(monkeypatch, provider)

    code = cli.main(["chat", "列出目录", "--root", str(workspace)])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert captured.out == "完成\n"
    assert "· " in captured.err
    assert "调用 list_dir" in captured.err
    assert "结果 list_dir" in captured.err
    assert "[verbose]" not in captured.err


def test_streamed_content_is_not_printed_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    """模型文本按分片落到 stdout，结尾不会因为补印而重复一遍。"""

    class _StreamingProvider(BaseProvider):
        async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
        ) -> LLMResponse:
            raise AssertionError("启用流式后不应再走非流式 chat")

        async def chat_stream(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
            on_text: object = None,
        ) -> LLMResponse:
            for piece in ("你", "好", "世界"):
                if callable(on_text):
                    on_text(piece)
            return LLMResponse(content="你好世界")

    _use_provider(monkeypatch, _StreamingProvider())

    code = cli.main(["chat", "打个招呼", "--root", str(workspace)])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert captured.out == "你好世界\n"


def test_streamed_content_without_trailing_newline_gets_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    class _StreamingProvider(BaseProvider):
        async def chat(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
        ) -> LLMResponse:
            raise AssertionError("应走流式")

        async def chat_stream(
            self,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] | None = None,
            on_text: object = None,
        ) -> LLMResponse:
            if callable(on_text):
                on_text("结尾带换行\n")
            return LLMResponse(content="结尾带换行\n")

    _use_provider(monkeypatch, _StreamingProvider())

    cli.main(["chat", "任务", "--root", str(workspace)])

    assert capsys.readouterr().out == "结尾带换行\n"


# ---------- 失败路径 ----------


def test_max_turns_reached_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="list_dir", arguments='{"path": "."}'),),
            )
        ]
    )
    _use_provider(monkeypatch, provider)

    code = cli.main(["chat", "任务", "--root", str(workspace), "--max-turns", "1"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_TASK_FAILED
    assert "最大轮数" in captured.err


def test_env_file_is_searched_from_cwd(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    searched: list[Path] = []

    def _record(start: Path, **kwargs: object) -> None:
        searched.append(start)
        return None

    monkeypatch.setattr(cli, "find_env_file", _record)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="ok")]))

    cli.main(["chat", "任务", "--root", str(workspace)])

    assert searched == [Path.cwd()]


def test_env_file_values_reach_the_environment(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(cli, "find_env_file", lambda *args, **kwargs: env_file)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="ok")]))

    code = cli.main(["chat", "任务", "--root", str(workspace)])

    assert code == cli.EXIT_OK
    assert os.environ.get("LLM_API_KEY") == "from-dotenv"


def test_verbose_reports_injected_env_keys(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(cli, "find_env_file", lambda *args, **kwargs: env_file)
    monkeypatch.setattr(cli, "load_env_file", lambda path, **kwargs: {"LLM_API_KEY": "x"})
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="ok")]))

    cli.main(["chat", "任务", "--root", str(workspace), "--verbose"])

    captured = capsys.readouterr()
    assert "已从" in captured.err
    assert "LLM_API_KEY" in captured.err
    # 不应打印密钥值
    assert "from-dotenv" not in captured.err and "=x" not in captured.err


def test_unreadable_env_file_warns_but_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workspace: Path,
    tmp_path: Path,
) -> None:
    def _boom(path: Path, **kwargs: object) -> dict:
        raise OSError("权限不足")

    monkeypatch.setattr(cli, "find_env_file", lambda *args, **kwargs: tmp_path / ".env")
    monkeypatch.setattr(cli, "load_env_file", _boom)
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="ok")]))

    code = cli.main(["chat", "任务", "--root", str(workspace)])

    assert code == cli.EXIT_OK
    assert "无法读取" in capsys.readouterr().err


def test_llm_error_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMError("网络不可达")]))

    code = cli.main(["chat", "任务", "--root", str(workspace)])

    captured = capsys.readouterr()
    assert code == cli.EXIT_TASK_FAILED
    assert "网络不可达" in captured.err
    assert captured.out == ""


# ---------- 约束存储接线 ----------


def _state_file(workspace: Path) -> Path:
    return workspace / cli.CONSTRAINTS_DIR / "constraints.json"


def test_verbose_reports_loaded_constraints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    path = _state_file(workspace)
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {"constraints": [{"id": "C5", "content": "禁止改迁移", "source": "agents_md"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))

    code = cli.main(["chat", "问题", "--root", str(workspace), "--verbose"])

    assert code == cli.EXIT_OK
    assert "已加载 1 条约束" in capsys.readouterr().err


def test_declared_constraints_are_persisted(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """用户在任务里声明约束后，即使这轮没触发压缩，也要落盘备查。"""
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))

    code = cli.main(["chat", "[CONSTRAINT] 代号 R3MJUD：输出必须是 JSON", "--root", str(workspace)])

    assert code == cli.EXIT_OK
    saved = json.loads(_state_file(workspace).read_text(encoding="utf-8"))
    assert [item["id"] for item in saved["constraints"]] == ["R3MJUD"]


def test_no_constraints_file_when_nothing_is_declared(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """没有约束来源时不该造出 constraints.json（会话日志另算，它每轮都有）。"""
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))

    assert cli.main(["chat", "普通问题", "--root", str(workspace)]) == cli.EXIT_OK
    assert not _state_file(workspace).exists()


# ---------- AGENTS.md 接线 ----------


def _write_agents_md(root: Path, *constraints: str) -> None:
    body = "# 项目规则\n\n## 关键约束\n\n" + "\n".join(constraints) + "\n"
    (root / "AGENTS.md").write_text(body, encoding="utf-8")


def _saved_constraints(workspace: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(_state_file(workspace).read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["constraints"]}


def test_agents_md_constraints_are_registered_on_start(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """任务启动时按目录层级读 AGENTS.md，C1/C2 进约束存储并标 agents_md。"""
    _write_agents_md(workspace, "- **C1**：必须兼容 Python 3.11。", "- **C2**：只允许标准库。")
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))

    assert cli.main(["chat", "普通问题", "--root", str(workspace)]) == cli.EXIT_OK

    saved = _saved_constraints(workspace)
    assert sorted(saved) == ["C1", "C2"]
    assert saved["C1"]["content"] == "必须兼容 Python 3.11。"
    assert all(item["source"] == "agents_md" for item in saved.values())


def test_agents_md_edit_takes_effect_on_the_next_run(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """改了 AGENTS.md 的 C1，下一次运行必须用新正文——文件是唯一真源。"""
    _write_agents_md(workspace, "- **C1**：改写前的正文。")
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))
    assert cli.main(["chat", "第一次", "--root", str(workspace)]) == cli.EXIT_OK
    assert _saved_constraints(workspace)["C1"]["content"] == "改写前的正文。"

    _write_agents_md(workspace, "- **C1**：改写后的正文。")
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))
    assert cli.main(["chat", "第二次", "--root", str(workspace)]) == cli.EXIT_OK

    assert _saved_constraints(workspace)["C1"]["content"] == "改写后的正文。"


def test_verbose_reports_agents_md_registration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _write_agents_md(workspace, "- **C1**：必须兼容 Python 3.11。")
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))

    code = cli.main(["chat", "问题", "--root", str(workspace), "--verbose"])

    assert code == cli.EXIT_OK
    assert "AGENTS.md 约束：新增 1 条" in capsys.readouterr().err


def test_workspace_without_agents_md_registers_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))

    code = cli.main(["chat", "问题", "--root", str(workspace), "--verbose"])

    assert code == cli.EXIT_OK
    assert "AGENTS.md 约束" not in capsys.readouterr().err


# ---------- session checkpoint 接线 ----------


def _session_path(workspace: Path) -> Path:
    return workspace / cli.CONSTRAINTS_DIR / "session.jsonl"


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_session_log_is_written_after_a_run(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))

    assert cli.main(["chat", "问题", "--root", str(workspace)]) == cli.EXIT_OK

    records = _records(_session_path(workspace))
    kinds = [item["type"] for item in records]
    assert kinds[0] == "reset"  # 全新会话：先划一条起跑线
    assert kinds[-1] == "state"
    assert "message" in kinds
    assert records[-1]["turns"] == 1
    assert records[-1]["tokens"] > 0


def test_second_run_resumes_the_previous_session(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """第二次运行必须带着上一轮的消息历史去问模型。"""
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="第一轮答案")]))
    assert cli.main(["chat", "第一轮问题", "--root", str(workspace)]) == cli.EXIT_OK

    second = _ScriptedProvider([LLMResponse(content="第二轮答案")])
    _use_provider(monkeypatch, second)
    assert cli.main(["chat", "第二轮问题", "--root", str(workspace)]) == cli.EXIT_OK

    sent = [str(item.get("content")) for item in second.calls[0]["messages"]]
    assert "第一轮问题" in sent
    assert "第一轮答案" in sent
    assert "第二轮问题" in sent
    assert sent.count("第二轮问题") == 1


def test_verbose_reports_the_restored_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], workspace: Path
) -> None:
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))
    cli.main(["chat", "第一轮", "--root", str(workspace)])
    capsys.readouterr()

    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))
    cli.main(["chat", "第二轮", "--root", str(workspace), "--verbose"])

    assert "已恢复会话" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_persist_session_appends_only_the_new_tail(tmp_path: Path) -> None:
    """没触发压缩时只追加新增的尾巴，日志保持纯追加。"""
    session = SessionStore(tmp_path / "session.jsonl")
    carried = [system_message("系统"), user_message("第一轮"), assistant_message("第一轮答案")]
    session.append_messages(carried)
    before = _records(session.path)  # type: ignore[arg-type]

    fresh = [*carried, user_message("第二轮"), assistant_message("第二轮答案")]
    _persist_session(
        session,
        SessionState(messages=tuple(carried)),
        _result(fresh, turns=2),
        Compactor(summarize=_noop_summary),
        ConstraintStore(),
    )

    records = _records(session.path)  # type: ignore[arg-type]
    assert records[: len(before)] == before  # 追加，不改写
    assert [item["type"] for item in records[len(before) :]] == ["message", "message", "state"]
    assert [item["content"] for item in session.load().messages][-2:] == [
        "第二轮",
        "第二轮答案",
    ]


@pytest.mark.asyncio
async def test_persist_session_resets_when_compaction_fired(tmp_path: Path) -> None:
    """压缩改写上下文后必须追一条 reset，否则回放会「原文 + 摘要」两份都在。"""
    session = SessionStore(tmp_path / "session.jsonl")
    carried = [system_message("系统"), user_message("很久以前的问题")]
    session.append_messages(carried)

    compactor = Compactor(summarize=_noop_summary)
    compactor.stats.add(
        CompactionEvent(tier=TIER1, tokens_before=100, tokens_after=10, duration_ms=0.1)
    )
    compacted = [system_message("系统"), user_message("压缩后的问题")]
    _persist_session(
        session,
        SessionState(messages=tuple(carried)),
        _result(compacted),
        compactor,
        ConstraintStore(),
    )

    kinds = [item["type"] for item in _records(session.path)]  # type: ignore[arg-type]
    assert kinds == ["message", "message", "reset", "message", "message", "state"]
    replayed = [item["content"] for item in session.load().messages]
    assert replayed == ["系统", "压缩后的问题"]


def test_session_write_failure_does_not_break_the_task(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """写日志失败（磁盘满、只读目录）只是「这次没存上」，不能影响退出码。"""
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="答案")]))

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("磁盘满了")

    monkeypatch.setattr(cli, "SessionStore", lambda *a, **k: _ExplodingSession())

    assert cli.main(["chat", "问题", "--root", str(workspace)]) == cli.EXIT_OK


class _ExplodingSession(SessionStore):
    def __init__(self) -> None:
        super().__init__(None)

    def load(self) -> SessionState:
        return SessionState()

    def append_reset(self) -> None:
        raise OSError("磁盘满了")

    def append_messages(self, messages: object) -> None:
        raise OSError("磁盘满了")

    def append_state(self, **kwargs: object) -> None:
        raise OSError("磁盘满了")


async def _noop_summary(messages: object) -> str:
    return "摘要"


def _result(messages: list[dict[str, Any]], *, turns: int = 1) -> LoopResult:
    return LoopResult(
        content="答案",
        turns=turns,
        stopped_reason="completed",
        messages=tuple(messages),
    )


def test_short_task_sees_agents_md_constraints(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """短任务不触发压缩，但 AGENTS.md 的约束仍然在 system prompt 里（ADR-016）。"""
    _write_agents_md(workspace, "- **C1**：必须兼容 Python 3.11。")
    provider = _ScriptedProvider([LLMResponse(content="好")])
    _use_provider(monkeypatch, provider)

    assert cli.main(["chat", "短任务", "--root", str(workspace)]) == cli.EXIT_OK

    systems = [item for item in provider.calls[0]["messages"] if item["role"] == "system"]
    assert len(systems) == 1
    assert "- [C1] 必须兼容 Python 3.11。" in systems[0]["content"]


def test_declared_constraint_persists_into_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """第一轮声明的约束由压缩环节吸收，第二轮随 system prompt 一起带上。

    第一轮拿不到是有意的：吸收发生在 `Compactor.compact()` 里、而 system prompt 在它之前
    就拼好了；不过那条约束本来就在用户消息里，模型当轮看得到。
    """
    _use_provider(monkeypatch, _ScriptedProvider([LLMResponse(content="好")]))
    cli.main(["chat", "[CONSTRAINT] 代号 R3MJUD：输出必须是 JSON", "--root", str(workspace)])

    second = _ScriptedProvider([LLMResponse(content="好")])
    _use_provider(monkeypatch, second)
    assert cli.main(["chat", "继续", "--root", str(workspace)]) == cli.EXIT_OK

    systems = [item for item in second.calls[0]["messages"] if item["role"] == "system"]
    assert "- [R3MJUD] 输出必须是 JSON" in systems[0]["content"]
