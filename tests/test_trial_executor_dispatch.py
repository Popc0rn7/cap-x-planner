from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from capx.envs import runner
from capx.envs.trial import CodeAgentTrialExecutor
from capx.envs.trial_base import TrialContext
from capx.envs.trial_factory import build_trial_executor, register_trial_executor
from capx.envs.trial_fine_grained import FineGrainedTrialExecutor
from capx.utils import launch_utils
from capx.utils.launch_utils import TrialSummary


def _summary(trial: int) -> TrialSummary:
    return TrialSummary(
        trial=trial,
        success=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        sandbox_rc=0,
        log="ok",
        task_completed=True,
    )


def test_missing_executor_defaults_to_existing_code_agent() -> None:
    assert isinstance(build_trial_executor(None), CodeAgentTrialExecutor)


def test_string_shorthand_selects_executor() -> None:
    assert isinstance(build_trial_executor("fine_grained"), FineGrainedTrialExecutor)


def test_fine_grained_configuration_is_owned_by_its_executor() -> None:
    executor = build_trial_executor(
        {
            "type": "fine_grained",
            "component_factory": "example.build_components",
            "max_turns": 7,
        }
    )

    assert isinstance(executor, FineGrainedTrialExecutor)
    assert executor.config == {
        "component_factory": "example.build_components",
        "max_turns": 7,
    }


def test_unknown_executor_lists_available_types() -> None:
    with pytest.raises(ValueError, match="Available: code_agent, fine_grained"):
        build_trial_executor("missing")


def test_registration_rejects_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_trial_executor("", lambda config: CodeAgentTrialExecutor(config))
    with pytest.raises(ValueError, match="already registered"):
        register_trial_executor(
            "code_agent",
            lambda config: CodeAgentTrialExecutor(config),
        )


def test_registration_allows_explicit_replacement() -> None:
    register_trial_executor(
        "test_replaceable",
        lambda config: CodeAgentTrialExecutor({"version": 1}),
    )
    register_trial_executor(
        "test_replaceable",
        lambda config: CodeAgentTrialExecutor({"version": 2}),
        replace=True,
    )
    executor = build_trial_executor("test_replaceable")
    assert isinstance(executor, CodeAgentTrialExecutor)
    assert executor.config == {"version": 2}


def test_runner_only_calls_common_trial_executor_interface(monkeypatch) -> None:
    received: dict[str, Any] = {}

    class FakeExecutor:
        def run(self, context: TrialContext) -> TrialSummary:
            received["context"] = context
            return _summary(context.trial)

        def build_timeout_summary(self, context, timeout_seconds, exc):
            raise AssertionError("timeout summary should not be used")

    def fake_build_executor(spec):
        received["executor_spec"] = spec
        return FakeExecutor()

    monkeypatch.setattr(runner, "build_trial_executor", fake_build_executor)
    env = object()
    args = object()
    summary = runner._run_single_trial_with_timeout(
        env=env,  # type: ignore[arg-type]
        trial=2,
        args=args,
        config={"trial_executor": {"type": "test"}},
        multi_turn_prompt="prompt",
        timeout_s=5,
    )

    assert summary.trial == 2
    assert received["executor_spec"] == {"type": "test"}
    context = received["context"]
    assert context.env is env
    assert context.args is args
    assert context.multi_turn_prompt == "prompt"


def test_runner_uses_executor_timeout_summary(monkeypatch) -> None:
    received: dict[str, Any] = {}

    class TimeoutExecutor:
        def run(self, context: TrialContext) -> TrialSummary:
            raise TimeoutError("primitive service stalled")

        def build_timeout_summary(self, context, timeout_seconds, exc):
            received["timeout"] = (context.trial, timeout_seconds, str(exc))
            return _summary(context.trial)

    monkeypatch.setattr(runner, "build_trial_executor", lambda spec: TimeoutExecutor())
    summary = runner._run_single_trial_with_timeout(
        env=object(),  # type: ignore[arg-type]
        trial=4,
        args=object(),
        config={},
        multi_turn_prompt=None,
        timeout_s=5,
    )

    assert summary.trial == 4
    assert received["timeout"] == (4, 5, "primitive service stalled")


def test_fine_grained_components_are_built_for_each_trial() -> None:
    calls: list[Any] = []

    def factory(*, env):
        calls.append(env)
        return object()

    executor = FineGrainedTrialExecutor({"component_factory": factory})
    with pytest.raises(TypeError, match="FineGrainedTrialComponents"):
        executor._build_components("first")
    with pytest.raises(TypeError, match="FineGrainedTrialComponents"):
        executor._build_components("second")
    assert calls == ["first", "second"]


def _launch_args() -> SimpleNamespace:
    return SimpleNamespace(
        config_path="unused.yaml",
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_parallel_ensemble=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, {"type": "code_agent"}),
        ({"type": "fine_grained", "max_turns": 9}, {"type": "fine_grained", "max_turns": 9}),
    ],
)
def test_load_config_reads_top_level_trial_executor(monkeypatch, configured, expected) -> None:
    loaded: dict[str, Any] = {"env": {"_target_": "example.Env"}}
    if configured is not None:
        loaded["trial_executor"] = configured
    monkeypatch.setattr(launch_utils.DictLoader, "load", lambda paths: loaded)

    _, config, _ = launch_utils._load_config(_launch_args())

    assert config["trial_executor"] == expected
    assert "workflow" not in config
