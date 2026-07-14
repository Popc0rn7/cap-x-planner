"""Agent-authored, evidence-backed binary rewards for fine-grained stages."""

from __future__ import annotations

import ast
import copy
import json
import signal
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from capx.envs.trial_fine_grained import ToolCall, ToolResult, WorldState
    from capx.planning.stage_planner import TurnContext


class RewardStatus(StrEnum):
    """Interpretation of one binary Stage reward."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class StageRewardRule:
    """Structured rule that tells an agent what evidence establishes success."""

    predicate: str
    description: str
    target: Any | None = None
    required_evidence: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not isinstance(self.description, str):
            raise TypeError("StageRewardRule predicate and description must be strings")
        object.__setattr__(self, "required_evidence", tuple(self.required_evidence))
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        if not self.predicate.strip():
            raise ValueError("StageRewardRule.predicate must be non-empty")
        if not self.description.strip():
            raise ValueError("StageRewardRule.description must be non-empty")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.required_evidence
        ):
            raise ValueError("StageRewardRule.required_evidence entries must be non-empty")
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed_tools):
            raise ValueError("StageRewardRule.allowed_tools entries must be non-empty")

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any]) -> StageRewardRule:
        """Convert the former free-form postcondition dictionary into a reward rule."""

        predicate = str(value.get("predicate", "")).strip()
        if not predicate:
            raise ValueError("Legacy postcondition requires a non-empty 'predicate'")
        known = {
            "predicate",
            "description",
            "target",
            "required_evidence",
            "allowed_tools",
        }
        target = value.get("target")
        extra = {str(key): item for key, item in value.items() if key not in known}
        if target is None and extra:
            target = extra
        return cls(
            predicate=predicate,
            description=str(value.get("description") or predicate.replace("_", " ")),
            target=target,
            required_evidence=tuple(str(item) for item in value.get("required_evidence", ())),
            allowed_tools=tuple(str(item) for item in value.get("allowed_tools", ())),
        )


@dataclass(frozen=True)
class StageRewardContext:
    """Read-only inputs supplied to the Stage reward code agent."""

    turn: TurnContext
    call: ToolCall
    result: ToolResult
    state_before: WorldState
    state_after: WorldState
    rules: tuple[StageRewardRule, ...]

    def __post_init__(self) -> None:
        predicates = [rule.predicate for rule in self.rules]
        if len(predicates) != len(set(predicates)):
            raise ValueError("Stage reward predicates must be unique within one ToolCall")


@dataclass(frozen=True)
class StageReward:
    """Auditable binary reward for one physical execution turn."""

    status: RewardStatus
    code: str
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    reward_valid: bool | None = None
    generated_code: str | None = None
    tool_trace: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, RewardStatus):
            object.__setattr__(self, "status", RewardStatus(self.status))
        if not isinstance(self.evidence, Mapping):
            raise TypeError("StageReward.evidence must be a mapping")
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "tool_trace", tuple(self.tool_trace))
        expected_reward = 1.0 if self.status is RewardStatus.SUCCESS else 0.0
        expected_valid = self.status is not RewardStatus.UNKNOWN
        reward = expected_reward if self.reward is None else float(self.reward)
        reward_valid = expected_valid if self.reward_valid is None else self.reward_valid
        if reward not in {0.0, 1.0}:
            raise ValueError("StageReward.reward must be 0.0 or 1.0")
        if reward != expected_reward:
            raise ValueError(f"{self.status.value} StageReward requires reward={expected_reward}")
        if bool(reward_valid) != expected_valid:
            raise ValueError(
                f"{self.status.value} StageReward requires reward_valid={expected_valid}"
            )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "reward_valid", bool(reward_valid))


@dataclass(frozen=True)
class ReadOnlyVerificationTool:
    """Trusted read-only evidence provider available to generated reward code."""

    name: str
    description: str
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("Verification tool names must be Python identifiers")
        if not callable(self.handler):
            raise TypeError("Verification tool handler must be callable")


class VerificationToolRegistry:
    """Registry whose entries are explicitly declared safe for verification."""

    def __init__(self, tools: tuple[ReadOnlyVerificationTool, ...] = ()) -> None:
        self._tools: dict[str, ReadOnlyVerificationTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate verification tool: {tool.name}")
            self._tools[tool.name] = tool

    def descriptions(self, allowed: tuple[str, ...]) -> tuple[dict[str, str], ...]:
        return tuple(
            {"name": name, "description": self._require(name).description} for name in allowed
        )

    def handler(self, name: str) -> Callable[..., Any]:
        return self._require(name).handler

    def _require(self, name: str) -> ReadOnlyVerificationTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown verification tool: {name}") from exc


class StageRewardCodeAgent(Protocol):
    """Generate concrete reward code from a structured rule and turn context."""

    def generate(
        self,
        context: StageRewardContext,
        tools: tuple[dict[str, str], ...],
    ) -> str: ...


class StaticStageRewardCodeAgent:
    """Deterministic code agent used by tests and simulation fixtures."""

    def __init__(self, code: str) -> None:
        self.code = code

    def generate(
        self,
        context: StageRewardContext,
        tools: tuple[dict[str, str], ...],
    ) -> str:
        del context, tools
        return self.code


class ModelStageRewardCodeAgent:
    """Model-backed agent that writes a bounded Stage reward function."""

    def __init__(self, model_args: Any) -> None:
        self.model_args = model_args

    def generate(
        self,
        context: StageRewardContext,
        tools: tuple[dict[str, str], ...],
    ) -> str:
        from capx.llm.client import query_model

        payload = {
            "turn": {
                "turn_id": context.turn.turn_id,
                "stage_id": context.turn.stage_id,
                "role": context.turn.role.value,
            },
            "call": _json_value(context.call),
            "result": _json_value(context.result),
            "state_before": _json_value(context.state_before),
            "state_after": _json_value(context.state_after),
            "rules": [_json_value(rule) for rule in context.rules],
            "tools": list(tools),
        }
        instructions = (
            "Write one Python function named compute_reward with signature "
            "compute_reward(state_before, state_after, tool_result, tools). "
            "It must return a dict with reward (0.0 or 1.0), code, evidence, and optional "
            "message, reward_valid, or unsafe. When there are multiple rules, also return "
            "rule_rewards mapping each predicate to 0.0 or 1.0. All rules must pass for "
            "reward 1.0. "
            "Use only the supplied dictionaries, safe builtins, and allowed tools. "
            "Do not import modules or mutate inputs. Return only Python code.\n\n"
            + json.dumps(payload, sort_keys=True)
        )
        response = query_model(
            self.model_args,
            [{"role": "user", "content": [{"type": "text", "text": instructions}]}],
        )
        content = response.get("content", "") if isinstance(response, Mapping) else response
        return _extract_python(str(content))


class RewardCodeError(RuntimeError):
    """Generated reward code was rejected or failed during execution."""


class RewardCodeTimeout(RewardCodeError):
    """Generated reward code exceeded its execution deadline."""


class _RewardCodeValidator(ast.NodeVisitor):
    _forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.For,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    _forbidden_names = {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
    _read_only_methods = {"count", "get", "items", "keys", "values"}

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, self._forbidden_nodes):
            raise RewardCodeError(f"Forbidden syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            raise RewardCodeError("Reward code must contain exactly one function")
        function = node.body[0]
        if function.name != "compute_reward":
            raise RewardCodeError("Reward function must be named compute_reward")
        if [argument.arg for argument in function.args.args] != [
            "state_before",
            "state_after",
            "tool_result",
            "tools",
        ]:
            raise RewardCodeError("compute_reward has an invalid signature")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self._forbidden_names or node.id.startswith("__"):
            raise RewardCodeError(f"Forbidden name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise RewardCodeError(f"Forbidden attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "compute_reward":
            raise RewardCodeError("Recursive reward code is not allowed")
        if isinstance(node.func, ast.Attribute) and node.func.attr not in self._read_only_methods:
            raise RewardCodeError(f"Forbidden method call: {node.func.attr}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Attribute | ast.Subscript):
                raise RewardCodeError("Reward code cannot mutate attributes or containers")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Attribute | ast.Subscript):
            raise RewardCodeError("Reward code cannot mutate attributes or containers")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        raise RewardCodeError("Augmented assignment is not allowed")


class ReadOnlyStageRewardSandbox:
    """Validate and execute agent-authored reward code against copied inputs."""

    _safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "round": round,
        "str": str,
        "sum": sum,
    }

    def __init__(
        self,
        tools: VerificationToolRegistry | None = None,
        *,
        timeout_s: float = 2.0,
        max_tool_calls: int = 8,
        max_code_chars: int = 12_000,
    ) -> None:
        if timeout_s <= 0 or max_tool_calls < 0 or max_code_chars < 1:
            raise ValueError("Reward sandbox limits must be positive")
        self.tools = tools or VerificationToolRegistry()
        self.timeout_s = timeout_s
        self.max_tool_calls = max_tool_calls
        self.max_code_chars = max_code_chars

    def tool_descriptions(self, context: StageRewardContext) -> tuple[dict[str, str], ...]:
        return self.tools.descriptions(_allowed_tools(context.rules))

    def execute(self, code: str, context: StageRewardContext) -> StageReward:
        if len(code) > self.max_code_chars:
            raise RewardCodeError("Reward code exceeds max_code_chars")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise RewardCodeError(f"Invalid reward code: {exc.msg}") from exc
        _RewardCodeValidator().visit(tree)

        tool_trace: list[dict[str, Any]] = []
        allowed = _allowed_tools(context.rules)
        tool_functions: dict[str, Callable[..., Any]] = {}
        for name in allowed:
            handler = self.tools.handler(name)

            def tool_wrapper(
                tool_name: str,
                tool_handler: Callable[..., Any],
            ) -> Callable[..., Any]:
                def call_tool(*args: Any, **kwargs: Any) -> Any:
                    if len(tool_trace) >= self.max_tool_calls:
                        raise RewardCodeError("Verification tool call budget exhausted")
                    clean_args = copy.deepcopy(args)
                    clean_kwargs = copy.deepcopy(kwargs)
                    output = tool_handler(*clean_args, **clean_kwargs)
                    clean_output = _json_value(output)
                    tool_trace.append(
                        {
                            "tool": tool_name,
                            "args": _json_value(clean_args),
                            "kwargs": _json_value(clean_kwargs),
                            "output": clean_output,
                        }
                    )
                    return copy.deepcopy(clean_output)

                return call_tool

            tool_functions[name] = tool_wrapper(name, handler)

        namespace: dict[str, Any] = {"__builtins__": self._safe_builtins}
        try:
            with _execution_deadline(self.timeout_s):
                exec(compile(tree, "<stage_reward>", "exec"), namespace, namespace)
                output = namespace["compute_reward"](
                    copy.deepcopy(context.state_before),
                    copy.deepcopy(context.state_after),
                    _json_value(context.result),
                    tool_functions,
                )
        except RewardCodeError:
            raise
        except TimeoutError as exc:
            raise RewardCodeTimeout("Reward code timed out") from exc
        except Exception as exc:
            raise RewardCodeError(f"Reward code failed: {type(exc).__name__}: {exc}") from exc

        return self._reward_from_output(output, context, code, tuple(tool_trace))

    @staticmethod
    def _reward_from_output(
        output: Any,
        context: StageRewardContext,
        code: str,
        tool_trace: tuple[dict[str, Any], ...],
    ) -> StageReward:
        if not isinstance(output, Mapping):
            raise RewardCodeError("compute_reward must return a dictionary")
        raw_reward = output.get("reward")
        if isinstance(raw_reward, bool) or raw_reward not in {0, 0.0, 1, 1.0}:
            raise RewardCodeError("Generated reward must be 0.0 or 1.0")
        reward = float(raw_reward)
        if len(context.rules) > 1:
            raw_rule_rewards = output.get("rule_rewards")
            if not isinstance(raw_rule_rewards, Mapping):
                raise RewardCodeError("Multiple rules require a rule_rewards dictionary")
            predicates = {rule.predicate for rule in context.rules}
            if set(raw_rule_rewards) != predicates:
                raise RewardCodeError("rule_rewards must contain every predicate exactly once")
            values = tuple(raw_rule_rewards[predicate] for predicate in predicates)
            if any(isinstance(value, bool) or value not in {0, 0.0, 1, 1.0} for value in values):
                raise RewardCodeError("Each rule reward must be 0.0 or 1.0")
            combined_reward = float(all(float(value) == 1.0 for value in values))
            if reward != combined_reward:
                raise RewardCodeError("Stage reward must be the conjunction of all rule rewards")
        evidence = output.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise RewardCodeError("Generated evidence must be a dictionary")
        missing = sorted(
            {
                key
                for rule in context.rules
                for key in rule.required_evidence
                if key not in evidence
            }
        )
        reward_valid = bool(output.get("reward_valid", True)) and not missing
        unsafe = bool(output.get("unsafe", False))
        if unsafe and reward != 0.0:
            raise RewardCodeError("Unsafe reward output must use reward 0.0")
        if not reward_valid and reward != 0.0:
            raise RewardCodeError("Invalid reward output must use reward 0.0")
        if unsafe:
            status = RewardStatus.UNSAFE
        elif not reward_valid:
            status = RewardStatus.UNKNOWN
        elif reward == 1.0:
            status = RewardStatus.SUCCESS
        else:
            status = RewardStatus.FAILURE
        message = str(output.get("message", ""))
        if missing:
            message = f"Missing required evidence: {', '.join(missing)}"
        return StageReward(
            status=status,
            code=str(output.get("code", status.value)),
            message=message,
            evidence=dict(_json_value(evidence)),
            reward=reward,
            reward_valid=reward_valid,
            generated_code=code,
            tool_trace=tool_trace,
        )


class AgenticStageRewardVerifier:
    """Ask an agent for concrete code, then execute it in a read-only sandbox."""

    def __init__(
        self,
        code_agent: StageRewardCodeAgent,
        sandbox: ReadOnlyStageRewardSandbox | None = None,
    ) -> None:
        self.code_agent = code_agent
        self.sandbox = sandbox or ReadOnlyStageRewardSandbox()

    def compute_reward(self, context: StageRewardContext) -> StageReward:
        if not context.rules:
            return StageReward(
                status=RewardStatus.UNKNOWN,
                code="missing_reward_rule",
                message="ToolCall has no Stage reward rule",
            )
        try:
            tools = self.sandbox.tool_descriptions(context)
            code = self.code_agent.generate(context, tools)
            return self.sandbox.execute(code, context)
        except (RewardCodeError, ValueError) as exc:
            return StageReward(
                status=RewardStatus.UNKNOWN,
                code="reward_code_error",
                message=str(exc),
                generated_code=code if "code" in locals() else None,
            )
        except Exception as exc:
            return StageReward(
                status=RewardStatus.UNKNOWN,
                code="reward_agent_error",
                message=f"{type(exc).__name__}: {exc}",
            )


def _allowed_tools(rules: tuple[StageRewardRule, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for rule in rules for name in rule.allowed_tools))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    return repr(value)


def _extract_python(response: str) -> str:
    text = response.strip()
    if "```" not in text:
        return text
    blocks = text.split("```")
    if len(blocks) < 3:
        return text
    code = blocks[1]
    if code.lstrip().startswith("python"):
        code = code.lstrip()[len("python") :]
    return code.strip()


@contextmanager
def _execution_deadline(seconds: float):
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def timeout_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


__all__ = [
    "AgenticStageRewardVerifier",
    "ModelStageRewardCodeAgent",
    "ReadOnlyStageRewardSandbox",
    "ReadOnlyVerificationTool",
    "RewardCodeError",
    "RewardCodeTimeout",
    "RewardStatus",
    "StageReward",
    "StageRewardCodeAgent",
    "StageRewardContext",
    "StageRewardRule",
    "StaticStageRewardCodeAgent",
    "VerificationToolRegistry",
]
