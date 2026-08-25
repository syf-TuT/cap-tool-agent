from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from capx.rl.capsule.controller import (
    ControllerProtocolError,
    ControllerRepairCollector,
    FrozenControllerConfig,
    OpenAICompatibleControllerTransport,
    python_base_unit_spans,
)
from capx.rl.capsule.group import CollectionInfrastructureError, ProgramCandidate
from capx.rl.capsule.schema import ProgramReplayResultV1, ReplayOutcome, TaskInstanceV1


def _task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="Stack the cubes.",
        environment="robosuite_cube_stack",
        api="franka_privileged",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def _failure(p0: ProgramCandidate) -> ProgramReplayResultV1:
    task = _task()
    return ProgramReplayResultV1(
        task_id=task.task_id,
        environment_seed=task.environment_seed,
        program_sample_id=p0.program_sample_id,
        source=p0.source,
        initial_state_sha256=task.initial_state_sha256,
        outcome=ReplayOutcome.PROGRAM_ERROR,
        raw_reward=0.0,
        binary_reward=0.0,
        task_completed=False,
        error_type="NameError",
        error_message="missing helper",
        diagnostics={"stderr": "NameError: missing helper"},
    )


class _ScriptedTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[dict[str, str], ...]] = []

    def complete(self, messages: tuple[dict[str, str], ...]) -> str:
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, str)
        return response


def _action(action: str, **fields: object) -> str:
    return json.dumps({"action": action, **fields})


def test_python_base_units_are_stable_top_level_statement_spans() -> None:
    source = "# keep this comment\nx = 1\n\ndef helper():\n    return x\n"

    spans = python_base_unit_spans(source)

    assert [span.unit_id for span in spans] == ["group_0", "group_1"]
    assert [source[span.start_offset : span.end_offset] for span in spans] == [
        "x = 1",
        "def helper():\n    return x",
    ]


@pytest.mark.parametrize("source", ["if (", "# comments only\n", ""])
def test_python_base_units_fall_back_to_whole_program_for_unparseable_source(source: str) -> None:
    spans = python_base_unit_spans(source)

    assert len(spans) == 1
    assert spans[0].unit_id == "program"
    assert spans[0].start_offset == 0
    assert spans[0].end_offset == len(source)


def test_collector_runs_edit_sequence_without_replay_and_reconstructs_pt() -> None:
    p0 = ProgramCandidate("base-0", "broken = True\n")
    transport = _ScriptedTransport(
        [
            _action(
                "append",
                generation_id="recovery_1",
                unit_id="body",
                source="recover = False\n",
                rationale="add recovery",
            ),
            _action(
                "replace",
                target="base:group_0",
                source="broken = False",
                rationale="fix original",
            ),
            _action(
                "replace",
                target="recovery:recovery_1:body",
                source="recover = True\n",
                rationale="fix appended code",
            ),
            _action("finish", rationale="complete"),
        ]
    )
    collector = ControllerRepairCollector(transport=transport, max_turns=12)

    trace = collector(_task(), p0, _failure(p0), 0, 0, "repair-0")

    assert len(transport.calls) == 4
    assert [edit.target for edit in trace.edits] == [
        "recovery:recovery_1:body",
        "base:group_0",
        "recovery:recovery_1:body",
    ]
    assert trace.final_source == "broken = False\n\nrecover = True\n"
    assert trace.reconstruct() == trace.final_source
    assert trace.audits[-1].event_type == "finish"
    joined_prompts = "\n".join(message["content"] for call in transport.calls for message in call)
    assert "Do not execute" in joined_prompts
    assert "repair-0" in joined_prompts
    assert "initial_state_sha256" not in joined_prompts
    assert "NameError: missing helper" in joined_prompts


def test_invalid_and_inspect_actions_only_enter_audit() -> None:
    transport = _ScriptedTransport(
        [
            "not-json",
            _action("inspect", message="show current targets"),
            _action("finish"),
        ]
    )
    collector = ControllerRepairCollector(transport=transport, max_turns=3)

    p0 = ProgramCandidate("base-0", "x = 1\n")
    trace = collector(_task(), p0, _failure(p0), 0, 0, "repair-0")

    assert trace.edits == ()
    assert [audit.event_type for audit in trace.audits] == [
        "parse_failure",
        "inspect",
        "finish",
    ]
    assert trace.final_source == trace.base_source


def test_transport_failure_is_typed_as_collection_infrastructure_error() -> None:
    collector = ControllerRepairCollector(
        transport=_ScriptedTransport([TimeoutError("controller timeout")]),
        max_turns=1,
    )

    with pytest.raises(CollectionInfrastructureError, match="controller request failed"):
        p0 = ProgramCandidate("base-0", "x = 1\n")
        collector(_task(), p0, _failure(p0), 0, 0, "repair-0")


def test_non_text_transport_response_is_protocol_error() -> None:
    class _BadTransport:
        def complete(self, messages):
            del messages
            return {"action": "finish"}

    collector = ControllerRepairCollector(transport=_BadTransport(), max_turns=1)

    with pytest.raises(ControllerProtocolError, match="text"):
        p0 = ProgramCandidate("base-0", "x = 1\n")
        collector(_task(), p0, _failure(p0), 0, 0, "repair-0")


def test_frozen_controller_config_is_strict() -> None:
    config = FrozenControllerConfig(
        endpoint="http://controller.invalid/v1",
        model="controller-model",
        api_key_env="CAPX_CONTROLLER_API_KEY",
    )
    assert config.frozen is True
    assert config.request_timeout_s == 300.0
    assert config.max_output_tokens == 512

    with pytest.raises(ValueError, match="frozen"):
        FrozenControllerConfig(
            endpoint="http://controller.invalid/v1",
            model="controller-model",
            api_key_env="CAPX_CONTROLLER_API_KEY",
            frozen=False,
        )
    with pytest.raises(ValueError, match="max_turns"):
        FrozenControllerConfig(
            endpoint="http://controller.invalid/v1",
            model="controller-model",
            api_key_env="CAPX_CONTROLLER_API_KEY",
            max_turns=13,
        )
    for invalid in (True, 0, -1):
        with pytest.raises((TypeError, ValueError), match="max_output_tokens"):
            FrozenControllerConfig(
                endpoint="http://controller.invalid/v1",
                model="controller-model",
                api_key_env="CAPX_CONTROLLER_API_KEY",
                max_output_tokens=invalid,
            )


def test_openai_transport_is_lazy_and_uses_dedicated_credentials(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"finish"}'))]
            )

    class _Client:
        chat = SimpleNamespace(completions=_Completions())

    client_construction: list[dict[str, str]] = []

    def factory(**kwargs):
        client_construction.append(kwargs)
        return _Client()

    monkeypatch.setenv("CAPX_CONTROLLER_API_KEY", "controller-secret")
    config = FrozenControllerConfig(
        endpoint="http://controller.invalid/v1",
        model="controller-model",
        api_key_env="CAPX_CONTROLLER_API_KEY",
    )
    transport = OpenAICompatibleControllerTransport(config, client_factory=factory)
    assert client_construction == []

    response = transport.complete(({"role": "user", "content": "repair"},))

    assert response == '{"action":"finish"}'
    assert client_construction == [
        {
            "api_key": "controller-secret",
            "base_url": "http://controller.invalid/v1",
            "timeout": 300.0,
        }
    ]
    assert calls == [
        {
            "model": "controller-model",
            "messages": [{"role": "user", "content": "repair"}],
            "temperature": 0.7,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
    ]
