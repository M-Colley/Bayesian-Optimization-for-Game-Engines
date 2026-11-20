"""Headless Python client and runner for the loopback NDJSON optimization backends.

Drives ``bo.py``, ``mobo.py``, ``dbo.py``, ``cabop_bo.py``, ``cabop_mobo.py`` and
``meta_mobo.py`` from Python simulation evaluators (e.g. physics, biomechanics, or
synthetic benchmarks) over the same newline-delimited UTF-8 JSON wire protocol used
by the Unity and Unreal Engine integrations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from bo_protocol import (  # noqa: E402
    HOST,
    RECV_CHUNK_BYTES,
    _ReceiveBuffer,
    close_connection,
    validate_raw_samples,
)

_RESERVED_COLUMNS = frozenset(
    {
        "UserID",
        "ConditionID",
        "GroupID",
        "Timestamp",
        "Iteration",
        "Phase",
        "IsBest",
        "IsPareto",
        "Context",
    }
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class ParameterSpec:
    """Continuous decision variable ``[low, high]``."""

    key: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("ParameterSpec.key must be a non-empty string")
        lo = _finite_number(self.low, f"parameter '{self.key}' low")
        hi = _finite_number(self.high, f"parameter '{self.key}' high")
        if lo >= hi:
            raise ValueError(
                f"Parameter '{self.key}' requires low < high, got ({lo}, {hi})"
            )
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def to_wire(self, order: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "init": {"low": self.low, "high": self.high},
            "optSeqOrder": int(order),
        }


@dataclass(frozen=True)
class ObjectiveSpec:
    """Bounded scalar objective ``[low, high]`` with minimize/maximize sense."""

    key: str
    low: float
    high: float
    minimize: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("ObjectiveSpec.key must be a non-empty string")
        if not isinstance(self.minimize, bool):
            raise TypeError("ObjectiveSpec.minimize must be a bool")
        lo = _finite_number(self.low, f"objective '{self.key}' low")
        hi = _finite_number(self.high, f"objective '{self.key}' high")
        if lo >= hi:
            raise ValueError(
                f"Objective '{self.key}' requires low < high, got ({lo}, {hi})"
            )
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def to_wire(self, order: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "init": {
                "low": self.low,
                "high": self.high,
                "minimize": 1 if self.minimize else 0,
            },
            "optSeqOrder": int(order),
        }


def backend_entrypoint(optimizer_backend: str, objective_count: int) -> str:
    """Resolve the canonical ``Backend/*.py`` script for *(backend, n_objectives)*."""
    if isinstance(objective_count, bool) or objective_count < 1:
        raise ValueError(f"objective_count must be >= 1, got {objective_count!r}")
    name = str(optimizer_backend).strip().lower()
    single = objective_count == 1
    if name == "botorch":
        return "bo.py" if single else "mobo.py"
    if name == "cabop":
        return "cabop_bo.py" if single else "cabop_mobo.py"
    if name == "dbo":
        if not single:
            raise ValueError("dbo supports single-objective studies only")
        return "dbo.py"
    if name == "meta-taf":
        if single:
            raise ValueError("meta-taf requires at least 2 objectives")
        return "meta_mobo.py"
    raise ValueError(
        f"Unknown optimizer_backend {optimizer_backend!r}; expected "
        "'botorch', 'cabop', 'dbo' or 'meta-taf'."
    )


@dataclass(frozen=True)
class StudyConfig:
    """Complete study specification matching the backend ``init`` protocol."""

    parameters: tuple[ParameterSpec, ...]
    objectives: tuple[ObjectiveSpec, ...]
    optimizer_backend: str = "botorch"
    num_sampling_iterations: int = 5
    num_optimization_iterations: int = 10
    num_restarts: int = 10
    raw_samples: int = 1024
    mc_samples: int = 128
    seed: int = 3
    warm_start: bool = False
    warm_start_objective_format: str = "auto"
    initial_parameters_data_path: str = ""
    initial_objectives_data_path: str = ""
    user_id: str = "sim"
    condition_id: str = "default"
    group_id: str = "0"
    context: Mapping[str, Any] | None = None
    extra_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params = tuple(self.parameters)
        objs = tuple(self.objectives)
        if not params:
            raise ValueError("StudyConfig requires at least one ParameterSpec")
        if not objs:
            raise ValueError("StudyConfig requires at least one ObjectiveSpec")
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "objectives", objs)

        p_keys = [p.key for p in params]
        o_keys = [o.key for o in objs]
        if len(set(p_keys)) != len(p_keys):
            raise ValueError("Duplicate parameter keys in StudyConfig")
        if len(set(o_keys)) != len(o_keys):
            raise ValueError("Duplicate objective keys in StudyConfig")
        overlap = sorted(set(p_keys).intersection(o_keys))
        if overlap:
            raise ValueError(f"Parameter and objective keys overlap: {overlap}")
        reserved = sorted(set(p_keys + o_keys).intersection(_RESERVED_COLUMNS))
        if reserved:
            raise ValueError(
                f"Parameter/objective keys collide with reserved log columns: {reserved}"
            )

        if self.num_sampling_iterations < 0 or self.num_optimization_iterations < 0:
            raise ValueError("Iteration counts must be non-negative")
        if not self.warm_start and self.num_sampling_iterations < 1:
            raise ValueError(
                "num_sampling_iterations must be >= 1 when warm_start is False"
            )
        if self.num_restarts < 1 or self.raw_samples < 1 or self.mc_samples < 1:
            raise ValueError("num_restarts, raw_samples and mc_samples must be >= 1")
        validate_raw_samples(self.num_restarts, self.raw_samples)
        backend_entrypoint(self.optimizer_backend, len(objs))

    def entrypoint(self) -> str:
        return backend_entrypoint(self.optimizer_backend, len(self.objectives))

    def to_init_message(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "optimizerBackend": self.optimizer_backend,
            "nParameters": len(self.parameters),
            "nObjectives": len(self.objectives),
            "numSamplingIterations": int(self.num_sampling_iterations),
            "numOptimizationIterations": int(self.num_optimization_iterations),
            "batchSize": 1,
            "numRestarts": int(self.num_restarts),
            "rawSamples": int(self.raw_samples),
            "mcSamples": int(self.mc_samples),
            "seed": int(self.seed),
            "warmStart": bool(self.warm_start),
            "warmStartObjectiveFormat": str(self.warm_start_objective_format),
            "initialParametersDataPath": str(self.initial_parameters_data_path),
            "initialObjectivesDataPath": str(self.initial_objectives_data_path),
        }
        cfg.update(dict(self.extra_config))
        msg: dict[str, Any] = {
            "type": "init",
            "config": cfg,
            "parameters": [p.to_wire(i) for i, p in enumerate(self.parameters)],
            "objectives": [o.to_wire(i) for i, o in enumerate(self.objectives)],
            "user": {
                "userId": str(self.user_id),
                "conditionId": str(self.condition_id),
                "groupId": str(self.group_id),
            },
        }
        if self.context is not None:
            msg["context"] = dict(self.context)
        return msg


@dataclass(frozen=True)
class TrialRecord:
    """One completed parameter -> objective evaluation."""

    iteration: int
    parameters: dict[str, float]
    objectives: dict[str, float]
    coverage: float | None = None


@dataclass(frozen=True)
class StudyResult:
    """Completed or early-stopped optimization run."""

    trials: tuple[TrialRecord, ...]
    coverages: tuple[float, ...]
    finished: bool
    stopped_early: bool

    def pareto_trials(
        self, objectives: Sequence[ObjectiveSpec]
    ) -> tuple[TrialRecord, ...]:
        """Return non-dominated trials respecting each objective's sense."""
        if not self.trials:
            return ()
        oriented: list[tuple[float, ...]] = []
        for trial in self.trials:
            oriented.append(
                tuple(
                    -trial.objectives[o.key] if o.minimize else trial.objectives[o.key]
                    for o in objectives
                )
            )
        front: list[TrialRecord] = []
        seen: set[tuple[float, ...]] = set()
        for i, cand in enumerate(oriented):
            if cand in seen:
                continue
            dominated = False
            for j, other in enumerate(oriented):
                if i == j:
                    continue
                if all(b >= a for a, b in zip(cand, other)) and any(
                    b > a for a, b in zip(cand, other)
                ):
                    dominated = True
                    break
            if not dominated:
                seen.add(cand)
                front.append(self.trials[i])
        return tuple(front)


def _send_line(conn: Any, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(encoded)


def _recv_message(conn: Any, buf: _ReceiveBuffer) -> dict[str, Any] | None:
    while True:
        while buf.lines:
            raw = buf.lines.popleft().decode("utf-8", errors="replace").rstrip("\r")
            if not raw.strip():
                continue
            msg = json.loads(raw)
            if isinstance(msg, dict):
                return msg
        chunk = conn.recv(RECV_CHUNK_BYTES)
        if not chunk:
            trailing = b"".join(buf.partial).decode("utf-8", errors="replace").strip()
            buf.reset()
            if trailing:
                raise RuntimeError(
                    f"Connection closed with unterminated JSON data: {trailing[:120]!r}"
                )
            return None
        buf.feed(chunk)


EvaluatorFn = Callable[[dict[str, float], int], Mapping[str, float]]
StopFn = Callable[[TrialRecord], str | bool | None]


def run_connected_study(
    conn: Any,
    study: StudyConfig,
    evaluator: EvaluatorFn,
    *,
    stop_when: StopFn | None = None,
) -> StudyResult:
    """Drive an already-connected backend socket from *evaluator*."""
    buf = _ReceiveBuffer()
    expected_p = {p.key for p in study.parameters}
    obj_specs = {o.key: o for o in study.objectives}
    expected_o = set(obj_specs)

    _send_line(conn, study.to_init_message())
    trials: list[TrialRecord] = []
    coverages: list[float] = []
    finished = False
    stopped_early = False

    while True:
        msg = _recv_message(conn, buf)
        if msg is None:
            if not finished and not stopped_early:
                raise ConnectionError(
                    "Backend closed the connection before optimization_finished"
                )
            break

        mtype = msg.get("type")
        if mtype == "parameters":
            iteration = int(msg["iteration"])
            raw_vals = msg.get("values") or {}
            if set(raw_vals) != expected_p:
                raise ValueError(
                    f"Backend sent parameter keys {sorted(raw_vals)}, "
                    f"expected {sorted(expected_p)}"
                )
            params = {k: _finite_number(raw_vals[k], k) for k in study.parameters_keys()}
            evaluated = evaluator(dict(params), iteration)
            if set(evaluated) != expected_o:
                raise ValueError(
                    f"Evaluator returned keys {sorted(evaluated)}, "
                    f"expected {sorted(expected_o)}"
                )
            clean_obj: dict[str, float] = {}
            for key, spec in obj_specs.items():
                val = _finite_number(evaluated[key], f"objective '{key}'")
                if not (spec.low <= val <= spec.high):
                    raise ValueError(
                        f"Objective '{key}' value {val} is outside [{spec.low}, {spec.high}]"
                    )
                clean_obj[key] = val

            _send_line(conn, {"type": "objectives", "values": clean_obj})
            record = TrialRecord(
                iteration=iteration, parameters=params, objectives=clean_obj
            )
            trials.append(record)

            if stop_when is not None:
                decision = stop_when(record)
                if decision:
                    reason = (
                        decision
                        if isinstance(decision, str)
                        else f"stop_when triggered at iteration {iteration}"
                    )
                    _send_line(conn, {"type": "stop", "reason": reason})
                    stopped_early = True
                    break

        elif mtype in ("coverage", "tempCoverage"):
            cov = _finite_number(msg.get("value"), mtype)
            coverages.append(cov)
            if trials and trials[-1].coverage is None:
                last = trials[-1]
                trials[-1] = TrialRecord(
                    iteration=last.iteration,
                    parameters=last.parameters,
                    objectives=last.objectives,
                    coverage=cov,
                )
        elif mtype == "optimization_finished":
            finished = True
            break

    return StudyResult(
        trials=tuple(trials),
        coverages=tuple(coverages),
        finished=finished,
        stopped_early=stopped_early,
    )


def _study_parameter_keys(self: StudyConfig) -> tuple[str, ...]:
    return tuple(p.key for p in self.parameters)


StudyConfig.parameters_keys = _study_parameter_keys  # type: ignore[attr-defined]


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def run_subprocess_study(
    study: StudyConfig,
    evaluator: EvaluatorFn,
    *,
    backend_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    python_executable: str | None = None,
    port: int = 0,
    accept_timeout_sec: float = 30.0,
    socket_timeout_sec: float = 600.0,
    stop_when: StopFn | None = None,
) -> StudyResult:
    """Spawn the matching backend script on loopback TCP and run *study*."""
    root = Path(_SCRIPT_DIR) if backend_dir is None else Path(backend_dir).resolve()
    script = root / study.entrypoint()
    if not script.is_file():
        raise FileNotFoundError(f"Backend entrypoint not found: {script}")

    chosen_port = _allocate_loopback_port() if port == 0 else int(port)
    workdir = Path.cwd() if log_dir is None else Path(log_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    env = dict(
        os.environ,
        BO_PORT=str(chosen_port),
        BO_ACCEPT_TIMEOUT_SEC=str(accept_timeout_sec),
        BO_SOCKET_TIMEOUT_SEC=str(socket_timeout_sec),
        PYTHONDONTWRITEBYTECODE="1",
    )
    proc = subprocess.Popen(
        [python_executable or sys.executable, str(script)],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    conn: socket.socket | None = None
    try:
        deadline = time.monotonic() + accept_timeout_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout is not None else ""
                raise RuntimeError(
                    f"Backend exited prematurely with code {proc.returncode}:\n{output}"
                )
            try:
                conn = socket.create_connection((HOST, chosen_port), timeout=0.2)
                break
            except OSError:
                time.sleep(0.05)
        if conn is None:
            raise TimeoutError(
                f"Backend {script.name} did not listen on {HOST}:{chosen_port} "
                f"within {accept_timeout_sec}s"
            )
        conn.settimeout(socket_timeout_sec)
        result = run_connected_study(conn, study, evaluator, stop_when=stop_when)
        close_connection(conn)
        conn = None
        rc = proc.wait(timeout=30.0)
        if rc != 0:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"Backend exited with code {rc}:\n{output}")
        return result
    finally:
        if conn is not None:
            close_connection(conn)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if proc.stdout is not None:
            proc.stdout.close()
