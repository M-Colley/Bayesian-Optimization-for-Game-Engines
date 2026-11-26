"""Tests for the headless Python loopback evaluator client (Backend/bo_client.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

REPO_ROOT = _TESTS_DIR.parent
CLIENT_PATH = REPO_ROOT / "Backend" / "bo_client.py"


def load_client_module():
    spec = importlib.util.spec_from_file_location("bo_client_under_test", CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class BoClientTests(unittest.TestCase):
    def test_spec_and_study_validation(self) -> None:
        client = load_client_module()
        p = client.ParameterSpec("tilt_deg", 0.0, 60.0)
        o_min = client.ObjectiveSpec("effort", 0.0, 10.0, minimize=True)
        o_max = client.ObjectiveSpec("throughput", 0.0, 12.0, minimize=False)

        study = client.StudyConfig(
            parameters=(p,),
            objectives=(o_min, o_max),
            num_sampling_iterations=3,
            num_optimization_iterations=2,
            num_restarts=2,
            raw_samples=16,
            mc_samples=8,
            seed=7,
        )
        self.assertEqual(study.entrypoint(), "mobo.py")
        init = study.to_init_message()
        self.assertEqual(init["type"], "init")
        self.assertEqual(init["config"]["nParameters"], 1)
        self.assertEqual(init["config"]["nObjectives"], 2)
        self.assertEqual(init["objectives"][0]["init"]["minimize"], 1)
        self.assertEqual(init["objectives"][1]["init"]["minimize"], 0)

        with self.assertRaises(ValueError):
            client.ParameterSpec("bad", 5.0, 5.0)
        with self.assertRaises(ValueError):
            client.StudyConfig(
                parameters=(client.ParameterSpec("Phase", 0.0, 1.0),),
                objectives=(o_min,),
            )
        with self.assertRaises(ValueError):
            client.StudyConfig(
                parameters=(p,),
                objectives=(o_min, o_max),
                optimizer_backend="dbo",
            )

    def test_connected_multi_objective_loop_and_pareto_filter(self) -> None:
        client = load_client_module()
        study = client.StudyConfig(
            parameters=(
                client.ParameterSpec("distance_m", 0.3, 0.7),
                client.ParameterSpec("tilt_rad", 0.0, 1.0),
            ),
            objectives=(
                client.ObjectiveSpec("effort", 0.0, 5.0, minimize=True),
                client.ObjectiveSpec("throughput", 0.0, 10.0, minimize=False),
            ),
            num_sampling_iterations=2,
            num_optimization_iterations=2,
            num_restarts=2,
            raw_samples=16,
            mc_samples=8,
        )

        designs = [
            {"distance_m": 0.35, "tilt_rad": 0.2},
            {"distance_m": 0.65, "tilt_rad": 0.8},
            {"distance_m": 0.45, "tilt_rad": 0.4},
            {"distance_m": 0.50, "tilt_rad": 0.5},
        ]
        server_sock, client_sock = socket.socketpair()
        received_init: list[dict] = []

        def fake_backend() -> None:
            try:
                reader = server_sock.makefile("rb")
                init_line = reader.readline()
                received_init.append(json.loads(init_line.decode("utf-8")))
                for i, design in enumerate(designs, start=1):
                    server_sock.sendall(
                        (
                            json.dumps(
                                {"type": "parameters", "iteration": i, "values": design}
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    reply = json.loads(reader.readline().decode("utf-8"))
                    self.assertEqual(reply["type"], "objectives")
                    cov = 0.2 * i
                    mtype = "tempCoverage" if i <= 2 else "coverage"
                    server_sock.sendall(
                        (json.dumps({"type": mtype, "value": cov}) + "\n").encode(
                            "utf-8"
                        )
                    )
                server_sock.sendall(b'{"type": "optimization_finished"}\n')
            finally:
                server_sock.close()

        worker = threading.Thread(target=fake_backend, daemon=True)
        worker.start()
        try:
            def evaluate(params: dict[str, float], iteration: int) -> dict[str, float]:
                del iteration
                d = params["distance_m"]
                t = params["tilt_rad"]
                return {
                    "effort": (d - 0.35) ** 2 + (t - 0.2) ** 2,
                    "throughput": 8.0 - 4.0 * (d - 0.50) ** 2 - 2.0 * (t - 0.5) ** 2,
                }

            result = client.run_connected_study(client_sock, study, evaluate)
        finally:
            client_sock.close()
            worker.join(timeout=5.0)

        self.assertTrue(result.finished)
        self.assertFalse(result.stopped_early)
        self.assertEqual(len(result.trials), 4)
        self.assertEqual(len(received_init), 1)
        self.assertAlmostEqual(result.trials[-1].coverage, 0.8)

        front = result.pareto_trials(study.objectives)
        front_iters = [t.iteration for t in front]
        # Design 1 minimizes effort; design 4 maximizes throughput; design 2 is dominated by both.
        self.assertIn(1, front_iters)
        self.assertIn(4, front_iters)
        self.assertNotIn(2, front_iters)

    def test_early_stop_sends_stop_message_cleanly(self) -> None:
        client = load_client_module()
        study = client.StudyConfig(
            parameters=(client.ParameterSpec("x", 0.0, 1.0),),
            objectives=(client.ObjectiveSpec("loss", 0.0, 10.0, minimize=True),),
            num_sampling_iterations=2,
            num_optimization_iterations=2,
            num_restarts=2,
            raw_samples=16,
        )
        server_sock, client_sock = socket.socketpair()
        stop_messages: list[dict] = []

        def fake_backend() -> None:
            try:
                reader = server_sock.makefile("rb")
                _ = reader.readline()  # init
                server_sock.sendall(
                    b'{"type":"parameters","iteration":1,"values":{"x":0.25}}\n'
                )
                _ = reader.readline()  # objectives
                stop_line = reader.readline()
                if stop_line:
                    stop_messages.append(json.loads(stop_line.decode("utf-8")))
            finally:
                server_sock.close()

        worker = threading.Thread(target=fake_backend, daemon=True)
        worker.start()
        try:
            result = client.run_connected_study(
                client_sock,
                study,
                lambda p, _it: {"loss": p["x"]},
                stop_when=lambda trial: "target_reached"
                if trial.objectives["loss"] <= 0.3
                else None,
            )
        finally:
            client_sock.close()
            worker.join(timeout=5.0)

        self.assertTrue(result.stopped_early)
        self.assertFalse(result.finished)
        self.assertEqual(len(result.trials), 1)
        self.assertEqual(stop_messages[0]["type"], "stop")
        self.assertEqual(stop_messages[0]["reason"], "target_reached")

    def test_subprocess_runner_executes_against_loopback_server_script(self) -> None:
        client = load_client_module()
        study = client.StudyConfig(
            parameters=(client.ParameterSpec("width", 0.02, 0.10),),
            objectives=(client.ObjectiveSpec("score", 0.0, 1.0, minimize=False),),
            num_sampling_iterations=2,
            num_optimization_iterations=0,
            num_restarts=1,
            raw_samples=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bo = tmp_path / "bo.py"
            fake_bo.write_text(
                "import json, os, socket\n"
                "port = int(os.environ['BO_PORT'])\n"
                "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "s.bind(('127.0.0.1', port))\n"
                "s.listen(1)\n"
                "conn, _ = s.accept()\n"
                "s.close()\n"
                "r = conn.makefile('rb')\n"
                "_ = json.loads(r.readline().decode('utf-8'))\n"
                "for it, w in ((1, 0.04), (2, 0.08)):\n"
                "    conn.sendall((json.dumps({'type':'parameters','iteration':it,'values':{'width':w}})+'\\n').encode())\n"
                "    _ = json.loads(r.readline().decode('utf-8'))\n"
                "    conn.sendall((json.dumps({'type':'coverage','value':float(it)*0.25})+'\\n').encode())\n"
                "conn.sendall(b'{\"type\":\"optimization_finished\"}\\n')\n"
                "conn.close()\n",
                encoding="utf-8",
            )
            result = client.run_subprocess_study(
                study,
                lambda p, _it: {"score": p["width"] * 5.0},
                backend_dir=tmp_path,
                log_dir=tmp_path / "logs",
            )
            self.assertTrue(result.finished)
            self.assertEqual([t.iteration for t in result.trials], [1, 2])
            self.assertAlmostEqual(result.trials[1].objectives["score"], 0.4)


if __name__ == "__main__":
    unittest.main()
