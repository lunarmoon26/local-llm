import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from deployment import local_llm as m


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        shutil.copytree(Path(__file__).parent / "fixtures/serve", self.source / "serve")
        self.r = {"id": "maya/test-model", "kind": "maya-glm", "project": self.root / "maya/test-model",
                  "config": "installed.json", "python": "python", "api_model": "test-model", "min_free_mib": 1,
                  "locks": ["logs/assets.lock"]}
        self.r["project"].mkdir(parents=True)
        self.cfg = {"cwd": str(self.source), "exe": sys.executable, "model_name": "test-model",
                    "gpu": [0, 1], "args": ["--max-context", "524288"]}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.patches = [patch.object(m, "ROOT", self.root), patch.object(m, "STATE", self.root / "state"),
                        patch.object(m, "PORT", port), patch.object(m, "recipe", return_value=self.r),
                        patch.object(m, "validate", side_effect=lambda r: (copy.deepcopy(self.cfg), sys.executable, dict(os.environ))),
                        patch.object(m, "gpu_guard")]
        for p in self.patches:
            p.start()
        self.addCleanup(self.cleanup_service)

    def cleanup_service(self):
        try:
            m.stop(10)
            state = m.current()
            if state:
                try:
                    os.waitpid(state["pid"], os.WNOHANG)
                except ChildProcessError:
                    pass
        finally:
            for p in reversed(self.patches):
                p.stop()

    def test_selector_refuses_paths_and_accepts_both_forms(self):
        self.assertEqual(m.selector(["maya", "test-model"]), "maya/test-model")
        self.assertEqual(m.selector(["maya/test-model"]), "maya/test-model")
        for args in (["../maya/test-model"], ["maya/../x"], ["unknown/model"], []):
            with self.assertRaises(ValueError):
                m.selector(args)

    def test_render_preserves_engine_flags_but_uses_common_listener(self):
        before = copy.deepcopy(self.cfg)
        cfg, env = m.render(self.r, self.cfg, {"MAYA_MODEL_MANAGEMENT": "1", "STRATA_GLM_MISS_SKIP_W": "0.9",
                                             "STRATA_API_KEY": "environment-secret", "PATH": "/bin"}, "run")
        self.assertEqual(self.cfg, before)
        self.assertEqual(cfg["args"], before["args"])
        self.assertEqual((cfg["host"], cfg["port"]), ("127.0.0.1", m.PORT))
        self.assertFalse(cfg["model_management"])
        self.assertNotIn("api_key", cfg)
        self.assertNotIn("environment-secret", json.dumps(cfg))
        self.assertEqual(env["STRATA_API_KEY"], "environment-secret")
        self.assertNotIn("STRATA_GLM_MISS_SKIP_W", env)
        self.assertEqual(env["MAYA_MODEL_MANAGEMENT"], "0")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_deepseek_skip_guards_and_gpu_order_are_preserved(self):
        r = {**self.r, "kind": "maya-deepseek"}
        cfg = {**self.cfg, "_deployment": {"cuda_visible_devices": "1,0"}}
        del cfg["gpu"]
        cfg, env = m.render(r, cfg, {}, "run")
        self.assertEqual(cfg["gpu"], [1, 0])
        self.assertEqual(env["MAYA_DS4_DISK_SKIP_W"], "0")
        self.assertEqual(env["STRATA_DS4_RAM_SKIP_W"], "0")

    def test_start_health_smoke_and_stop_without_gpu(self):
        m.start(self.r["id"], 10)
        state = m.status()
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["context"], 524288)
        self.assertEqual(Path(state["config"]).stat().st_mode & 0o777, 0o600)
        m.smoke()
        m.stop(10)
        self.assertFalse(m.owned_live(state))
        self.assertEqual(m.status()["status"], "stopped")

    def test_service_and_recipe_locks_are_inherited_by_server(self):
        m.start(self.r["id"], 10)
        for path in (m.STATE / "service.lock", self.r["project"] / "logs/assets.lock"):
            with path.open("a+") as stream:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m.stop(10)
        with (m.STATE / "service.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_same_start_is_idempotent_and_different_recipe_refuses(self):
        m.start(self.r["id"], 10)
        pid = m.current()["pid"]
        m.start(self.r["id"], 10)
        self.assertEqual(m.current()["pid"], pid)
        with self.assertRaisesRegex(ValueError, "before switching"):
            m.start("maya/another-model", 10)
        self.assertTrue(m.owned_live(m.current()))

    def test_busy_asset_refuses_without_spawn(self):
        path = self.r["project"] / "logs/assets.lock"
        path.parent.mkdir()
        with path.open("a+") as stream, patch.object(m.subprocess, "Popen") as popen:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(m.BusyError):
                m.start(self.r["id"], 10)
        popen.assert_not_called()

    def test_busy_port_refuses_without_spawn(self):
        with socket.socket() as sock, patch.object(m.subprocess, "Popen") as popen:
            sock.bind(("127.0.0.1", m.PORT))
            sock.listen()
            with self.assertRaises(OSError):
                m.start(self.r["id"], 10)
        popen.assert_not_called()

    def test_failed_start_preserves_log_and_releases_lock(self):
        self.cfg["test_fail"] = True
        with self.assertRaisesRegex(ValueError, "before readiness"):
            m.start(self.r["id"], 10)
        state = m.current()
        self.assertEqual(state["status"], "failed")
        self.assertTrue(Path(state["log"]).is_file())
        with (m.STATE / "service.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_unowned_receipt_and_symlink_are_refused(self):
        m.write_json(m.STATE / "service.json", {"owner": "somebody-else"})
        with self.assertRaisesRegex(ValueError, "Unowned"):
            m.current()
        (m.STATE / "service.json").unlink()
        (m.STATE / "service.json").symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            m.write_json(m.STATE / "service.json", {})
        (m.STATE / "service.json").unlink()

    def test_stale_pid_receipt_never_signals_any_process(self):
        state = {"owner": m.OWNER, "pid": os.getpid(), "start_ticks": "wrong", "status": "ready"}
        m.write_json(m.STATE / "service.json", state)
        with patch.object(m.signal, "pidfd_send_signal") as send:
            m.stop()
        send.assert_not_called()

    def test_matching_pid_but_different_command_refuses_ownership(self):
        state = {"pid": os.getpid(), "start_ticks": m.start_ticks(os.getpid()), "config": "/not/my/config"}
        with self.assertRaisesRegex(ValueError, "refusing ownership"):
            m.owned_live(state)

    def test_stop_timeout_never_escalates_to_kill(self):
        m.start(self.r["id"], 10)
        with patch.object(m.signal, "pidfd_send_signal") as send, patch.object(m.select, "select", return_value=([], [], [])):
            with self.assertRaisesRegex(ValueError, "no SIGKILL"):
                m.stop(0.01)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.args[1], m.signal.SIGTERM)
        self.assertEqual(m.current()["status"], "stopping")

    def test_start_timeout_leaves_owned_service_for_explicit_stop(self):
        with patch.object(m, "health_check", side_effect=ValueError("not ready")):
            with self.assertRaisesRegex(ValueError, "left running"):
                m.start(self.r["id"], 0.01)
        self.assertTrue(m.owned_live(m.current()))

    def test_receipt_write_failure_does_not_orphan_started_server(self):
        write = m.write_json
        def fail_receipt(path, value):
            if path.name == "service.json":
                raise OSError("fixture receipt write failure")
            write(path, value)
        with patch.object(m, "write_json", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "receipt write failure"):
                m.start(self.r["id"], 10)
        self.assertFalse(m._CHILDREN)
        with (m.STATE / "service.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_all_shell_wrappers_parse(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("*.sh"):
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_private_recipe_env_preserves_explicit_overrides(self):
        (self.r["project"] / ".env").write_text("MAYA_PYTHON=/file/python\nSTRATA_API_KEY=file-key\n")
        with patch.dict(os.environ, {"MAYA_PYTHON": "/explicit/python"}, clear=True):
            env = m.recipe_env(self.r)
        self.assertEqual(env["MAYA_PYTHON"], "/explicit/python")
        self.assertEqual(env["STRATA_API_KEY"], "file-key")


if __name__ == "__main__":
    unittest.main()
