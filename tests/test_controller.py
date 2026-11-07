import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

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
        self.acquire_sleep_inhibitor = m.acquire_sleep_inhibitor
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.patches = [patch.object(m, "ROOT", self.root), patch.object(m, "STATE", self.root / "state"),
                        patch.object(m, "PORT", port), patch.object(m, "recipe", return_value=self.r),
                         patch.object(m, "validate", side_effect=lambda r: (copy.deepcopy(self.cfg), sys.executable, dict(os.environ))),
                        patch.object(m, "gpu_guard"),
                        patch.object(m, "acquire_sleep_inhibitor", side_effect=lambda name: os.open(os.devnull, os.O_RDONLY))]
        for p in self.patches:
            p.start()
        self.addCleanup(self.cleanup_service)

    def cleanup_service(self):
        try:
            m.stop(10, force=True)
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
        self.assertTrue(state["sleep_inhibitor"]["held"])
        self.assertEqual(Path(state["config"]).stat().st_mode & 0o777, 0o600)
        m.smoke()
        m.stop(10)
        self.assertFalse(m.owned_live(state))
        self.assertEqual(m.status()["status"], "stopped")
        self.assertFalse(m.status()["sleep_inhibitor"]["held"])

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

    def test_busy_reusable_port_still_refuses_without_spawn(self):
        with socket.socket() as sock, patch.object(m.subprocess, "Popen") as popen:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", m.PORT))
            sock.listen()
            with self.assertRaises(OSError):
                m.start(self.r["id"], 10)
        popen.assert_not_called()

    def test_start_reuses_time_wait_without_waiting(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", m.PORT))
            listener.listen()
            client.settimeout(5)
            client.connect(("127.0.0.1", m.PORT))
            connection, _ = listener.accept()
            connection.close()  # The server closes first, leaving its port in TIME_WAIT.
            self.assertEqual(client.recv(1), b"")
        with socket.socket() as exclusive:
            with self.assertRaises(OSError):
                exclusive.bind(("127.0.0.1", m.PORT))
        m.start(self.r["id"], 10)
        self.assertEqual(m.status()["status"], "ready")

    def test_failed_start_preserves_log_and_releases_lock(self):
        self.cfg["test_fail"] = True
        with self.assertRaisesRegex(ValueError, "before readiness"):
            m.start(self.r["id"], 10)
        state = m.current()
        self.assertEqual(state["status"], "failed")
        self.assertTrue(Path(state["log"]).is_file())
        self.assertTrue(m.wait_sleep_inhibitor(state, 5))
        self.assertFalse(m.sleep_inhibitor(state)["held"])
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

    def test_stop_stops_native_engine_child(self):
        self.cfg["test_spawn_engine"] = True
        m.start(self.r["id"], 10)
        state = m.current()
        engines = m.native_processes(state)
        self.assertEqual(len(engines), 1)
        self.assertEqual(state["native_processes"], engines)
        m.stop(10)
        self.assertEqual(m.native_processes(state), [])
        self.assertEqual(m.status()["status"], "stopped")

    def test_stop_recovers_reparented_engine_and_blocks_start(self):
        self.cfg["test_spawn_engine"] = True
        m.start(self.r["id"], 10)
        state = m.current()
        parent = m._CHILDREN[state["pid"]]
        parent.terminate()
        parent.wait(timeout=5)
        self.assertFalse(m.owned_live(state))
        self.assertEqual(m.status()["status"], "orphaned")
        self.assertTrue(m.status()["sleep_inhibitor"]["held"])
        # The guardian holds protection and asset locks after the wrapper is gone.
        with (self.r["project"] / "logs/assets.lock").open("a+") as stream:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaisesRegex(ValueError, "children are still running"):
            m.start(self.r["id"], 10)
        m.stop(10)
        self.assertEqual(m.native_processes(state), [])
        self.assertFalse(m.sleep_inhibitor(state)["held"])

    def test_stop_timeout_retains_child_and_force_stops_it(self):
        self.cfg.update(test_spawn_engine=True, test_stubborn_engine=True)
        m.start(self.r["id"], 10)
        state = m.current()
        with self.assertRaisesRegex(ValueError, "no SIGKILL sent"):
            m.stop(0.05)
        self.assertEqual(m.status()["status"], "stopping")
        self.assertTrue(m.status()["sleep_inhibitor"]["held"])
        self.assertEqual(len(m.native_processes(state)), 1)
        self.assertTrue(m.current()["remaining_pids"])
        m.stop(10, force=True)
        self.assertEqual(m.native_processes(state), [])
        self.assertEqual(m.status()["status"], "stopped")
        self.assertFalse(m.sleep_inhibitor(state)["held"])

    def test_inhibitor_failure_refuses_before_spawning_model(self):
        with patch.object(m, "acquire_sleep_inhibitor", side_effect=ValueError("inhibitor denied")), \
                patch.object(m.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "inhibitor denied"):
                m.start(self.r["id"], 10)
        popen.assert_not_called()
        self.assertIsNone(m.current())
        with (m.STATE / "service.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_loading_service_is_already_sleep_protected(self):
        def loading(state, key):
            self.assertTrue(m.sleep_inhibitor(state)["held"])
            raise ValueError("not ready")
        # Poll readiness only after the actual CPU fixture answers; no model is loaded.
        health = m.health_check
        def not_ready(state, key):
            health(state, key)
            loading(state, key)
        with patch.object(m, "health_check", side_effect=not_ready):
            with self.assertRaisesRegex(ValueError, "left running"):
                m.start(self.r["id"], 0.6)
        self.assertTrue(m.sleep_inhibitor(m.current())["held"])

    def test_guard_survives_forced_wrapper_exit(self):
        self.cfg["test_spawn_engine"] = True
        m.start(self.r["id"], 10)
        state = m.current()
        parent = m._CHILDREN[state["pid"]]
        parent.kill()
        parent.wait(timeout=5)
        self.assertTrue(m.sleep_inhibitor(state)["held"])
        self.assertEqual(len(m.native_processes(state)), 1)
        m.stop(10, force=True)
        self.assertFalse(m.sleep_inhibitor(state)["held"])

    def test_guard_failure_prevents_engine_launch(self):
        with patch.object(m, "SLEEP_HELPER", self.root / "missing-helper.py"):
            with self.assertRaisesRegex(ValueError, "before readiness"):
                m.start(self.r["id"], 10)
        self.assertEqual(m.native_processes(m.current()), [])

    def test_sleep_receipt_cannot_claim_unrelated_pid(self):
        self.cfg["test_spawn_engine"] = True
        m.start(self.r["id"], 10)
        state = m.current()
        receipt = Path(state["config"]).with_name("sleep-inhibitor.json")
        original = m.read_json(receipt)
        try:
            m.write_json(receipt, {"pid": os.getpid(), "start_ticks": m.start_ticks(os.getpid()), "fd": 0})
            self.assertFalse(m.sleep_inhibitor(state)["held"])
        finally:
            m.write_json(receipt, original)

    def test_logind_inhibits_sleep_only_without_authentication_prompt(self):
        # Restore the real acquisition function, but keep D-Bus entirely mocked.
        acquire = self.acquire_sleep_inhibitor
        bus, manager = MagicMock(), MagicMock()
        fd = os.open(os.devnull, os.O_RDONLY)
        manager.Inhibit.return_value.take.return_value = fd
        manager.Get.return_value = False
        fake = types.SimpleNamespace(SystemBus=MagicMock(return_value=bus),
                                     Interface=MagicMock(return_value=manager), DBusException=RuntimeError)
        try:
            with patch.dict(sys.modules, {"dbus": fake}):
                self.assertEqual(acquire(self.r["id"]), fd)
            manager.Inhibit.assert_called_once_with("sleep", "local-llm",
                "maya/test-model is loaded; stop it before sleeping", "block", timeout=10)
            fake.SystemBus.assert_called_once_with(private=True)
            bus.close.assert_called_once()
        finally:
            os.close(fd)

    def test_start_during_sleep_transition_releases_inhibitor_and_refuses(self):
        acquire = self.acquire_sleep_inhibitor
        bus, manager = MagicMock(), MagicMock()
        fd = os.open(os.devnull, os.O_RDONLY)
        manager.Inhibit.return_value.take.return_value = fd
        manager.Get.return_value = True
        fake = types.SimpleNamespace(SystemBus=MagicMock(return_value=bus),
                                     Interface=MagicMock(return_value=manager), DBusException=RuntimeError)
        with patch.dict(sys.modules, {"dbus": fake}):
            with self.assertRaisesRegex(ValueError, "already preparing for sleep"):
                acquire(self.r["id"])
        with self.assertRaises(OSError):
            os.fstat(fd)
        bus.close.assert_called_once()

    def test_missing_guardian_is_reported_and_same_start_refuses(self):
        m.start(self.r["id"], 10)
        with patch.object(m, "sleep_inhibitor", return_value={"held": False}):
            self.assertEqual(m.status()["status"], "unprotected")
            with self.assertRaisesRegex(ValueError, "no sleep inhibitor"):
                m.start(self.r["id"], 10)

    def test_matching_engine_outside_owned_session_is_untouched(self):
        other = subprocess.Popen([sys.executable, str(self.source / "serve/native.py"),
                                  "--serve", *self.cfg["args"]], cwd=self.source, start_new_session=True)
        self.addCleanup(lambda: other.poll() is None and other.kill())
        try:
            self.cfg["test_spawn_engine"] = True
            m.start(self.r["id"], 10)
            m.stop(10, force=True)
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=5)

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
