import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from deployment import local_llm as c, prepare_maya as m


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.r = {"project": self.root, "kind": "maya-glm", "config": "installed.json", "python": "python",
                  "source_revision": "pinned", "id": "maya/glm", "locks": ["assets.lock"]}
        self.cfg = {"cwd": str(self.root / "source with spaces"), "gpu": [1, 0],
                    "_deployment": {"data": str(self.root / "data")},
                    "args": ["--glm-pack", str(self.root / "model/pack"), "--max-context", "524288"]}
        (self.root / "installed.json").write_text(json.dumps(self.cfg))
        for p in (patch.object(c, "ROOT", self.root), patch.object(c, "STATE", self.root / "state"),
                  patch.object(c, "recipe", return_value=self.r),
                  patch.object(c, "recipe_env", return_value={"PATH": os.environ["PATH"]})):
            p.start()
            self.addCleanup(p.stop)

    def test_command_uses_upstream_without_start_download_or_calibration(self):
        cmd = m.command(self.r, self.cfg, sys.executable, {})
        for flag in ("--setup", "--no-start", "--yes", "--plain", "--no-vision"):
            self.assertIn(flag, cmd)
        for flag in ("--download-model", "--repack", "--rebuild", "--calibrate", "--speed-calibration"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(c.flag(cmd, "--drafting"), "on")
        self.assertEqual(c.flag(cmd, "--gguf-dir"), str(self.root / "model"))
        self.assertEqual(c.flag(cmd, "--context"), "524288")
        self.assertEqual(c.flag(cmd, "--gpus"), "1,0")

    def test_deepseek_command_keeps_local_model_path(self):
        r = {**self.r, "kind": "maya-deepseek", "model_directory": "repo"}
        cfg = {**self.cfg, "_deployment": {"data_dir": str(self.root / "data")}}
        cmd = m.command(r, cfg, sys.executable, {})
        self.assertEqual(c.flag(cmd, "--model-dir"), str(self.root / "data/models/repo"))
        self.assertEqual(c.flag(cmd, "--ds4-quant"), "exl3")

    def test_preview_does_not_execute_or_write(self):
        before = sorted(self.root.rglob("*"))
        with patch.object(m.subprocess, "run") as run:
            m.prepare(self.r["id"])
        run.assert_not_called()
        self.assertEqual(before, sorted(self.root.rglob("*")))

    def test_busy_service_refuses_execution(self):
        c.STATE.mkdir()
        with (c.STATE / "service.lock").open("a+") as lock, patch.object(m.subprocess, "run") as run:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(c.BusyError):
                m.prepare(self.r["id"], run=True)
        run.assert_not_called()

    def test_dirty_or_wrong_source_refuses_execution(self):
        for output in (["wrong"], ["pinned", " M maya.py"]):
            with patch.object(m.subprocess, "check_output", side_effect=output), patch.object(m.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    m.prepare(self.r["id"], run=True)
            run.assert_not_called()

    def test_execute_inherits_locks_and_sanitizes_environment(self):
        with patch.object(m.subprocess, "check_output", side_effect=["pinned", ""]), \
                patch.object(c, "recipe_env", return_value={"PATH": "/bin", "STRATA_API_KEY": "private", "MAYA_OTHER": "1"}), \
                patch.object(m.subprocess, "run") as run:
            m.prepare(self.r["id"], run=True)
        self.assertEqual(len(run.call_args.kwargs["pass_fds"]), 2)
        env = run.call_args.kwargs["env"]
        self.assertNotIn("STRATA_API_KEY", env)
        self.assertNotIn("MAYA_OTHER", env)
        self.assertEqual(env["MAYA_MODEL_MANAGEMENT"], "0")

    def test_uuid_gpu_selection_resolves_without_changing_order(self):
        cfg = {"_deployment": {"cuda_visible_devices": "GPU-b,GPU-a"}}
        with patch.object(c.subprocess, "check_output", return_value="0, GPU-a\n1, GPU-b\n"):
            self.assertEqual(c.gpu_devices(cfg), [1, 0])

    def test_recipe_wrappers_forward_from_paths_with_spaces(self):
        repo = Path(__file__).resolve().parents[1]
        for model in ("glm-5.3-flash", "deepseek-v4.1-flash"):
            root = self.root / "project with spaces" / "maya" / model
            root.mkdir(parents=True)
            common = root.parent.parent / "deployment"
            common.mkdir(exist_ok=True)
            (common / "prepare_maya.py").write_text("import json,sys; print(json.dumps(sys.argv[1:]))\n")
            wrapper = root / "install.sh"
            wrapper.write_bytes((repo / "maya" / model / "install.sh").read_bytes())
            result = subprocess.run(["bash", str(wrapper)], text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout), ["maya/" + model])
