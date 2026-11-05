"""Tiny on-disk fixtures for read-only Maya integrity checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from deployment import maya_assets as m


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.r = {"project": self.root, "kind": "maya-glm", "source_revision": "source-pin"}
        self.model = self.root / "models/glm"
        self.pack = self.model / "pack"
        self.names = ["index.txt", "dense.bin", "native_experts.txt", "compat-bf16.json"] + [
            "tokenizer/" + n for n in ("vocab.json", "merges.txt", "tokenizer.json", "token_type.json", "chat_template.jinja")]
        for name in self.names:
            path = self.pack / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        self.shard = self.model / "model.gguf"
        self.shard.write_bytes(b"tiny weights")
        self.write(self.pack / "compat-bf16.json", {"source": str(self.shard)})
        manifest = {"revision": "model-pin", "files": [{"name": self.shard.name,
                    "size": self.shard.stat().st_size, "sha256": m.digest(self.shard)}]}
        self.write(self.root / "deployment/model.json", manifest)
        verified = {"model": manifest, "files": {self.shard.name: m.signature(self.shard)}}
        self.write(self.model / "verified.json", verified)
        self.write(self.pack / "deployment.json", {"source_revision": "source-pin", "model": verified,
                   "files": {n: {"stat": m.signature(self.pack / n)} for n in self.names}})
        self.exe = self.root / "build/maya"
        self.exe.parent.mkdir()
        self.exe.write_bytes(b"tiny binary")
        self.write(self.exe.parent / "deployment-build.json", {"source": {"revision": "source-pin"},
                   "engine_sha256": m.digest(self.exe)})
        self.cfg = {"exe": str(self.exe), "args": ["--glm-pack", str(self.pack)],
                    "tokenizer": str(self.pack / "tokenizer"), "model_name": "glm"}
        self.r["config_sha256"] = m.profile_digest(self.cfg)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_valid_assets_are_read_only(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        m.validate(self.r, self.cfg)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_changed_settings_fail_closed(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["args"] += ["--tp", "2"]
        with self.assertRaisesRegex(ValueError, "profile changed"):
            m.validate(self.r, cfg)

    def test_non_serving_metadata_and_private_key_do_not_affect_fingerprint(self):
        cfg = {**self.cfg, "_generated_by": "ignored", "_deployment": {}, "log": "private", "api_key": ""}
        self.assertEqual(m.profile_digest(cfg), self.r["config_sha256"])

    def test_changed_model_pack_and_binary_are_detected(self):
        for path, error in ((self.shard, "model shard"), (self.pack / "dense.bin", "Pack changed"),
                            (self.exe, "Engine differs")):
            body = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, error):
                m.validate(self.r, self.cfg)
            path.write_bytes(body)
            # Refresh fixture stat receipts after each deliberate mutation.
            verified = m.read(self.model / "verified.json")
            verified["files"][self.shard.name] = m.signature(self.shard)
            self.write(self.model / "verified.json", verified)
            receipt = m.read(self.pack / "deployment.json")
            receipt["model"] = verified
            for name in self.names:
                receipt["files"][name]["stat"] = m.signature(self.pack / name)
            self.write(self.pack / "deployment.json", receipt)

    def deepseek(self):
        model = self.root / "models/deepseek"
        model.mkdir(parents=True)
        r = {"model_directory": "deepseek"}
        engram = self.root / "models/DeepSeek-V4.1-Flash-1017-Engram"
        engram.mkdir()
        names = ["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"]
        for name in names:
            (engram / name).write_bytes(b"table")
            (model / name).symlink_to(engram / name)
        (model / "weights.safetensors").write_bytes(b"weights")
        self.write(model / "model.safetensors.index.json", {"weight_map": {"w": "weights.safetensors"}})
        (self.pack / "ds4.txt").write_text(f"model_dir {model}\n")
        names.insert(0, "weights.safetensors")
        (self.pack / "tensors.txt").write_text("".join(f"file {i} {n}\n" for i, n in enumerate(names)) + "t value\n")
        return r, model, engram

    def test_deepseek_index_and_shared_real_engram(self):
        r, model, engram = self.deepseek()
        m.check_deepseek(r, self.pack)
        target = engram / "model-00047-of-00048.safetensors"
        target.unlink()
        target.symlink_to(model / "weights.safetensors")
        with self.assertRaisesRegex(ValueError, "shared real file"):
            m.check_deepseek(r, self.pack)

    def test_deepseek_rejects_bad_index_or_missing_shard(self):
        r, model, _ = self.deepseek()
        (model / "weights.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing model shard"):
            m.check_deepseek(r, self.pack)
        (self.pack / "tensors.txt").write_text("file 1 weights.safetensors\n")
        with self.assertRaisesRegex(ValueError, "Invalid pack shard"):
            m.check_deepseek(r, self.pack)
