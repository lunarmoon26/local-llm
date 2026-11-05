"""Read-only integrity checks for installed Maya assets; no setup or serving implementation."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def flag(args, name):
    return args[args.index(name) + 1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def profile_digest(cfg):
    # Identity, paths, engine flags, GPU selection and environment are reviewed together.
    profile = {k: v for k, v in cfg.items() if k not in ("_generated_by", "_deployment", "log", "api_key")}
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(r, cfg):
    if profile_digest(cfg) != r["config_sha256"]:
        raise ValueError("Installed Maya profile changed; review settings before updating its recipe fingerprint")
    args = cfg["args"]
    pack = Path(flag(args, "--glm-pack" if r["kind"] == "maya-glm" else "--pack"))
    if Path(cfg["tokenizer"]) != pack / "tokenizer":
        raise ValueError("Tokenizer does not belong to the selected pack")
    names = ["tokenizer/" + n for n in ("vocab.json", "merges.txt", "tokenizer.json",
                                       "token_type.json", "chat_template.jinja")]
    if r["kind"] == "maya-glm":
        names += ["index.txt", "dense.bin", "native_experts.txt", "compat-bf16.json"]
    else:
        names += ["ds4.txt", "tensors.txt", "engram.bin"]
    for name in names:
        if not (pack / name).is_file():
            raise ValueError(f"Maya pack is missing {name}")
    if r["kind"] == "maya-glm":
        check_glm(r, cfg, pack, names)
    else:
        check_deepseek(r, pack)


def check_glm(r, cfg, pack, names):
    manifest = read(r["project"] / "deployment/model.json")
    model = pack.parent
    verified = read(model / "verified.json")
    if verified.get("model") != manifest:
        raise ValueError("Model verification differs from the pinned manifest")
    for entry in manifest["files"]:
        path = model / entry["name"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"] or
                verified["files"].get(path.name) != signature(path)):
            raise ValueError(f"Missing or changed verified model shard: {path.name}")
    receipt = read(pack / "deployment.json")
    if (receipt.get("source_revision") != r["source_revision"] or receipt.get("model") != verified or
            set(receipt.get("files", {})) != set(names)):
        raise ValueError("Pack does not match the verified model and source")
    for name in names:
        path = pack / name
        if path.is_symlink() or receipt["files"][name]["stat"] != signature(path):
            raise ValueError(f"Pack changed: {name}")
    first = model / manifest["files"][0]["name"]
    if Path(read(pack / "compat-bf16.json")["source"]).resolve() != first.resolve():
        raise ValueError("Pack indexes a different model")
    if (pack / "experts.bin").exists():
        raise ValueError("Unexpected duplicated experts.bin")
    build = read(Path(cfg["exe"]).parent / "deployment-build.json")
    if build["source"]["revision"] != r["source_revision"] or build["engine_sha256"] != digest(cfg["exe"]):
        raise ValueError("Engine differs from its pinned build receipt")


def check_deepseek(r, pack):
    fields = dict(line.split(None, 1) for line in (pack / "ds4.txt").read_text().splitlines() if line.strip())
    model = Path(fields["model_dir"]).resolve()
    relative = Path(r["model_directory"])
    if tuple(model.parts[-len(relative.parts):]) != relative.parts:
        raise ValueError("Pack does not point to the pinned DeepSeek model")
    root = model.parents[len(relative.parts) - 1]
    engram = {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}
    indexed = set(read(model / "model.safetensors.index.json")["weight_map"].values())
    packed = set()
    with (pack / "tensors.txt").open() as stream:
        for line in stream:
            if not line.startswith("file "):
                break
            _, index, name = line.rstrip("\n").split(maxsplit=2)
            if index != str(len(packed)) or name in packed or Path(name).name != name:
                raise ValueError("Invalid pack shard entry")
            packed.add(name)
    if not indexed or packed != indexed | engram:
        raise ValueError("Pack shard list differs from the model index")
    for name in packed:
        path = model / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing model shard or target outside model storage: {name}")
        if name in engram:
            target = root / "DeepSeek-V4.1-Flash-1017-Engram" / name
            if not path.is_symlink() or path.resolve() != target or target.is_symlink():
                raise ValueError("Engram must link to its shared real file")
