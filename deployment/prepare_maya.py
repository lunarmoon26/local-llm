"""Preview, or explicitly run, the pinned upstream Maya installer. Never activate its output."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from . import local_llm as controller
except ImportError:
    import local_llm as controller


def command(r, cfg, python, env):
    """Translate the installed profile's asset locations, not reimplement Maya setup."""
    if not r["kind"].startswith("maya-"):
        raise ValueError("Only Maya recipes use this installer")
    source = Path(cfg["cwd"])
    args = cfg["args"]
    devices = controller.gpu_devices(cfg)
    cmd = [python, str(source / "maya.py"), "--setup", "--yes", "--plain", "--no-start",
           "--drafting", "on", "--no-vision", "--parallel", "1",
           "--host", controller.HOST, "--port", str(controller.PORT),
           "--context", controller.flag(args, "--max-context"),
           "--gpus", ",".join(map(str, devices)), "--config", str(r["project"] / r["config"])]
    if r["kind"] == "maya-glm":
        pack = Path(controller.flag(args, "--glm-pack"))
        model = pack.parent
        cmd += ["--gguf-dir", str(model), "--draft", "mtp"]
        data = cfg["_deployment"]["data"]
    else:
        data = cfg["_deployment"]["data_dir"]
        model = Path(data) / "models" / r["model_directory"]
        cmd += ["--model-dir", str(model), "--ds4-quant", "exl3", "--mmap-experts", "on"]
    cmd += ["--models-dir", str(Path(data) / "models"),
            "--llama-dir", env.get("GLM_GGML_DIR") or str(Path(data) / "sources/llama.cpp"),
            "--vram-reserve-mib", "4096"]
    return cmd


def prepare(name, run=False):
    r = controller.recipe(name)
    cfg = controller.read_json(r["project"] / r["config"])
    env = controller.recipe_env(r)
    python = next((env[k] for k in r.get("python_env", []) if env.get(k)), str(r["project"] / r["python"]))
    cmd = command(r, cfg, python, env)
    print(shlex.join(cmd), flush=True)
    print("Upstream may install dependencies, build and generate packs/configs. "
          "Existing serving config stays selected; generated output needs review.", flush=True)
    if not run:
        print("Preview only. Stop all models, then pass --run to execute upstream setup.")
        return
    with controller.control_lock(), ExitStack() as stack:
        # Use the same inherited locks as serving; refuse, never stop another process.
        paths = [controller.STATE / "service.lock", *(r["project"] / p for p in r.get("locks", []))]
        locks = []
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Refusing symlinked lock: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise controller.BusyError("A service or preparation holds a lock; stop it explicitly first")
            locks.append(stream)
        # Revalidate after locking, including the installer/build sources, not only serve/.
        revision = subprocess.check_output(["git", "-C", cfg["cwd"], "rev-parse", "HEAD"], text=True).strip()
        if revision != r["source_revision"]:
            raise ValueError("Maya checkout differs from the recipe pin")
        dirty = subprocess.check_output(["git", "-C", cfg["cwd"], "status", "--porcelain",
                                         "--untracked-files=all"], text=True)
        if dirty:
            raise ValueError("Maya checkout has local changes; review before running its installer")
        env = {k: v for k, v in env.items() if not k.startswith(("MAYA_", "STRATA_", "DS41_"))}
        env.update(MAYA_MODEL_MANAGEMENT="0", STRATA_MODEL_MANAGEMENT="0")
        env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
        subprocess.run(cmd, cwd=cfg["cwd"], env=env, check=True,
                       pass_fds=tuple(stream.fileno() for stream in locks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--run", action="store_true", help="allow upstream dependency/build/pack/config changes")
    args = parser.parse_args()
    prepare(args.recipe, args.run)


if __name__ == "__main__":
    try:
        main()
    except controller.BusyError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(73)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
