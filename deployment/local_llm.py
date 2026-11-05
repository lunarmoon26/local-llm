"""One explicitly selected native LLM service; no installs, downloads or automatic switches."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
OWNER = "local-llm/1"
HOST, PORT = "127.0.0.1", 8080
KINDS = {"maya-glm", "maya-deepseek", "strata-qwen"}
_CHILDREN = {}  # Retain/reap children when lifecycle functions run in one Python process (tests/embedding).


class BusyError(ValueError):
    """A conflicting lock refuses work without changing another process."""


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked state: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def lock_file(name: str):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / name
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked lock: {path}")
    return path.open("a+")


@contextmanager
def control_lock():
    with lock_file("control.lock") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def selector(parts) -> str:
    name = "/".join(parts)
    if not re.fullmatch(r"(?:maya|strata)/[a-z0-9][a-z0-9.-]*", name):
        raise ValueError("Specify <engine>/<model>, e.g. maya/glm-5.3-flash")
    return name


def recipe(name: str) -> dict:
    selector([name])
    project = ROOT / name
    manifest = read_json(project / "recipe.json")
    if (manifest.get("schema") != "local-llm.recipe/1" or manifest.get("id") != name or
            manifest.get("kind") not in KINDS or manifest["kind"].split("-")[0] != name.split("/")[0]):
        raise ValueError("Recipe identity or engine is invalid")
    for value in [manifest["config"], manifest["python"], *manifest.get("locks", [])]:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Recipe paths must be relative to their project")
    return {**manifest, "project": project.resolve()}


def flag(args, name, default=None):
    return args[args.index(name) + 1] if name in args[:-1] else default


def recipe_env(r: dict) -> dict:
    """Load private recipe settings; explicit environment values take precedence."""
    env = dict(os.environ)
    script = r["project"] / ".env"
    if script.is_file():
        output = subprocess.check_output([
            "bash", "-c", 'set -a; source "$1"; exec "$2" -c '
            "'import os,json; print(json.dumps(dict(os.environ)))'", "bash", str(script), sys.executable],
            text=True, timeout=30)
        env = {**json.loads(output), **env}
    return env


def validate(r: dict) -> tuple[dict, str, dict]:
    p = r["project"] / r["config"]
    if p.is_symlink():
        raise ValueError("Refusing a symlinked installed configuration")
    cfg = read_json(p)
    if cfg["model_name"] != r["api_model"] or cfg.get("host", HOST) != HOST:
        raise ValueError("Installed model identity/listener does not match recipe")
    source = Path(cfg["cwd"]).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != r["source_revision"]:
        raise ValueError("Engine checkout differs from recipe pin")
    dirty = subprocess.check_output([
        "git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--",
        "serve", "tools/maya_names.py", "tools/maya_quants.py", "tools/strata_tokenizer.py"], text=True)
    if dirty:
        raise ValueError("Serving sources have changes; review before loading a model")
    exe = Path(cfg["exe"])
    if not exe.is_absolute() or not exe.is_file() or not os.access(exe, os.X_OK):
        raise ValueError("Configured engine is missing or not executable")
    env = recipe_env(r)
    python = next((env[k] for k in r.get("python_env", []) if env.get(k)), str(r["project"] / r["python"]))
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError("Recipe Python environment is missing")
    if r["kind"].startswith("maya-"):
        try:
            from . import maya_assets
        except ImportError:  # Executed by the shell entrypoint rather than imported as a package.
            import maya_assets
        maya_assets.validate(r, cfg)
    else:
        args = cfg["args"]
        for key in ("--pack", "--native", "--mtp", "--expert-profile"):
            value = flag(args, key)
            if not value or not Path(value).exists():
                raise ValueError(f"Qwen asset is missing: {key}")
        pack = Path(flag(args, "--pack"))
        for name in ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json"):
            if not (pack / name).is_file():
                raise ValueError(f"Qwen pack is missing {name}")
        native = Path(flag(args, "--native"))
        for i in range(1, 4):
            if not native.with_name(native.name.replace("00001-of-00003", f"{i:05d}-of-00003")).is_file():
                raise ValueError("Qwen GGUF shard is missing")
        if (flag(args, "--max-context") != "524288" or flag(args, "--rope-scaling") != "yarn" or
                flag(args, "--rope-scale") != "2" or flag(args, "--resident-budget-gib") != "12"):
            raise ValueError("Qwen differs from its installed bounded-RAM/YaRN profile")
    return cfg, python, env


def gpu_devices(cfg):
    if "gpu" in cfg:
        return cfg["gpu"]
    selected = cfg["_deployment"]["cuda_visible_devices"].split(",")
    if all(i.isdigit() for i in selected):
        return list(map(int, selected))
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                     text=True, timeout=15)
    indices = {uuid.strip(): int(index) for index, uuid in (line.split(",") for line in output.splitlines())}
    return [indices[i] for i in selected]


def render(r, cfg, env, run: str):
    cfg = json.loads(json.dumps(cfg))
    key = os.environ.get("LOCAL_LLM_API_KEY") or env.get("STRATA_API_KEY") or cfg.pop("api_key", "")
    env = {k: v for k, v in env.items() if not k.startswith(("MAYA_", "STRATA_", "DS41_"))}
    cfg.pop("api_key", None)
    if key:
        env["STRATA_API_KEY"] = key
    cfg.update(host=HOST, port=PORT, open_browser=False, model_management=False,
               log=str(ROOT / "logs" / r["id"] / f"{run}.engine.log"))
    if "gpu" not in cfg:
        cfg["gpu"] = gpu_devices(cfg)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               MAYA_MODEL_MANAGEMENT="0", STRATA_MODEL_MANAGEMENT="0")
    if r["kind"] == "maya-deepseek":
        for prefix in ("MAYA_DS4_", "STRATA_DS4_"):
            for suffix in ("DISK_SKIP_W", "DISK_SKIP_AUTO", "RAM_SKIP_W", "SLOT_KEEP", "RAM_TIER"):
                env[prefix + suffix] = "0"
        env["DS4_YOCO"] = "1"
    return cfg, env


def gpu_guard(r, cfg):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    free = {int(a): int(b) for a, b in (line.split(",") for line in output.splitlines())}
    gpus = cfg["gpu"]
    if not isinstance(gpus, list) or len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Installed recipe must select two distinct GPUs")
    for gpu in gpus:
        if free[gpu] < r["min_free_mib"]:
            raise ValueError(f"GPU {gpu} has {free[gpu]} MiB free; recipe requires {r['min_free_mib']}. "
                             "Stop the occupying service explicitly; no other process was stopped.")


def start_ticks(pid):
    text = Path(f"/proc/{int(pid)}/stat").read_text()
    tail = text[text.rindex(")") + 2:].split()
    return None if tail[0] == "Z" else tail[19]


def current():
    path = STATE / "service.json"
    if path.is_symlink():
        raise ValueError("Refusing a symlinked process receipt")
    if not path.exists():
        return None
    state = read_json(path)
    if state.get("owner") != OWNER:
        raise ValueError("Unowned process receipt; refusing to replace or signal it")
    return state


def owned_live(state):
    if not state:
        return False
    try:
        if start_ticks(state["pid"]) != state["start_ticks"]:
            return False
        args = Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b"\0")
        if os.fsencode(state["config"]) not in args:
            raise ValueError("PID matches but its command does not: refusing ownership")
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def request(path, key="", body=None, timeout=3):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}",
                                 None if body is None else json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def health_check(state, key=""):
    h = request("/health", key)
    if (h.get("status") != "ok" or h.get("unloaded") or h.get("loaded") is False or
            h.get("max_context") != state["context"]):
        raise ValueError("Listener is not loaded with this recipe's context")
    if not any(m["id"] == state["model"] for m in request("/v1/models", key)["data"]):
        raise ValueError("Listener advertises a different model; it was not stopped")
    return h


def update_state(state, **change):
    with control_lock():
        old = current()
        if old and old["run"] == state["run"]:
            write_json(STATE / "service.json", {**old, **change})


def start(name, timeout=600):
    r = recipe(name)
    with control_lock():
        old = current()
        if owned_live(old):
            if old["recipe"] != name:
                raise ValueError(f"{old['recipe']} is already {old['status']}; run bash stop.sh before switching")
            print(json.dumps({"status": old["status"], "recipe": name, "already_running": True}))
            return
        lock = lock_file("service.lock")
        recipe_locks = []
        child, registered = None, False
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BusyError("Service lock is held; no process was started or stopped")
            with socket.socket() as sock:
                sock.bind((HOST, PORT))
            cfg, python, original_env = validate(r)
            for relative in r.get("locks", []):
                path = r["project"] / relative
                if path.is_symlink():
                    raise ValueError("Refusing symlinked recipe lock")
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a+")
                recipe_locks.append(stream)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise BusyError("Recipe assets/build are busy; no model was started")
            run = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            cfg, env = render(r, cfg, original_env, run)
            gpu_guard(r, cfg)
            config = STATE / "runs" / run / "server.json"
            write_json(config, cfg)
            log = ROOT / "logs" / name / f"{run}.server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
            source = Path(cfg["cwd"]).resolve()
            command = ([python, "-u", "-m", "serve.server"] if r["kind"].startswith("maya-") else
                       [python, "-u", str(source / "serve/server.py")])
            command += ["--engine", "maya" if r["kind"].startswith("maya-") else "strata",
                        "--config", str(config), "--host", HOST, "--port", str(PORT)]
            with log.open("xb") as output:
                child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock.fileno(), *(s.fileno() for s in recipe_locks)))
                _CHILDREN[child.pid] = child
            ticks = start_ticks(child.pid)
            if ticks is None:
                raise ValueError(f"Server exited during startup; see {log}")
            state = {"owner": OWNER, "run": run, "recipe": name, "pid": child.pid, "start_ticks": ticks,
                     "config": str(config), "log": str(log), "model": cfg["model_name"],
                     "context": int(flag(cfg["args"], "--max-context")), "status": "starting",
                     "host": HOST, "port": PORT, "started_at": time.time()}
            write_json(STATE / "service.json", state)
            registered = True
            key = env.get("STRATA_API_KEY", "")
        except BaseException:
            if child is not None and not registered:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=120)
                _CHILDREN.pop(child.pid, None)
            raise
        finally:
            # No LOCK_UN: the inherited open-file description keeps this locked in the server.
            lock.close()
            for stream in recipe_locks:
                stream.close()
    print(f"Starting {name}; PID {state['pid']}; log {log}", flush=True)
    deadline = time.monotonic() + timeout
    while owned_live(state):
        try:
            health_check(state, key)
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise ValueError(f"Still starting after {timeout}s; left running. See bash status.sh and {log}")
            time.sleep(0.5)
        else:
            update_state(state, status="ready")
            print(json.dumps({"status": "ready", "recipe": name, "context": state["context"],
                              "url": f"http://{HOST}:{PORT}/v1", "pid": state["pid"]}))
            return
    update_state(state, status="failed" if child.poll() not in (0, None) else "stopped", exit_code=child.poll())
    child.wait()
    _CHILDREN.pop(child.pid, None)
    raise ValueError(f"Server ended before readiness; see {log}")


def stop(timeout=120):
    with control_lock():
        state = current()
        if not owned_live(state):
            print("No owned live service to stop.")
            return
        fd = os.pidfd_open(state["pid"])
        try:
            # Recheck after opening the pidfd so PID reuse cannot redirect the signal.
            if not owned_live(state):
                raise ValueError("Process identity changed; no signal was sent")
            write_json(STATE / "service.json", {**state, "status": "stopping"})
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            if not select.select([fd], [], [], timeout)[0]:
                raise ValueError("Graceful stop timed out; process left stopping, no SIGKILL sent")
        finally:
            os.close(fd)
        child = _CHILDREN.pop(state["pid"], None)
        if child is not None:
            child.wait(timeout=5)
        write_json(STATE / "service.json", {**state, "status": "stopped", "stopped_at": time.time()})
        print(f"Stopped {state['recipe']} gracefully.")


def status():
    state = current()
    if not state:
        return {"status": "stopped", "recipe": None}
    out = dict(state)
    if not owned_live(state):
        out["status"] = "stopped" if state["status"] == "stopped" else "failed"
    else:
        try:
            r = recipe(state["recipe"])
            key = os.environ.get("LOCAL_LLM_API_KEY") or recipe_env(r).get("STRATA_API_KEY", "")
            health_check(state, key)
            out["status"] = "ready"
        except (OSError, ValueError) as error:
            out["health_error"] = str(error)
            if out["status"] == "ready":
                out["status"] = "unhealthy"
    return out


def smoke(name=None):
    state = current()
    if not owned_live(state) or (name and name != state["recipe"]):
        raise ValueError("Requested recipe is not the managed live service")
    r = recipe(state["recipe"])
    key = os.environ.get("LOCAL_LLM_API_KEY") or recipe_env(r).get("STRATA_API_KEY", "")
    health_check(state, key)
    result = request("/v1/chat/completions", key, {
        "model": state["model"], "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "temperature": 0, "max_tokens": 128, "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
    if result["choices"][0]["message"]["content"].strip() != "323":
        raise ValueError("Arithmetic smoke check failed")
    print(json.dumps({"status": "pass", "recipe": state["recipe"], "context": state["context"], "answer": "323"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "check", "start", "stop", "status", "smoke"))
    parser.add_argument("recipe", nargs="*")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--json", action="store_true", help="status is always JSON")
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error("timeout must be positive")
    name = selector(args.recipe) if args.recipe else None
    if args.action in ("check", "start") and not name:
        parser.error("a recipe is required")
    if args.action in ("list", "stop", "status") and name:
        parser.error("this action does not take a recipe")
    if args.action == "list":
        for path in sorted([*ROOT.glob("maya/*/recipe.json"), *ROOT.glob("strata/*/recipe.json")]):
            r = recipe(str(path.parent.relative_to(ROOT)))
            print(f"{r['id']}\t{r['api_model']}")
    elif args.action == "check":
        cfg, _, _ = validate(recipe(name))
        print(f"Checked {name}: {cfg['model_name']}, context {flag(cfg['args'], '--max-context')}; no inference/download.")
    elif args.action == "start":
        start(name, args.timeout or 600)
    elif args.action == "stop":
        stop(args.timeout or 120)
    elif args.action == "status":
        print(json.dumps(status(), indent=2))
    else:
        smoke(name)


if __name__ == "__main__":
    try:
        main()
    except BusyError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(73)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
