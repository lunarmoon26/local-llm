"""Keep a logind sleep inhibitor alive until the wrapper and native engines exit."""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from local_llm import native_processes, owned_live, start_ticks, write_json


def guard(fd, ready_fd, state):
    # A wrapper's termination or terminal closure must not drop orphan protection.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    os.fstat(fd)
    os.write(ready_fd, b"ready\n")
    os.close(ready_fd)
    while True:
        try:
            if not owned_live(state) and not native_processes(state):
                return
        except (OSError, ValueError, KeyError):
            # Failure to inspect ownership is not evidence that GPU cleanup ended.
            pass
        time.sleep(0.1)


def run(fd, locks, command):
    config = command[command.index("--config") + 1]
    state = {"pid": os.getpid(), "start_ticks": start_ticks(os.getpid()), "config": config}
    read_fd, ready_fd = os.pipe()
    try:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard", "--fd", str(fd),
             "--ready-fd", str(ready_fd), "--pid", str(state["pid"]),
             "--start-ticks", state["start_ticks"], "--config", config],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            start_new_session=True, pass_fds=(fd, ready_fd, *locks))
        os.close(ready_fd)
        ready_fd = None
        if not select.select([read_fd], [], [], 10)[0] or os.read(read_fd, 32) != b"ready\n":
            raise RuntimeError("Sleep guardian did not become ready; engine was not launched")
        ticks = start_ticks(child.pid)
        if ticks is None:
            raise RuntimeError("Sleep guardian exited; engine was not launched")
        write_json(Path(config).with_name("sleep-inhibitor.json"),
                   {"pid": child.pid, "start_ticks": ticks, "fd": fd})
    finally:
        os.close(read_fd)
        if ready_fd is not None:
            os.close(ready_fd)
    os.close(fd)  # Only the independent guardian needs to retain the inhibitor.
    os.execvpe(command[0], command, os.environ)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "guard"))
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--lock-fd", type=int, action="append", default=[])
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--start-ticks")
    parser.add_argument("--config")
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    if args.mode == "guard":
        guard(args.fd, args.ready_fd,
              {"pid": args.pid, "start_ticks": args.start_ticks, "config": args.config})
    else:
        run(args.fd, args.lock_fd, argv[separator + 1:])


if __name__ == "__main__":
    main()
