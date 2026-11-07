"""CPU-only native-engine stand-in for child ownership and shutdown tests."""
import signal
import sys
import time

if "--ignore-term" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(1)
