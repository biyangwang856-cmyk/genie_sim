"""Open an SSH tunnel to the team's locally bound trajectory studio."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import webbrowser


LOG = Path(__file__).resolve().with_name("launcher.log")


def connected(url):
    try:
        with urllib.request.urlopen(url + "studio/health", timeout=2) as response:
            data = json.load(response)
        return data.get("service") == "nexus-trajectory-studio" and data.get("worker_ready") is True
    except (OSError, ValueError):
        return False


def main(host, port, *, open_browser):
    url = f"http://127.0.0.1:{port}/"
    if connected(url):
        if open_browser:
            webbrowser.open(url)
        return

    forward = f"127.0.0.1:{port}:127.0.0.1:{port}"
    command = ["ssh", "-N", "-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
               "-L", forward, host]
    with LOG.open("a", encoding="utf-8") as log:
        tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for _ in range(40):
                if tunnel.poll() is not None:
                    raise RuntimeError(f"SSH tunnel closed. See {LOG}")
                if connected(url):
                    if open_browser:
                        webbrowser.open(url)
                    tunnel.wait()
                    return
                time.sleep(.5)
            raise RuntimeError("The studio did not respond through the tunnel")
        finally:
            if tunnel.poll() is None:
                tunnel.terminate()
                tunnel.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GENIE_STUDIO_SSH_HOST", "genie-lab"))
    parser.add_argument("--port", type=int, default=os.environ.get("GENIE_STUDIO_PORT", "18765"))
    parser.add_argument("--no-open", action="store_true")
    options = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@-]*", options.host):
        parser.error("--host must be an OpenSSH host alias or user@host")
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        main(options.host, options.port, open_browser=not options.no_open)
    except (OSError, RuntimeError) as error:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"Could not open the studio: {error}\n")
        parser.exit(1, f"{error}\n")
