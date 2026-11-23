#!/usr/bin/env python3
"""Load-on-first-request gateway for llama-server.

The gateway listens on one port and keeps it open while no model is loaded. The first POST request starts llama-server
with the command line given after "--", waits until it answers /health and forwards the request. After --idle-minutes
without a request the server is stopped, which frees the GPU memory; the next request loads the model again.

  python wake-gateway.py --listen 127.0.0.1:8080 --backend-port 8081 --idle-minutes 15 --model-id inkling-2-mini -- \
      llama-server -m model.gguf ... --host 127.0.0.1 --port 8081

- Streamed requests ("stream": true) get the 200 status and the SSE headers at once, then an SSE comment every 10 s
  while the model loads or the request waits for a slot, so that clients and relays keep the connection. When the client
  leaves, the upstream request is closed and llama-server cancels the task.
- Requests that are not streamed wait for the model without a keep-alive: a relay with a short timeout may give up.
- GET /v1/models is answered by the gateway while no model is loaded (listing the models does not load one).
- Only a server that the gateway started is stopped. A server that already answers on the backend port is used as it
  is and left running.
- The server is stopped when the gateway stops (Ctrl+C, SIGTERM). On Windows this also holds for a gateway that is
  killed; elsewhere a gateway killed with SIGKILL leaves the server running.
- GET /__wake_status reports the state as JSON. The server output goes to --server-log.

Standard library only (Python 3.9 or later). The environment of the gateway is passed on to llama-server.
"""

from __future__ import annotations

import argparse
import http.client
import json
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PING_SECONDS = 10.0
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
    "upgrade", "host", "content-length",
}


def log(message: str) -> None:
    sys.stderr.write("%s %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))
    sys.stderr.flush()


_job = None  # Windows job object, kept open for the life of the gateway


def tie_to_gateway(process: subprocess.Popen) -> None:
    """Windows: the server ends when the gateway ends, even a gateway that is killed. Elsewhere: nothing."""
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if _job is None:
            job = kernel32.CreateJobObjectW(None, None)
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not job or not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                return
            _job = job
        kernel32.AssignProcessToJobObject(_job, int(process._handle))
    except (OSError, AttributeError):
        pass


class Backend:
    """The llama-server process behind the gateway: started on demand, stopped when idle."""

    def __init__(self, host: str, port: int, command: list[str], idle_seconds: float, load_timeout: float, log_path: str) -> None:
        self.host = host
        self.port = port
        self.command = command
        self.idle_seconds = idle_seconds
        self.load_timeout = load_timeout
        self.log_path = log_path
        self.lock = threading.Lock()  # one start or stop at a time
        self.count_lock = threading.Lock()
        self.active = 0
        self.last_used = time.monotonic()
        self.process: subprocess.Popen | None = None
        self.last_error = ""
        self.last_load_seconds: float | None = None
        if idle_seconds > 0 and command:
            threading.Thread(target=self._idle_loop, daemon=True).start()

    def begin(self) -> None:
        with self.count_lock:
            self.active += 1
            self.last_used = time.monotonic()

    def end(self) -> None:
        with self.count_lock:
            self.active -= 1
            self.last_used = time.monotonic()

    def get(self, path: str, timeout: float) -> tuple[int, bytes] | None:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read()
        except (OSError, http.client.HTTPException):
            return None
        finally:
            connection.close()

    def up(self, timeout: float = 2.0) -> bool:
        answer = self.get("/health", timeout)  # llama-server answers 503 while it loads the model
        return answer is not None and answer[0] == 200

    def started_here(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure(self) -> str | None:
        """Returns None once the server answers, or the reason why it could not be started."""
        with self.lock:
            if self.up():
                return None
            if not self.command:
                return self._fail("no server answers on port %d and no command line was given to start one" % self.port)
            self._stop_process()  # a leftover that no longer answers
            started = time.monotonic()
            log("no model loaded: starting %s" % self.command[0])
            try:
                with open(self.log_path, "ab") as output:  # a file, not a pipe: nobody has to read it
                    self.process = subprocess.Popen(
                        self.command,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            except OSError as exc:
                return self._fail("the server could not be started: %s" % exc)
            tie_to_gateway(self.process)
            while time.monotonic() - started < self.load_timeout:
                code = self.process.poll()
                if code is not None:
                    self.process = None
                    return self._fail("the server exited with code %d while loading (see %s)" % (code, self.log_path))
                if self.up(1.0):
                    self.last_error = ""
                    self.last_load_seconds = time.monotonic() - started
                    with self.count_lock:
                        self.last_used = time.monotonic()
                    log("model ready after %.0f s" % self.last_load_seconds)
                    return None
                time.sleep(0.5)
            self._stop_process()
            return self._fail("the server did not answer within %.0f s (see %s)" % (self.load_timeout, self.log_path))

    def _fail(self, reason: str) -> str:
        self.last_error = reason
        log(reason)
        return reason

    def _stop_process(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)

    def stop(self) -> None:
        with self.lock:
            self._stop_process()

    def _slots_busy(self) -> bool:
        answer = self.get("/slots", 3.0)  # a client may use the server on its own port
        if answer is None or answer[0] != 200:
            return False
        try:
            return any(isinstance(slot, dict) and slot.get("is_processing") for slot in json.loads(answer[1]))
        except (ValueError, TypeError):
            return False

    def _idle_loop(self) -> None:
        while True:
            time.sleep(30.0)
            if not self.started_here():
                continue
            with self.lock:
                with self.count_lock:
                    idle = time.monotonic() - self.last_used
                    busy = self.active > 0
                if busy or idle < self.idle_seconds or not self.started_here():
                    continue
                if self._slots_busy():
                    with self.count_lock:
                        self.last_used = time.monotonic()
                    continue
                log("idle for %.0f min: stopping the server" % (idle / 60.0))
                self._stop_process()

    def status(self) -> dict:
        with self.count_lock:
            idle = time.monotonic() - self.last_used
            active = self.active
        return {
            "model_loaded": self.up(1.5),
            "started_by_gateway": self.started_here(),
            "server_pid": self.process.pid if self.started_here() else None,
            "active_requests": active,
            "idle_seconds": round(idle),
            "idle_stop_seconds": self.idle_seconds,
            "last_load_seconds": self.last_load_seconds,
            "last_error": self.last_error,
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # only starts, stops and failures are logged

    def _send(self, status: int, payload: bytes, content_type: str = "application/json", headers: list[tuple[str, str]] | None = None) -> None:
        self.close_connection = True
        self.send_response(status)
        for name, value in headers or [("Content-Type", content_type)]:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, status: int, data: object) -> None:
        self._send(status, json.dumps(data).encode())

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "server_error", "code": status}})

    def _upstream_headers(self, body: bytes | None) -> dict[str, str]:
        headers = {name: value for name, value in self.headers.items() if name.lower() not in HOP_BY_HOP}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        return headers

    def _forward(self, body: bytes | None) -> None:
        backend: Backend = self.server.backend
        connection = http.client.HTTPConnection(backend.host, backend.port, timeout=self.server.upstream_timeout)
        try:
            connection.request(self.command, self.path, body=body, headers=self._upstream_headers(body))
            response = connection.getresponse()
            payload = response.read()
            headers = [(name, value) for name, value in response.getheaders() if name.lower() not in HOP_BY_HOP]
            self._send(response.status, payload, headers=headers)
        except (OSError, http.client.HTTPException) as exc:
            self._send_error(502, "the model server did not answer: %s" % exc)
        finally:
            connection.close()

    def _forward_stream(self, body: bytes) -> None:
        backend: Backend = self.server.backend
        connection = http.client.HTTPConnection(backend.host, backend.port, timeout=self.server.upstream_timeout)
        events: queue.Queue = queue.Queue()
        cancelled = threading.Event()

        def pump() -> None:
            try:
                problem = backend.ensure()
                if problem:
                    events.put(("error", problem))
                    return
                if cancelled.is_set():  # the client left while the model was loading
                    return
                connection.request(self.command, self.path, body=body, headers=self._upstream_headers(body))
                response = connection.getresponse()
                events.put(("status", response.status))
                while True:
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    events.put(("chunk", chunk))
                events.put(("end",))
            except Exception as exc:  # closed by us after the client left, or the server went away
                events.put(("error", "the model server did not answer: %s" % exc))

        threading.Thread(target=pump, daemon=True).start()
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        status = 200
        try:
            self.wfile.write(b": waiting\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = events.get(timeout=PING_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if event[0] == "status":
                    status = event[1]
                elif event[0] == "chunk":
                    data = event[1]
                    if status >= 400 and not data.startswith(b"data:"):
                        data = b"data: " + data.strip() + b"\n\n"  # a JSON error body: the 200 status line is already sent
                    self.wfile.write(data)
                    self.wfile.flush()
                elif event[0] == "end":
                    break
                else:
                    message = json.dumps({"error": {"message": event[1], "type": "server_error"}})
                    self.wfile.write(("data: %s\n\n" % message).encode())
                    self.wfile.flush()
                    break
        except OSError:
            pass  # the client left: the upstream request is closed below
        finally:
            cancelled.set()
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(2)
            except OSError:
                pass
            connection.close()

    def _handle(self) -> None:
        backend: Backend = self.server.backend
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/__wake_status":
            self._send_json(200, backend.status())
            return
        length_header = self.headers.get("Content-Length")
        if self.command in ("POST", "PUT", "PATCH") and length_header is None:
            self._send_error(411, "a Content-Length header is required")
            return
        body = self.rfile.read(int(length_header)) if length_header and int(length_header) > 0 else None
        if self.command != "POST":
            # Reading requests never load the model.
            if backend.up(1.0):
                self._forward(body)
            elif path in ("/v1/models", "/models"):
                entry = {"id": self.server.model_id, "object": "model", "owned_by": "local", "meta": {"sleeping": True}}
                self._send_json(200, {"object": "list", "data": [entry]})
            elif path == "/health":
                self._send_json(200, {"status": "ok", "sleeping": True})
            elif self.command == "OPTIONS":
                self._send(204, b"", headers=[("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), ("Access-Control-Allow-Headers", "*")])
            else:
                self._send_error(503, "no model is loaded; it loads on the first POST request")
            return
        streamed = False
        if body:
            try:
                parsed = json.loads(body)
                streamed = isinstance(parsed, dict) and parsed.get("stream") is True
            except ValueError:
                pass
        backend.begin()
        try:
            if streamed:
                self._forward_stream(body)
                return
            problem = backend.ensure()
            if problem:
                self._send_error(503, problem)
                return
            self._forward(body)
        finally:
            backend.end()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _handle


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen", default="127.0.0.1:8080", help="address and port of the gateway (default %(default)s)")
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8081, help="the --port of the llama-server command line (default %(default)s)")
    parser.add_argument("--idle-minutes", type=float, default=15.0, help="minutes without a request before the server is stopped; 0 keeps it (default %(default)s)")
    parser.add_argument("--load-timeout", type=float, default=600.0, help="seconds allowed for the server to answer after its start (default %(default)s)")
    parser.add_argument("--upstream-timeout", type=float, default=1800.0, help="seconds without a byte from the server before a request fails (default %(default)s)")
    parser.add_argument("--model-id", default="local", help="model id listed by /v1/models while no model is loaded (default %(default)s)")
    parser.add_argument("--server-log", default="wake-gateway-server.log", help="file that receives the output of llama-server (default %(default)s)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="after --: the llama-server command line")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    host, _, port = args.listen.rpartition(":")
    if not host or not port.isdigit():
        parser.error("--listen takes address:port")
    if command and str(args.backend_port) not in command:
        log("warning: the command line does not mention port %d (--backend-port)" % args.backend_port)
    backend = Backend(args.backend_host, args.backend_port, command, args.idle_minutes * 60.0, args.load_timeout, args.server_log)
    server = Gateway((host, int(port)), Handler)
    server.backend = backend
    server.model_id = args.model_id
    server.upstream_timeout = args.upstream_timeout

    def leave(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, leave)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, leave)
    log("gateway on %s, llama-server on port %d, stop after %s min without a request" % (args.listen, args.backend_port, args.idle_minutes))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()  # a server started here does not outlive the gateway
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
