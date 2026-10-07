#!/usr/bin/env python3
"""View the Mirte camera on the MAC, over rosbridge. Never moves the robot.

Runs on the Mac, NOT on the robot -- it needs no ROS installed here at all.

Why rosbridge instead of plain ROS 2: the robot runs Humble and the Mac's conda
env is Jazzy, and ROS 2 does not guarantee cross-distro communication. On top of
that, DDS is peer-to-peer -- the robot opens its own connections back to us, which
an iPhone hotspot's client isolation silently drops. rosbridge has neither problem:
it is one ordinary WebSocket server on the robot, we are the client, and every
frame comes back inside the connection WE opened.

Setup (once):
    pip install roslibpy

Then, in one terminal, open the tunnel and leave it running:
    ssh -N -L 9090:localhost:9090 mirte

And in another:
    ./mac_camera_view.py                 # live view at http://localhost:8081
    ./mac_camera_view.py --once          # save a single frame and exit
    ./mac_camera_view.py --save 20       # save 20 frames into ./frames/
    ./mac_camera_view.py --host 172.20.10.4 --no-tunnel    # skip the tunnel

The camera already publishes JPEG (/camera/image_raw/compressed), so the live
view passes those bytes straight through to the browser without ever decoding
them. That keeps this script dependency-free apart from roslibpy, and it keeps
the robot's CPU out of it -- it idles around load 9 on 4 cores and has very
little headroom.
"""

import argparse
import base64
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import roslibpy
except ImportError:
    sys.exit("roslibpy is missing. Install it with:  pip install roslibpy")


TOPIC = "/camera/image_raw/compressed"
# ROS 2 message names carry the /msg/ segment. rosbridge also accepts the old
# ROS 1 style, but being explicit avoids a silent no-messages-ever subscription.
MSG_TYPE = "sensor_msgs/msg/CompressedImage"


class FrameBuffer:
    """Holds the newest frame and wakes up whoever is waiting for one.

    Only ever the NEWEST frame. If a browser or a slow disk falls behind we drop
    the frames in between rather than queueing them -- on a live camera view a
    stale frame is worse than a skipped one, and an unbounded queue over flaky
    WiFi is how you end up watching a minute-old picture.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

    def put(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_for_new(self, last_seq, timeout=10.0):
        """Block until a frame newer than last_seq arrives. Returns (jpeg, seq)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout):
                return None, last_seq
            return self._jpeg, self._seq

    @property
    def seq(self):
        with self._cond:
            return self._seq


frames = FrameBuffer()


def on_image(msg):
    """rosbridge hands us JSON, so the image arrives base64-encoded in 'data'."""
    try:
        frames.put(base64.b64decode(msg["data"]))
    except Exception as e:  # a malformed frame must not kill the subscriber
        print(f"  bad frame ignored: {type(e).__name__}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Live view: re-serve the frames as MJPEG so a browser can display them.
# ---------------------------------------------------------------------------

INDEX = b"""<!doctype html><meta charset=utf-8><title>Mirte camera</title>
<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}
img{max-width:100%;max-height:100vh;image-rendering:pixelated}</style>
<img src="/stream.mjpg" alt="Mirte camera">
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # one line per frame would bury everything else

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX)))
            self.end_headers()
            self.wfile.write(INDEX)
            return

        if self.path == "/snapshot.jpg":
            jpeg, _ = frames.wait_for_new(frames.seq - 1, timeout=10.0)
            if jpeg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpeg, seq = frames.wait_for_new(seq, timeout=15.0)
                if jpeg is None:
                    break  # camera went quiet; let the browser reconnect
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab was closed. Not an error.


# ---------------------------------------------------------------------------

def connect(host, port, timeout=15.0):
    ros = roslibpy.Ros(host=host, port=port)
    ros.run()
    deadline = time.time() + timeout
    while not ros.is_connected and time.time() < deadline:
        time.sleep(0.1)
    if not ros.is_connected:
        sys.exit(
            f"Could not reach rosbridge at {host}:{port}.\n"
            "  - Is the tunnel up?   ssh -N -L 9090:localhost:9090 mirte\n"
            "  - Is it running?      ssh mirte \"ss -ltn | grep 9090\"\n"
            "  - Bypassing the tunnel? add --no-tunnel --host <robot ip>"
        )
    return ros


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="localhost",
                    help="rosbridge host (default localhost, i.e. through the SSH tunnel)")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--no-tunnel", action="store_true",
                    help="cosmetic: only changes the hint text if the connection fails")
    ap.add_argument("--serve-port", type=int, default=8081,
                    help="local port for the live view (default 8081)")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="frames per second to ask the robot for (default 5). "
                         "Higher costs robot CPU and WiFi it does not have to spare.")
    ap.add_argument("--once", action="store_true", help="save one frame and exit")
    ap.add_argument("--save", type=int, metavar="N", help="save N frames into ./frames/ and exit")
    ap.add_argument("--topic", default=TOPIC)
    args = ap.parse_args()

    ros = connect(args.host, args.port)
    print(f"connected to rosbridge at {args.host}:{args.port}")

    # throttle_rate is in milliseconds between messages, and queue_length=1 tells
    # rosbridge to drop rather than buffer when we cannot keep up.
    topic = roslibpy.Topic(
        ros, args.topic, MSG_TYPE,
        throttle_rate=int(1000 / max(args.fps, 0.1)),
        queue_length=1,
    )
    topic.subscribe(on_image)
    print(f"subscribed to {args.topic} at ~{args.fps:g} fps")

    try:
        if args.once or args.save:
            n = 1 if args.once else args.save
            outdir = "." if args.once else "frames"
            os.makedirs(outdir, exist_ok=True)
            seq = 0
            for i in range(n):
                jpeg, seq = frames.wait_for_new(seq, timeout=20.0)
                if jpeg is None:
                    sys.exit("timed out waiting for a camera frame -- is the camera publishing?")
                path = os.path.join(outdir, "frame.jpg" if args.once else f"frame_{i:04d}.jpg")
                with open(path, "wb") as f:
                    f.write(jpeg)
                print(f"saved {path}  ({len(jpeg)} bytes)")
            return

        # Bind to 127.0.0.1 only. Binding to 0.0.0.0 would put the robot's camera
        # on whatever network this Mac happens to be on.
        server = ThreadingHTTPServer(("127.0.0.1", args.serve_port), Handler)
        server.daemon_threads = True
        print(f"live view: http://localhost:{args.serve_port}   (Ctrl-C to stop)")
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        topic.unsubscribe()
        ros.terminate()


if __name__ == "__main__":
    main()
