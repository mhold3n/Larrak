"""WebSocket server for live dashboard event broadcasting.

Broadcasts execution events to connected dashboard clients.

Usage:
    # Start server
    python -m provenance.ws_server --port 8765

    # In orchestrator code
    from provenance.ws_server import start_background_server
    start_background_server()
"""

from __future__ import annotations

import argparse
import asyncio
import threading

try:
    import websockets

    # Explicitly verify which 'serve' to use or use websockets.serve directly
    from websockets.server import serve as ws_serve  # type: ignore

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions or different structure
        import websockets

        ws_serve = websockets.serve  # type: ignore
        WEBSOCKETS_AVAILABLE = True
    except ImportError:
        WEBSOCKETS_AVAILABLE = False
        websockets = None  # type: ignore

from campro.logging import get_logger
from provenance.execution_events import ExecutionEvent, add_listener, get_history

log = get_logger(__name__)

# Connected clients
_clients: set = set()
_server_task: asyncio.Task | None = None
_loop: asyncio.AbstractEventLoop | None = None


async def handler(websocket) -> None:  # type: ignore
    """Handle WebSocket connection."""
    _clients.add(websocket)
    log.info(f"Dashboard connected ({len(_clients)} total)")

    try:
        # Send event history on connect
        history = get_history(50)
        for event in history:
            await websocket.send(event.to_json())

        # Keep connection alive
        async for _ in websocket:
            pass  # We only send, not receive
    except Exception as e:
        log.debug(f"WebSocket error: {e}")
    finally:
        _clients.discard(websocket)
        log.info(f"Dashboard disconnected ({len(_clients)} total)")


def broadcast_event(event: ExecutionEvent) -> None:
    """Broadcast event to all connected clients (sync wrapper)."""
    if not _clients or not _loop:
        return

    message = event.to_json()

    async def send_all() -> None:
        disconnected = set()
        for client in _clients.copy():
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
        _clients.difference_update(disconnected)

    try:
        asyncio.run_coroutine_threadsafe(send_all(), _loop)
    except Exception as e:
        log.debug(f"Broadcast error: {e}")


async def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run the WebSocket server."""
    if not WEBSOCKETS_AVAILABLE:
        log.error("websockets not installed. Run: pip install websockets")
        return

    global _loop
    _loop = asyncio.get_event_loop()

    # Register as event listener
    add_listener(broadcast_event)

    log.info(f"Starting WebSocket server on ws://{host}:{port}")

    async with ws_serve(handler, host, port):
        await asyncio.Future()  # Run forever


_server_thread: threading.Thread | None = None


def start_background_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start server in background thread (idempotent)."""
    if not WEBSOCKETS_AVAILABLE:
        log.warning("websockets not installed, live dashboard disabled")
        return

    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        log.info(f"WebSocket server already running on port {port}")
        return

    def run() -> None:
        try:
            asyncio.run(run_server(host, port))
        except Exception as e:
            log.error(f"WebSocket server thread failed: {e}")

    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()
    log.info(f"WebSocket server started in background on port {port}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Dashboard WebSocket server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    if not WEBSOCKETS_AVAILABLE:
        print("Error: websockets not installed. Run: pip install websockets")
        return 1

    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        log.info("Server stopped")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
