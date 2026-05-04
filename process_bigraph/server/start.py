"""
CLI entry point for hosting process-bigraph processes over a network protocol.

Usage:
    python -m process_bigraph.server.start --host 0.0.0.0 --port 22222
    python -m process_bigraph.server.start --protocol rest --port 22222

Currently supported protocols: rest (FastAPI/uvicorn). Future: grpc, etc.
Each protocol's server-side dependencies are gated behind an optional extra.
"""

import fire


def start(host: str = "0.0.0.0", port: int = 22222, protocol: str = "rest"):
    """Start a process-bigraph protocol server.

    Args:
        host: bind address (default 0.0.0.0)
        port: bind port (default 22222)
        protocol: which transport to expose (default 'rest'). Each protocol
            has its own server-side dependency extra:
              rest: pip install process-bigraph[server-rest]
    """
    from process_bigraph import allocate_core

    if protocol == "rest":
        try:
            import uvicorn
            from process_bigraph.server.rest import start_server
        except ImportError as e:
            raise ImportError(
                "REST server requires the [server-rest] extra: "
                "pip install process-bigraph[server-rest]"
            ) from e
        core = allocate_core()
        app = start_server(core)
        uvicorn.run(app, host=host, port=port)
    else:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Supported: rest"
        )


if __name__ == "__main__":
    fire.Fire(start)
