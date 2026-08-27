"""CLI - vsector command."""
from __future__ import annotations

import argparse
import uvicorn

from .infra.config import get_settings
from .infra.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(prog="vsector", description="Vsector Vector Database")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Start REST + gRPC server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--grpc-port", type=int, default=50051)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("health", help="Check health")
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    if args.cmd == "serve":
        host = args.host or settings.host
        port = args.port or settings.port
        print(f"Starting Vsector REST http://{host}:{port}{settings.api_prefix}  gRPC :{args.grpc_port}")
        uvicorn.run("vsector.api.rest:app", host=host, port=port, reload=args.reload, log_level=settings.log_level.lower())
    elif args.cmd == "health":
        import httpx
        settings = get_settings()
        try:
            r = httpx.get(f"http://{settings.host}:{settings.port}/health", timeout=2)
            print(r.json())
        except Exception as e:
            print(f"health check failed: {e}")
    elif args.cmd == "version":
        print(settings.version)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
