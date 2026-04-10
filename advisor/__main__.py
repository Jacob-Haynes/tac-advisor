"""Entry point: python -m advisor"""

import argparse
import logging

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Tactical LLM Advisor")
    parser.add_argument("--tak-host", "--fts-host", default="127.0.0.1", help="TAK server host")
    parser.add_argument("--tak-port", "--fts-port", type=int, default=8087, help="TAK server CoT TCP port")
    parser.add_argument("--web-port", type=int, default=8090, help="Web UI port")
    parser.add_argument("--mesh-port", default="/dev/ttyUSB0", help="Meshtastic serial port")
    parser.add_argument("--no-mesh", action="store_true", help="Disable Meshtastic query interface")
    parser.add_argument("--model", default=None, help="GGUF model filename (auto-detect by default)")
    parser.add_argument("--simulate", action="store_true", help="Use simulated tactical data")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("advisor")
    args = parse_args()

    # LLM engine
    from advisor.llm_engine import LLMEngine

    engine = LLMEngine()
    if args.model:
        engine.load(args.model)
    else:
        engine.load()

    # CoT listener
    if args.simulate:
        from advisor.cot_listener import SimulatedCotListener
        listener = SimulatedCotListener(host=args.tak_host, port=args.tak_port)
    else:
        from advisor.cot_listener import CotListener
        listener = CotListener(host=args.tak_host, port=args.tak_port)
    listener.start()

    # Meshtastic query handler
    if not args.no_mesh:
        try:
            from advisor.mesh_query import MeshQueryHandler
            mesh = MeshQueryHandler(engine, listener, None)
            mesh.start(args.mesh_port)
        except Exception:
            log.warning("Meshtastic query handler failed to start", exc_info=True)

    # Web UI
    from advisor.web import app as web_module

    web_module.engine = engine
    web_module.listener = listener
    app = web_module.create_app()

    log.info("Tactical Advisor started — Web UI at http://0.0.0.0:%d", args.web_port)

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.web_port, log_level="warning")
    except KeyboardInterrupt:
        log.info("Shutting down")
        listener.stop()


if __name__ == "__main__":
    main()