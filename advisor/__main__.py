"""Entry point: python -m advisor"""

import argparse
import logging
import os

import uvicorn


def parse_args():
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Tactical LLM Advisor")
    parser.add_argument("--tak-host", "--fts-host", default=env("TAK_HOST", "127.0.0.1"),
                        help="TAK server host (env TAK_HOST)")
    parser.add_argument("--tak-port", "--fts-port", type=int, default=int(env("TAK_PORT", "8087")),
                        help="TAK server CoT TCP port (env TAK_PORT; OTS on the Pi: 8087)")
    parser.add_argument("--web-port", type=int, default=int(env("WEB_PORT", "8090")), help="Web UI port")
    mesh = parser.add_mutually_exclusive_group()
    mesh.add_argument("--serial", nargs="?", const="/dev/ttyUSB0", default=env("ADVISOR_SERIAL") or None,
                      metavar="PORT",
                      help="Standalone: open the radio directly on PORT instead of meshbus. "
                           "Do not use while mesh-bridge is running.")
    mesh.add_argument("--no-mesh", action="store_true", default=env("ADVISOR_NO_MESH", "") == "1",
                      help="Disable the mesh query interface")
    parser.add_argument("--mesh-port", default=None, help=argparse.SUPPRESS)  # old name for --serial
    parser.add_argument("--model", default=env("ADVISOR_MODEL") or None,
                        help="GGUF model filename (auto-detect by default)")
    parser.add_argument("--backend", choices=["auto", "gemini", "local"], default=None,
                        help="LLM backend (env LLM_BACKEND, default auto)")
    parser.add_argument("--simulate", action="store_true", help="Use simulated tactical data (no TAK server)")
    args = parser.parse_args()
    if args.mesh_port and not args.serial and not args.no_mesh:
        args.serial = args.mesh_port
    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("advisor")
    args = parse_args()

    # LLM engine: Gemini if configured, local GGUF model as offline fallback
    from advisor.llm_engine import LLMEngine

    engine = LLMEngine(backend=args.backend)
    engine.load(args.model)
    if not engine.is_loaded:
        log.warning("No LLM backend available — the UI will run but queries return a setup message")

    # CoT listener
    if args.simulate:
        from advisor.cot_listener import SimulatedCotListener
        listener = SimulatedCotListener(host=args.tak_host, port=args.tak_port)
    else:
        from advisor.cot_listener import CotListener
        listener = CotListener(host=args.tak_host, port=args.tak_port)
    listener.start()

    # Mesh query handler (meshbus by default; the bridge owns the radio)
    link = None
    if not args.no_mesh:
        from advisor.mesh_link import MeshbusLink, SerialMeshLink
        from advisor.mesh_query import MeshQueryHandler

        mesh = MeshQueryHandler(engine, listener)
        if args.serial:
            log.info("Mesh queries via serial radio on %s (standalone)", args.serial)
            link = SerialMeshLink(mesh.on_text, port=args.serial)
        else:
            link = MeshbusLink("tac-advisor", mesh.on_text)
        mesh.set_link(link)
        mesh.start()
        link.start()

    # Web UI
    from advisor.web import app as web_module

    web_module.engine = engine
    web_module.listener = listener
    web_module.mesh_link = link
    app = web_module.create_app()

    log.info("Tactical Advisor started — Web UI at http://0.0.0.0:%d", args.web_port)

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.web_port, log_level="warning")
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        listener.stop()
        if link is not None:
            link.stop()


if __name__ == "__main__":
    main()
