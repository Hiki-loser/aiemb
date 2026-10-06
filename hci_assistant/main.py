#!/usr/bin/env python3
"""Smart Home Assistant (Vosk Streaming ASR + llama.cpp LLM).
Usage: python3 main.py [--debug]
"""

import sys
import os
import signal


def main():
    debug = "--debug" in sys.argv

    from pipeline import Pipeline
    pipeline = Pipeline(debug=debug)

    def handle_signal(signum, frame):
        print("\n[Shutdown]")
        pipeline.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.shutdown()
    except Exception as e:
        print("[Fatal] %s" % e)
        pipeline.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
