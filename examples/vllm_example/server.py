#!/usr/bin/env python3
"""Launch vLLM as an OpenAI-compatible server."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server")
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B',
                        help='HuggingFace model (default: Qwen/Qwen2.5-0.5B)')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model,
        '--host', args.host,
        '--port', str(args.port),
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
