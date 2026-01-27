#!/usr/bin/env python3
"""Query vLLM server with the same prompt as main.py."""

import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Query vLLM server")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="dummy",  # vLLM doesn't require auth by default
    )

    # Same prompt and params as main.py
    prompt = (
        "Write a haiku about artificial intelligence. "
        "End with the Haiku, don't say anything else."
    )

    completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=50,
    )

    print(f"Prompt: {prompt}")
    print(f"Response: {completion.choices[0].message.content}")


if __name__ == "__main__":
    main()
