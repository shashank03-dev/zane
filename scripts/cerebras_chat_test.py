"""Simple Cerebras chat completion smoke test.

Usage:
  1) Copy .env.example to .env
  2) Set CEREBRAS_API_KEY in .env
  3) Run: python scripts/cerebras_chat_test.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Fallback: continue with shell environment variables only.
        pass

    api_key = os.getenv("CEREBRAS_API_KEY")
    model = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")

    if not api_key:
        print("Missing CEREBRAS_API_KEY. Add it to .env or export it in your shell.")
        return 1

    try:
        from cerebras.cloud.sdk import Cerebras
    except Exception as exc:
        print("cerebras-cloud-sdk is not installed. Run: pip install cerebras-cloud-sdk")
        print(f"Details: {exc}")
        return 1

    try:
        client = Cerebras(api_key=api_key)
        completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Why is fast inference important for drug discovery pipelines?"},
            ],
            model=model,
            max_completion_tokens=256,
            temperature=0.2,
            top_p=1,
            stream=False,
        )
    except Exception as exc:
        print(f"Cerebras request failed: {exc}")
        return 1

    text = completion.choices[0].message.content
    print("Cerebras response:\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
