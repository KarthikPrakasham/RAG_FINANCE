import asyncio
import json
from pathlib import Path
from orchestration.chains import build_prompt
from orchestration.llm_client import LLMClient
from api.core.config import get_settings

def main():
    payload = json.loads(Path("backend/orchestration/sample_input.json").read_text())
    settings = get_settings()
    llm_client = LLMClient()

    prompt = build_prompt(
        question=payload["question"],
        history=payload.get("history"),
        context=payload.get("context"),
        system_instruction=payload.get("system_instruction"),
    )
    answer, usage = asyncio.run(llm_client.generate(prompt=prompt, settings=settings))

    print("PROMPT:\n", prompt)
    print("ANSWER:\n", answer)
    print("TOKENS:\n", usage)

if __name__ == "__main__":
    main()
