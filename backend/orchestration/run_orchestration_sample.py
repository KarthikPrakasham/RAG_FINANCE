import asyncio
import json
from pathlib import Path
from orchestration.chains import OrchestrationService
from api.core.config import get_settings

def main():
    payload = json.loads(Path("backend/orchestration/sample_input.json").read_text())
    service = OrchestrationService(settings=get_settings())

    answer, usage, prompt = asyncio.run(service.generate_answer(
        question=payload["question"],
        history=payload.get("history"),
        context=payload.get("context"),
        system_instruction=payload.get("system_instruction"),
    ))

    print("PROMPT:\n", prompt)
    print("ANSWER:\n", answer)
    print("TOKENS:\n", usage)

if __name__ == "__main__":
    main()
