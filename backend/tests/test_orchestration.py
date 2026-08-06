from orchestration.chains import build_prompt


def test_build_prompt_includes_context_history_and_question():
    history = [
        {"role": "user", "content": "What is the current trend?"},
        {"role": "assistant", "content": "It is rising."},
    ]
    context = [
        {"source_doc": "fred.csv", "content": "Consumer credit increased in April."},
    ]

    prompt = build_prompt(
        question="How should I interpret this trend?",
        history=history,
        context=context,
        system_instruction="You are a finance assistant.",
    )

    assert "You are a finance assistant." in prompt
    assert "History:" in prompt
    assert "What is the current trend?" in prompt
    assert "Context:" in prompt
    assert "Consumer credit increased in April." in prompt
    assert "User question:" in prompt
    assert "How should I interpret this trend?" in prompt
