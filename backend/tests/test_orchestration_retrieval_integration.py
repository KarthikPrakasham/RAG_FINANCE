"""
Integration test for the orchestration <-> retrieval contract.

Mocks only the network boundary (embed_query, retrieve_dense — OpenAI +
Pinecone) so the real code runs for everything in between:
    OrchestrationService.retrieve_context()
      -> retrieval.dense_retriever.retrieve_dense()  [mocked]
      -> retrieval.context_builder.build_context()   [real]
    OrchestrationService.generate_grounded_answer()
      -> build_prompt()                               [real]
      -> grounding gate                                [real]
      -> LLMClient.generate()                          [fake client, no network]
"""
import pytest

from api.core.config import Settings
from orchestration.chains import OrchestrationService, UNGROUNDED_RESPONSE
from retrieval.dense_retriever import RetrievedChunk


class FakeLLMClient:
    """Records the prompt it was called with; returns a fixed answer."""

    def __init__(self):
        self.last_prompt = None

    async def generate(self, prompt: str, settings: Settings):
        self.last_prompt = prompt
        return "The Fair Housing Act protects you.", {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


def _settings(grounding_threshold: float = 0.55) -> Settings:
    return Settings(grounding_threshold=grounding_threshold, rag_top_k=3)


@pytest.mark.asyncio
async def test_grounded_chunks_reach_the_llm_with_citations(monkeypatch):
    settings = _settings()
    fake_llm = FakeLLMClient()
    service = OrchestrationService(settings=settings, llm_client=fake_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            score=0.81,
            text="The Fair Housing Act prohibits discrimination in lending on the basis of race.",
            source_doc="FedReserve_Fair_Housing_Act.pdf",
            law="Fair Housing Act",
            section="Sec. 1",
            page_num=3,
            chunk_index=0,
        ),
    ]

    monkeypatch.setattr(
        "retrieval.embed_query.embed_query",
        lambda query, settings=None: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "retrieval.dense_retriever.retrieve_dense",
        lambda query_vector, top_k=None, settings=None: chunks,
    )

    result = await service.generate_grounded_answer(
        question="Was I denied housing credit because of my race?",
        history=[{"role": "user", "content": "Hi"}],
    )

    assert result.answer == "The Fair Housing Act protects you."
    assert result.token_usage["total_tokens"] == 15
    assert result.context.is_grounded is True
    assert len(result.context.citations) == 1
    assert result.context.citations[0].source_doc == "FedReserve_Fair_Housing_Act.pdf"
    # The prompt handed to the LLM must actually contain the retrieved text.
    assert "Fair Housing Act prohibits discrimination" in fake_llm.last_prompt
    assert "Was I denied housing credit" in fake_llm.last_prompt


@pytest.mark.asyncio
async def test_low_score_chunks_are_filtered_and_llm_is_never_called(monkeypatch):
    settings = _settings(grounding_threshold=0.55)
    fake_llm = FakeLLMClient()
    service = OrchestrationService(settings=settings, llm_client=fake_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c2",
            score=0.10,  # below grounding_threshold
            text="Unrelated content.",
            source_doc="Unrelated.pdf",
            law="",
            section="",
            page_num=None,
            chunk_index=None,
        ),
    ]

    monkeypatch.setattr(
        "retrieval.embed_query.embed_query",
        lambda query, settings=None: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "retrieval.dense_retriever.retrieve_dense",
        lambda query_vector, top_k=None, settings=None: chunks,
    )

    result = await service.generate_grounded_answer(question="What's the weather?")

    assert result.answer == UNGROUNDED_RESPONSE
    assert result.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert result.context.is_grounded is False
    assert result.context.citations == []
    assert fake_llm.last_prompt is None  # LLM must not be called when ungrounded


@pytest.mark.asyncio
async def test_no_retrieved_chunks_is_treated_as_ungrounded(monkeypatch):
    settings = _settings()
    fake_llm = FakeLLMClient()
    service = OrchestrationService(settings=settings, llm_client=fake_llm)

    monkeypatch.setattr(
        "retrieval.embed_query.embed_query",
        lambda query, settings=None: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "retrieval.dense_retriever.retrieve_dense",
        lambda query_vector, top_k=None, settings=None: [],
    )

    result = await service.generate_grounded_answer(question="Anything at all?")

    assert result.answer == UNGROUNDED_RESPONSE
    assert result.context.is_grounded is False
    assert fake_llm.last_prompt is None
