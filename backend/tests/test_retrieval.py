from retrieval.router import RetrievalService


def test_retrieval_service_returns_documents_for_policy_queries():
    service = RetrievalService()
    results = service.retrieve(
        "What protections exist if I think I was denied housing credit because of my race?"
    )

    assert results, "expected at least one retrieved chunk"
    assert any("Fair_Housing" in item["source_doc"] or "Fair Housing" in item["source_doc"] for item in results)
