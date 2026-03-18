"""Tests for synthesis research workflow with internet and AI enrichments."""

from drug_discovery.synthesis import RetrosynthesisPlanner


def test_plan_synthesis_with_research_enriches_output(monkeypatch):
    planner = RetrosynthesisPlanner()

    monkeypatch.setattr(
        planner,
        "plan_synthesis",
        lambda target_smiles, max_depth=5: {
            "target": target_smiles,
            "success": True,
            "num_steps": 3,
            "estimated_yield": 0.7,
        },
    )

    monkeypatch.setattr(
        planner.internet_search,
        "search_web",
        lambda query, max_results=5: [
            {"title": "Route A", "url": "https://example.org/route-a", "snippet": "", "source": "duckduckgo"}
        ],
    )

    from drug_discovery.web_scraping.scraper import AISynthesisChat

    monkeypatch.setattr(
        AISynthesisChat,
        "generate_synthesis_brief",
        lambda self, smiles, target_protein=None, research_hits=None: {
            "model_id": "fake-llama",
            "brief": "Use convergent coupling with protected amine intermediate.",
        },
    )

    result = planner.plan_synthesis_with_research(
        target_smiles="CCO",
        target_protein="EGFR",
        use_internet=True,
        use_ai_chat=True,
    )

    assert result["success"] is True
    assert result["research_hits"]
    assert result["ai_synthesis_guidance"]
    assert result["ai_model_id"] == "fake-llama"


def test_plan_synthesis_with_research_without_external_enrichment(monkeypatch):
    planner = RetrosynthesisPlanner()

    monkeypatch.setattr(
        planner,
        "plan_synthesis",
        lambda target_smiles, max_depth=5: {"target": target_smiles, "success": True},
    )

    result = planner.plan_synthesis_with_research(
        target_smiles="CCN",
        use_internet=False,
        use_ai_chat=False,
    )

    assert result["success"] is True
    assert "research_hits" in result
    assert result["research_hits"] == []
    assert "ai_synthesis_guidance" not in result
