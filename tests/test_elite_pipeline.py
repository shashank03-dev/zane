from drug_discovery.elite_stack import EliteStackPipeline


def test_elite_pipeline_ranks_candidates():
    pipeline = EliteStackPipeline()
    result = pipeline.run(
        molecules=["CCO", "CCN", "c1ccccc1"],
        reactants="CCO.CN",
        target_protein="EGFR",
        top_k=2,
    )

    assert result["success"] is True
    assert len(result["ranked_candidates"]) == 2
    assert result["ranked_candidates"][0]["composite_score"] >= result["ranked_candidates"][1]["composite_score"]
    assert "torchdrug" in result["integrations"]
