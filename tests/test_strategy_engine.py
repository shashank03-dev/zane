from drug_discovery.strategy import ProgramStrategyEngine, TargetProductProfile


def test_strategy_engine_returns_ranked_selection():
    engine = ProgramStrategyEngine()
    result = engine.evaluate_candidates(["CCO", "CCN", "c1ccccc1"], top_k=2)

    assert result["success"] is True
    assert len(result["selected"]) == 2
    assert result["selected"][0]["program_readiness"] >= result["selected"][1]["program_readiness"]


def test_strategy_engine_honors_custom_tpp_name():
    custom_tpp = TargetProductProfile(name="oncology_tpp", min_qed=0.5)
    engine = ProgramStrategyEngine(tpp=custom_tpp)
    result = engine.evaluate_candidates(["CCO"], top_k=1)

    assert result["selected"][0]["tpp"]["tpp"] == "oncology_tpp"
