from drug_discovery.integrations import INTEGRATIONS, get_integration_status


def test_elite_integrations_registered():
    expected_keys = {
        "molecular_transformer",
        "diffdock",
        "torchdrug",
        "openfold",
        "openmm",
        "pistachio",
    }
    assert expected_keys.issubset(set(INTEGRATIONS.keys()))


def test_elite_integrations_have_urls():
    for key in ("molecular_transformer", "diffdock", "torchdrug", "openfold", "openmm", "pistachio"):
        status = get_integration_status(key)
        assert status.url.startswith("https://github.com/")
        assert status.submodule_path is not None
