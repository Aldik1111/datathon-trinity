"""UI fixtures must not contact a developer's running local API."""
import pytest


@pytest.fixture(autouse=True)
def isolate_local_configuration(monkeypatch):
    # A developer's .env can point at private videos and live jobs. Unit and
    # subprocess integration tests supply their own synthetic configuration.
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')


@pytest.fixture(autouse=True)
def isolate_ui_readiness(request, monkeypatch):
    if request.node.path.name not in {'test_ui.py', 'test_safe_ui.py', 'test_scenario_ui.py'}:
        return
    from src.ui import client
    monkeypatch.setattr(client, 'get_readiness', lambda settings: dict(
        default_analysis_type='jury', omitted_analysis_type_routes_to='jury',
        configured_for_jury_contract=True, jury_contract_compatible=None,
        jury_contract_basis='Test fixture: contract not verified.'))
