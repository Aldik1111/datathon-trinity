from unittest.mock import patch

import pytest

from src.ui.client import APIClientError


def test_expired_cached_job_is_not_reused_for_a_live_result_link():
    from scripts.run_scenario_batch import reusable_job
    saved = {'source_media_sha256': 'same-media', 'result': {'job_id': 'expired', 'status': 'done', 'model_version': 'v1'}}
    with patch('scripts.run_scenario_batch.get_job', side_effect=APIClientError('Ошибка API (404): Job not found')):
        assert reusable_job(saved, 'same-media', 'v1', object()) is None


def test_valid_cached_job_is_refreshed_from_live_api_and_wrong_model_is_rejected():
    from scripts.run_scenario_batch import reusable_job
    saved = {'source_media_sha256': 'same-media', 'result': {'job_id': 'exists', 'status': 'done', 'model_version': 'v1'}}
    live = {'job_id': 'exists', 'status': 'done', 'model_version': 'v1', 'analysis_scope': 'bank_scenarios', 'demo': False}
    with patch('scripts.run_scenario_batch.get_job', return_value=live):
        assert reusable_job(saved, 'same-media', 'v1', object()) == live
    with patch('scripts.run_scenario_batch.get_job', return_value={**live, 'model_version': 'v2'}):
        assert reusable_job(saved, 'same-media', 'v1', object()) is None


def test_connection_failure_does_not_silently_create_a_replacement_job():
    from scripts.run_scenario_batch import reusable_job
    saved = {'source_media_sha256': 'same-media', 'result': {'job_id': 'exists', 'status': 'done', 'model_version': 'v1'}}
    with patch('scripts.run_scenario_batch.get_job', side_effect=APIClientError('Connection failed')):
        with pytest.raises(APIClientError):
            reusable_job(saved, 'same-media', 'v1', object())
