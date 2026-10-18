"""Actual adapter errors must not fall back to demo or a binary negative."""
import json
from pathlib import Path

import pytest

from src.api.settings import Settings
from src.vision.adapter import AnalysisError, run_analysis
from tests.test_video_coverage import make_video


def configured(tmp_path):
    return Settings(api_key='synthetic-test-key', upload_dir=tmp_path / 'uploads', max_upload_mb=1,
                    analyzer_mode='real', default_analysis_type='jury', jury_analyzer='bank_scenarios',
                    analyzer_callable='src.vision.safe_classifier:analyze_video',
                    model_path=str(tmp_path / 'safe'), scenario_model_path=str(tmp_path / 'scenario'),
                    rules_path=None)


def selected(path, mode='bank_scenarios'):
    Path(str(path) + '.analysis.json').write_text(json.dumps({'analysis_type': mode}))
    Path(str(path) + '.api-mode.json').write_text('{"analysis_type":"jury"}')


def test_jury_does_not_guess_contract_without_task_metadata(tmp_path):
    path = tmp_path / 'synthetic.mov'
    Path(str(path) + '.analysis.json').write_text('{"analysis_type":"bank_scenarios"}')
    with pytest.raises(AnalysisError, match='mode metadata'):
        run_analysis(str(path), configured(tmp_path))


def test_jury_rejects_a_safe_selection_instead_of_using_its_model(tmp_path):
    path = tmp_path / 'synthetic.mov'
    selected(path, 'safe_state')
    with pytest.raises(AnalysisError, match='configured local scenario'):
        run_analysis(str(path), configured(tmp_path))


@pytest.mark.parametrize('corrupt_model', [False, True])
def test_real_scenario_loader_failure_is_explicit(tmp_path, monkeypatch, corrupt_model):
    path = tmp_path / 'synthetic.mov'
    make_video(path)
    selected(path)
    settings = configured(tmp_path)
    # Isolate the stable environment based model interface during this test.
    monkeypatch.setenv('MODEL_PATH', settings.scenario_model_path)
    if corrupt_model:
        directory = Path(settings.scenario_model_path)
        directory.mkdir()
        (directory / 'metadata.json').write_text('not valid metadata')
        (directory / 'model.npz').write_bytes(b'not valid weights')
        (directory / 'backbone.onnx').write_bytes(b'not a backbone')
    with pytest.raises(AnalysisError, match='model could not be loaded'):
        run_analysis(str(path), settings)


def test_jury_corrupt_mov_does_not_return_a_negative(tmp_path):
    path = tmp_path / 'corrupt.mov'
    path.write_bytes(b'corrupt synthetic bytes')
    selected(path)
    with pytest.raises(AnalysisError, match='Video cannot be opened'):
        run_analysis(str(path), configured(tmp_path))
