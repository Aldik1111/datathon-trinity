"""Run frozen held-out videos through the real local HTTP queue, preserving results."""
import json
import re
import time
from pathlib import Path

from src.ui.client import get_job, load_settings, submit_video


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'artifacts/safe-dataset/manifest.json').read_text(encoding='utf-8'))
    output = root.parents[1] / 'outputs'
    settings = load_settings()
    reports = []
    for number in (39, 26):
        video = next(v for v in manifest['videos'] if re.fullmatch(r'[XХ]-' + str(number) + r'\.mov', v['filename']))
        assert video['split'] == 'test'
        started = time.monotonic()
        job = submit_video(settings, video['filename'], Path(video['path']).read_bytes(), roi=video['roi'])
        print(json.dumps({'video': video['filename'], 'job_id': job['job_id'], 'status': 'processing'}, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            result = get_job(settings, job['job_id'])
            if result['status'] != 'processing':
                break
            time.sleep(1)
        assert result['status'] == 'done', result.get('error', result['status'])
        assert result['demo'] is False and result['analysis_scope'] == 'safe_state'
        (output / f'X-{number}-recognition.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = {'video': video['filename'], 'split': 'test', 'job_id': result['job_id'],
                   'seconds': round(time.monotonic() - started, 2), 'verdict': result['verdict'],
                   'summary': result['summary'],
                   'events': [{k: e[k] for k in ('type', 'start_sec', 'end_sec', 'confidence')} for e in result['violations']]}
        reports.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (output / 'recognition-http-check.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    remaining = list((root / 'artifacts/uploads').glob('*.mov*'))
    assert not remaining, 'Temporary upload or calibration sidecar remains'


if __name__ == '__main__':
    main()
