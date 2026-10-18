"""Separate-process local HTTP smoke with synthetic MOV and actual local weights.

No user videos, cloud services, persistent API keys or image output are used.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import requests


def verify(model_path, output):
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    video = output / 'synthetic.mov'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 120))
    if not writer.isOpened():
        raise RuntimeError('Synthetic MOV encoder unavailable')
    for index in range(180):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        x = 10 + index % 100
        cv2.rectangle(frame, (x, 35), (x + 25, 85), (50, 170, 200), -1)
        writer.write(frame)
    writer.release()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    headers = {'X-API-Key': token}
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ, API_KEY=token, ANALYZER_MODE='real', DEFAULT_ANALYSIS_TYPE='bank_scenarios',
               ANALYZER_CALLABLE='src.vision.scenario_classifier:analyze_video',
               MODEL_PATH=str(model_path), SCENARIO_MODEL_PATH=str(model_path),
               UPLOAD_DIR=str(output/'uploads'), API_DB_PATH=str(output/'jobs.sqlite3'),
               ANALYZER_TIMEOUT_SEC='300', MAX_PENDING_JOBS='2', MAX_UPLOAD_MB='2',
               STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    with (output/'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'src.api.app:create_app', '--factory',
                                    '--host', '127.0.0.1', '--port', str(port), '--no-access-log'],
                                   env=env, stdout=log, stderr=log, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline = time.monotonic()+30
            while True:
                try:
                    readiness = requests.get(url+'/api/v1/readiness',headers=headers,timeout=1)
                    if readiness.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Local server did not become available')
                time.sleep(.1)
            assert requests.get(url+'/api/v1/readiness',timeout=2).status_code == 401
            assert requests.get(url+'/api/v1/jobs/missing',timeout=2).status_code == 401
            assert requests.post(url+'/api/v1/analyze',files={'file':('x.mov',b'fake')},timeout=2).status_code == 401
            assert requests.post(url+'/api/v1/analyze',headers=headers,files={'file':('x.zip',b'fake')},timeout=2).status_code == 415
            assert readiness.json()['omitted_analysis_type_routes_to'] == 'bank_scenarios'
            assert readiness.json()['jury_contract_compatible'] is False
            results, summaries = [], []
            for name in ('synthetic.mov', 'renamed-unseen.mov'):
                start = time.monotonic()
                pending = requests.post(url+'/api/v1/analyze',headers=headers,
                                        files={'file':(name,video.read_bytes(),'video/quicktime')},timeout=10)
                assert pending.status_code == 202
                job_id = pending.json()['job_id']
                statuses = [pending.json()['status']]
                while time.monotonic()-start < 300:
                    response = requests.get(url+'/api/v1/jobs/'+job_id, headers=headers,timeout=5)
                    response.raise_for_status()
                    result = response.json()
                    statuses.append(result['status'])
                    if result['status'] != 'processing':
                        break
                    time.sleep(.15)
                assert result['status'] == 'done', result.get('error','Job timed out')
                assert result['analysis_scope'] == 'bank_scenarios' and result['demo'] is False
                assert result['model_support_policy']['min_positive_media_groups'] in (1,2)
                frame_count = 0
                for event in result['violations']:
                    assert 1 <= len(event['evidence_frames']) <= 3
                    for frame in event['evidence_frames']:
                        jpeg = base64.b64decode(frame['image_base64'], validate=True)
                        assert cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR) is not None
                        assert 0 <= frame['timestamp_sec'] <= 6
                        frame_count += 1
                assert not list((output/'uploads').glob('*.mov*'))
                results.append({k:v for k,v in result.items() if k != 'job_id'})
                summaries.append(dict(statuses=sorted(set(statuses)),seconds=round(time.monotonic()-start,3),
                                      verdict=result['verdict'], event_count=len(result['violations']),
                                      valid_jpeg_frames=frame_count,upload_cleanup=True))
            assert results[0] == results[1], 'Filename changed model result'
            report = dict(model_version=results[0]['model_version'],model_support_policy=results[0]['model_support_policy'],
                          readiness=readiness.json(),checks=summaries,renaming_identical=True,
                          quality_claim=False,media='synthetic geometric MOV only')
            (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False))
            return report
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    verify(args.model,args.output)
