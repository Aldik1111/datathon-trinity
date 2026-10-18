import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_handoff


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_packet(tmp_path: Path, media: bytes, *, windows_paths=False):
    packet = tmp_path / 'packet'
    baseline = packet / 'baseline'
    baseline.mkdir(parents=True)
    group_id = digest(media)
    (baseline / 'annotations-after-batch.json').write_text('{"private":"synthetic"}', encoding='utf-8')
    comparison = {
        'classes': [{'type': 'synthetic_type'}],
        'rows': [{
            'row_id': 'synthetic-row', 'video_id': 'synthetic-video', 'filename': 'sample.mov',
            'seconds': 1.0, 'user_description': 'synthetic description', 'table_violation': True,
            'annotation_status': 'review', 'source_annotation_events': [],
            'duplicate_group_id': group_id, 'expected_types': ['synthetic_type'],
        }],
    }
    (baseline / 'scenario-comparison.json').write_text(json.dumps(comparison), encoding='utf-8')
    archive_path = baseline / 'Trinity-source-and-model.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('datathon-trinity/artifacts/scenario-model/model.npz', b'synthetic-model')
        archive.writestr('datathon-trinity/artifacts/scenario-model/metadata.json', b'{}')
        archive.writestr('datathon-trinity/artifacts/scenario-model/backbone.onnx', b'synthetic-backbone')
    manifest_path = packet / 'manifest-sha256.csv'
    entries = sorted(path for path in packet.rglob('*') if path.is_file() and path != manifest_path)
    with manifest_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'bytes', 'sha256'])
        writer.writeheader()
        for path in entries:
            writer.writerow({
                'file': (path.relative_to(packet).as_posix().replace('/', '\\')
                         if windows_paths else path.relative_to(packet).as_posix()),
                'bytes': path.stat().st_size,
                'sha256': digest(path.read_bytes()),
            })
    video_dir = tmp_path / 'authorized-videos'
    video_dir.mkdir()
    (video_dir / 'sample.mov').write_bytes(media)
    return packet, video_dir, group_id


def test_prepare_writes_synthetic_fixture_and_prints_only_aggregate_provenance(tmp_path, monkeypatch, capsys):
    packet, video_dir, _ = make_packet(tmp_path, b'synthetic movie bytes', windows_paths=True)
    output = tmp_path / 'artifacts'
    monkeypatch.setattr(prepare_handoff, '_check_ignored_output', lambda path: None)

    assert prepare_handoff.main(['--packet', str(packet), '--video-dir', str(video_dir), '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report['verified_packet_files'] == 3
    assert report['local_media_files'] == 1
    assert report['matched_rows'] == report['matched_unique_groups'] == 1
    prepared = json.loads((output / 'handoff-input' / 'manifest.json').read_text(encoding='utf-8'))
    assert prepared['rows'][0]['group_id'] == prepared['groups'][0]['group_id']
    assert (output / 'handoff-baseline' / 'scenario-model' / 'model.npz').read_bytes() == b'synthetic-model'
    assert 'synthetic description' not in json.dumps(report)
    assert str(video_dir) not in json.dumps(report)
    assert report['backbone_sha256'] == digest(b'synthetic-backbone')


def test_existing_baseline_is_immutable_and_changed_input_requires_new_output(tmp_path, monkeypatch):
    packet, video_dir, _ = make_packet(tmp_path, b'synthetic movie bytes')
    output = tmp_path / 'artifacts'
    second_video_dir = tmp_path / 'second-authorized-videos'
    second_video_dir.mkdir()
    (second_video_dir / 'renamed.mov').write_bytes(b'synthetic movie bytes')
    monkeypatch.setattr(prepare_handoff, '_check_ignored_output', lambda path: None)

    prepare_handoff.prepare(packet, video_dir, output)
    baseline_file = output / 'handoff-baseline' / 'scenario-model' / 'model.npz'
    before = baseline_file.read_bytes()
    with pytest.raises(ValueError, match='provenance differs'):
        prepare_handoff.prepare(packet, second_video_dir, output)
    assert baseline_file.read_bytes() == before

    baseline_file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='baseline differs'):
        prepare_handoff.prepare(packet, video_dir, output)


@pytest.mark.parametrize('relative', ['../outside', './inside', 'C:relative', 'C:/absolute', '\\\\server\\share'])
def test_unsafe_packet_paths_are_rejected(tmp_path, relative):
    root = tmp_path / 'packet'
    root.mkdir()
    with pytest.raises(ValueError, match='Invalid packet path'):
        prepare_handoff._manifest_file(root, relative)


def test_windows_manifest_paths_are_canonical_and_duplicate_spellings_rejected(tmp_path):
    packet, _, _ = make_packet(tmp_path, b'synthetic movie bytes', windows_paths=True)
    records, files = prepare_handoff.verify_packet(packet)
    assert all('/' in record['file'] for record in records)
    assert 'baseline/annotations-after-batch.json' in files

    manifest_path = packet / 'manifest-sha256.csv'
    with manifest_path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['baseline\\scenario-comparison.json',
                         (packet / 'baseline' / 'scenario-comparison.json').stat().st_size,
                         digest((packet / 'baseline' / 'scenario-comparison.json').read_bytes())])
    with pytest.raises(ValueError, match='Invalid packet checksum row'):
        prepare_handoff.verify_packet(packet)


def test_output_inside_source_is_rejected_before_writes(tmp_path, monkeypatch):
    packet, video_dir, _ = make_packet(tmp_path, b'synthetic movie bytes')
    monkeypatch.setattr(prepare_handoff, '_check_ignored_output', lambda path: None)
    with pytest.raises(ValueError, match='inside packet or video source'):
        prepare_handoff.prepare(packet, video_dir, packet / 'artifacts')
