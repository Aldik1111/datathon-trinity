"""Validate a local handoff packet and prepare ignored baseline/input artifacts.

This command reads an extracted handoff packet and an authorized local MOV folder.
It never sends media to a service and prints only aggregate counts/checksums.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = 'datathon-trinity/artifacts/'
MAX_ARTIFACT_BYTES = 50_000_000
MAX_ARTIFACT_TOTAL_BYTES = 300_000_000
SHA256_RE = re.compile(r'^[a-f0-9]{64}$')
MANIFEST_ROW_FIELDS = (
    'row_id', 'video_id', 'filename', 'seconds', 'user_description',
    'table_violation', 'annotation_status', 'source_annotation_events',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contained(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _manifest_file(packet: Path, relative: str) -> Path:
    # Packet manifests may use either Windows or POSIX separators. Canonicalize
    # before checking so duplicate spellings cannot bypass path validation.
    relative = relative.replace('\\', '/')
    drive, _ = ntpath.splitdrive(relative)
    if not relative or drive or any(piece in ('', '.', '..') for piece in relative.split('/')):
        raise ValueError('Invalid packet path')
    part = PurePosixPath(relative)
    if part.is_absolute() or any(piece in ('', '.', '..') for piece in part.parts):
        raise ValueError('Invalid packet path')
    target = packet.joinpath(*part.parts).resolve()
    if not _contained(packet, target) or not target.is_file():
        raise ValueError('Invalid packet file')
    return target


def verify_packet(packet: Path) -> tuple[list[dict[str, str]], dict[str, Path]]:
    packet = Path(packet).resolve()
    csv_path = packet / 'manifest-sha256.csv'
    if not packet.is_dir() or not csv_path.is_file():
        raise ValueError('Packet or checksum manifest missing')
    records = []
    seen = set()
    with csv_path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'file', 'bytes', 'sha256'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Invalid packet checksum manifest')
        for row in reader:
            rel = row.get('file', '').replace('\\', '/')
            size_text, expected = row.get('bytes', ''), row.get('sha256', '').lower()
            drive, _ = ntpath.splitdrive(rel)
            if drive or rel in seen or not size_text.isdigit() or not SHA256_RE.fullmatch(expected):
                raise ValueError('Invalid packet checksum row')
            seen.add(rel)
            target = _manifest_file(packet, rel)
            if target.stat().st_size != int(size_text) or sha256_file(target) != expected:
                raise ValueError('Packet integrity mismatch')
            records.append(dict(file=rel, bytes=size_text, sha256=expected))
    if not records:
        raise ValueError('Empty packet checksum manifest')
    by_name = {record['file']: _manifest_file(packet, record['file']) for record in records}
    return records, by_name


def _archive_artifacts(archive_path: Path) -> tuple[dict[str, bytes], str]:
    planned: dict[str, bytes] = {}
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(ARCHIVE_PREFIX):
                continue
            relative = entry.filename[len(ARCHIVE_PREFIX):]
            path = PurePosixPath(relative)
            if (not relative or path.is_absolute() or '\\' in relative
                    or any(piece in ('', '.', '..') for piece in relative.split('/'))):
                raise ValueError('Unsafe baseline artifact path')
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or entry.file_size > MAX_ARTIFACT_BYTES:
                raise ValueError('Unsafe baseline artifact')
            total += entry.file_size
            if total > MAX_ARTIFACT_TOTAL_BYTES or relative in planned:
                raise ValueError('Unsafe or duplicate baseline artifact')
            planned[relative] = archive.read(entry)
    required = {'scenario-model/model.npz', 'scenario-model/backbone.onnx'}
    if not planned or not required.issubset(planned):
        raise ValueError('Baseline scenario model or backbone missing')
    return planned, sha256_file(archive_path)


def _local_movs(video_dir: Path) -> tuple[dict[str, list[Path]], int]:
    video_dir = Path(video_dir).resolve()
    if not video_dir.is_dir():
        raise ValueError('Video directory missing')
    by_hash: dict[str, list[Path]] = {}
    count = 0
    for path in sorted(video_dir.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != '.mov':
            continue
        count += 1
        by_hash.setdefault(sha256_file(path), []).append(path.resolve())
    if not count:
        raise ValueError('No local MOV files found')
    return by_hash, count


def _build_manifest(packet: Path, files: dict[str, Path], video_dir: Path,
                    artifacts: dict[str, bytes], archive_sha256: str):
    comparison_path = files.get('baseline/scenario-comparison.json')
    annotation_path = files.get('baseline/annotations-after-batch.json')
    archive_path = files.get('baseline/Trinity-source-and-model.zip')
    if not comparison_path or not annotation_path or not archive_path:
        raise ValueError('Required packet evidence is missing')
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    local_by_hash, local_count = _local_movs(video_dir)
    classes = comparison.get('classes')
    rows = comparison.get('rows')
    if not isinstance(classes, list) or not isinstance(rows, list) or not rows:
        raise ValueError('Invalid source comparison')
    target_types = [item.get('type') for item in classes]
    if any(not isinstance(item, str) or not item for item in target_types) or len(set(target_types)) != len(target_types):
        raise ValueError('Invalid source class list')
    manifest = dict(schema_version=1, target_type_ids=target_types, rows=[], groups=[])
    by_group: dict[str, list[dict]] = {}
    seen_rows = set()
    for source in rows:
        row_id, group_id = source.get('row_id'), source.get('duplicate_group_id')
        if not isinstance(row_id, str) or not row_id or row_id in seen_rows:
            raise ValueError('Invalid or duplicate source row')
        seen_rows.add(row_id)
        if not isinstance(group_id, str) or not SHA256_RE.fullmatch(group_id.lower()):
            raise ValueError('Invalid source media digest')
        matches = local_by_hash.get(group_id.lower())
        if not matches:
            raise ValueError('A source media hash has no local MOV match')
        original = matches[0]
        row = {key: source[key] for key in MANIFEST_ROW_FIELDS if key in source}
        if len(row) != len(MANIFEST_ROW_FIELDS):
            raise ValueError('Incomplete source annotation row')
        types = source.get('expected_types')
        if (not isinstance(types, list) or len(set(types)) != len(types)
                or any(item not in target_types for item in types)):
            raise ValueError('Invalid source scenario types')
        row.update(group_id=group_id.lower(), table_type_ids=types,
                   original_media=dict(sha256=group_id.lower(), available_local=True, path=str(original)))
        manifest['rows'].append(row)
        by_group.setdefault(group_id.lower(), []).append(row)
    for digest, group_rows in by_group.items():
        manifest['groups'].append(dict(
            group_id=digest,
            row_ids=[row['row_id'] for row in group_rows],
            training_type_ids=sorted({kind for row in group_rows for kind in row['table_type_ids']}),
            local_original_path=group_rows[0]['original_media']['path'],
        ))
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    integrity = dict(
        verified_packet_files=len(files),
        local_media_files=local_count,
        matched_rows=len(manifest['rows']),
        matched_unique_groups=len(manifest['groups']),
        packet_manifest_sha256=sha256_file(packet / 'manifest-sha256.csv'),
        source_annotations_sha256=sha256_file(annotation_path),
        baseline_archive_sha256=archive_sha256,
        baseline_model_sha256=sha256_bytes(artifacts['scenario-model/model.npz']),
        backbone_sha256=sha256_bytes(artifacts['scenario-model/backbone.onnx']),
        prepared_manifest_sha256=sha256_bytes(manifest_bytes),
    )
    integrity_bytes = json.dumps(integrity, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    return manifest_bytes, integrity_bytes, integrity


def _check_ignored_output(output: Path) -> None:
    output = Path(output).resolve()
    repo = REPO_ROOT.resolve()
    if not _contained(repo, output) or output == repo:
        raise ValueError('Output must be a repository-local ignored directory')
    relative = os.path.relpath(output, repo).replace(os.sep, '/')
    result = subprocess.run(
        ['git', '-c', f'safe.directory={repo}', '-C', str(repo),
         'check-ignore', '--quiet', '--no-index', '--', relative],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise ValueError('Output directory is not ignored by Git')


def _preflight_immutable_directory(root: Path, planned: dict[str, bytes], *, label='baseline') -> None:
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Immutable {label} path is not a directory')
    descendants = list(root.rglob('*'))
    if any(path.is_symlink() for path in descendants):
        raise ValueError(f'Immutable {label} contains a symbolic link')
    current = {path.relative_to(root).as_posix() for path in descendants if path.is_file()}
    if current != set(planned):
        raise ValueError(f'Existing {label} artifact set differs; use a new output directory')
    for relative, expected in planned.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f'Existing {label} differs; use a new output directory')


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise ValueError('Existing provenance differs; use a new output directory')
        return
    with path.open('xb') as stream:
        stream.write(content)


def prepare(packet: Path, video_dir: Path, output: Path) -> dict:
    packet, video_dir, output = Path(packet).resolve(), Path(video_dir).resolve(), Path(output).resolve()
    if _contained(packet, output) or _contained(video_dir, output):
        raise ValueError('Output may not be inside packet or video source')
    _check_ignored_output(output)
    records, files = verify_packet(packet)
    archive_path = files.get('baseline/Trinity-source-and-model.zip')
    if not archive_path:
        raise ValueError('Baseline archive missing from packet')
    artifacts, archive_sha256 = _archive_artifacts(archive_path)
    manifest_bytes, integrity_bytes, report = _build_manifest(packet, files, video_dir, artifacts, archive_sha256)
    if report['verified_packet_files'] != len(records):
        raise ValueError('Packet integrity count mismatch')
    baseline_root = output / 'handoff-baseline'
    input_root = output / 'handoff-input'
    _preflight_immutable_directory(baseline_root, artifacts)
    _preflight_immutable_directory(input_root, {
        'manifest.json': manifest_bytes,
        'integrity.json': integrity_bytes,
    }, label='input provenance')
    if not baseline_root.exists():
        for relative, content in artifacts.items():
            _write_once(baseline_root.joinpath(*PurePosixPath(relative).parts), content)
    _write_once(input_root / 'manifest.json', manifest_bytes)
    _write_once(input_root / 'integrity.json', integrity_bytes)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True, help='Extracted local Trinity-handoff packet directory')
    parser.add_argument('--video-dir', required=True, help='Authorized local MOV directory; scanned recursively')
    parser.add_argument('--output', required=True, help='New Git-ignored output root inside this repository')
    args = parser.parse_args(argv)
    try:
        report = prepare(args.packet, args.video_dir, args.output)
    except Exception:
        print('ERROR: handoff preparation failed; check local inputs, integrity, and output provenance.', file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
