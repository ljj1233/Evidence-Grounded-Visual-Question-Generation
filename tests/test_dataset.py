"""Offline candidate import: split isolation, source preservation and input leakage."""
import json
import zipfile
from pathlib import Path

import pytest

from egvqg.dataset import REVISION, download_file, import_scienceqa, bundle_scienceqa_images
from egvqg.records import read_jsonl, sha256_file


def test_scienceqa_import_preserves_raw_and_excludes_answer_fields(tmp_path):
    raw = tmp_path / 'data/raw/scienceqa' / REVISION
    raw.mkdir(parents=True)
    problems = {
        '1': dict(split='train', hint='A rule', image='image.png', question='ORIGINAL QUESTION',
                  choices=['ORIGINAL OPTION'], answer=0, lecture='LECTURE', solution='SOLUTION'),
        '2': dict(split='train', hint='', image='image.png'),
        '3': dict(split='train', hint='Rule', image=None),
        '4': dict(split='test', hint='Test-only rule', image='image.png'),
        '5': dict(split='train', hint='Another rule', image='image.png'),
    }
    (raw / 'problems.json').write_text(json.dumps(problems))
    (raw / 'pid_splits.json').write_text(json.dumps({'train': ['1', '2', '3', '5'], 'val': [], 'test': ['4']}))
    (raw / 'LICENSE-DATA').write_text('synthetic test fixture')
    with zipfile.ZipFile(raw / 'train.zip', 'w') as archive:
        for pid in ('1', '5'):
            archive.writestr(f'train/{pid}/image.png', b'same synthetic image')
    before = sha256_file(raw / 'problems.json')
    result = import_scienceqa(tmp_path, limit=3, offline=True)
    rows = read_jsonl(Path(result['materials']))
    assert {row['source_id'] for row in rows} == {'1', '5'}
    assert len({row['source_group_id'] for row in rows}) == 1
    assert all(row['review_status'] == 'pending' and row['split'] == 'train' for row in rows)
    rendered = json.dumps(rows)
    assert all(secret not in rendered for secret in ('ORIGINAL QUESTION', 'ORIGINAL OPTION', 'LECTURE', 'SOLUTION'))
    decisions = read_jsonl(Path(result['materials']).with_name('selection.jsonl'))
    assert next(d for d in decisions if d['source_id'] == '2')['reasons'] == ['empty_hint']
    assert next(d for d in decisions if d['source_id'] == '3')['reasons'] == ['no_image']
    assert before == sha256_file(raw / 'problems.json')
    assert import_scienceqa(tmp_path, limit=3, offline=True) == result
    explicit = import_scienceqa(tmp_path, limit=1, source_ids=['5'], offline=True)
    assert [row['source_id'] for row in read_jsonl(Path(explicit['materials']))] == ['5']
    with pytest.raises(ValueError, match='candidate pool'):
        import_scienceqa(tmp_path, limit=1, source_ids=['4'], offline=True)
    with pytest.raises(ValueError, match='unique'):
        import_scienceqa(tmp_path, limit=3, source_ids=['1', '1'], offline=True)
    (raw / 'images/train/1/image.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='raw image changed'):
        import_scienceqa(tmp_path, limit=3, offline=True)


def test_download_cache_checks_hash_and_resumes_partial(tmp_path, monkeypatch):
    import io
    import egvqg.dataset as module
    requests = []

    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 3-5/6', 'Content-Length': '3'}

    def fetch(request, timeout):
        requests.append(request)
        assert request.get_header('Range') == 'bytes=3-'
        return Response(b'def')

    monkeypatch.setattr(module.urllib.request, 'urlopen', fetch)
    path = tmp_path / 'raw.zip'
    path.with_name('raw.zip.part').write_bytes(b'abc')
    receipt = download_file('https://example.invalid/raw.zip', path, max_bytes=10)
    assert path.read_bytes() == b'abcdef'
    assert download_file('https://example.invalid/raw.zip', path, max_bytes=10) == receipt
    assert len(requests) == 1
    path.write_bytes(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        download_file('https://example.invalid/raw.zip', path, max_bytes=10)


@pytest.mark.parametrize('mode',['single','extra_figure','unmapped_choices'])
def test_original_image_bundle_preserves_figures_without_guessing_choice_alignment(tmp_path,mode):
    from PIL import Image
    from io import BytesIO
    raw=tmp_path/'data/raw/scienceqa'/REVISION;raw.mkdir(parents=True)
    def png(color):
        b=BytesIO();Image.new('RGB',(40,30),color).save(b,format='PNG');return b.getvalue()
    files={'image.png':png('red')}
    if mode=='extra_figure':files['image_2.png']=png('blue')
    if mode!='single':files.update({'choice_0.png':png('green'),'choice_1.png':png('yellow')})
    with zipfile.ZipFile(raw/'train.zip','w') as z:
        for name,data in files.items():z.writestr('train/1/'+name,data)
    result=bundle_scienceqa_images(tmp_path,'1',split='train',output_dir=tmp_path/'data/processed/bundle',n_choices=2)
    assert len(result['image_assets'])==(2 if mode=='extra_figure' else 1)
    for a in result['image_assets']:assert (tmp_path/a['path']).read_bytes()==files[Path(a['path']).name]
    if mode=='extra_figure':
        assert [a['role'] for a in result['image_assets']]==['Original figure','Additional figure 2']
        assert Image.open(tmp_path/result['image_path']).size==(40,124)
    else:assert result['image_path'].endswith('/images/train/1/image.png')
    assert len(result['excluded_unmapped_choice_assets'])==(0 if mode=='single' else 2)
    before=sha256_file(tmp_path/result['image_path'])
    assert bundle_scienceqa_images(tmp_path,'1',split='train',output_dir=tmp_path/'data/processed/bundle',n_choices=2)==result
    assert sha256_file(tmp_path/result['image_path'])==before
