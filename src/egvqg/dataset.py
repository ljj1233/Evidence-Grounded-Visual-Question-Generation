"""Pinned official ScienceQA downloads and conservative hint+image candidate import."""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .records import MaterialRecord, sha256_file, stable_id, write_json, write_jsonl

REVISION = "2cbf8318e07b9ece895bb2ae605e71e38d623264"
GITHUB = f"https://raw.githubusercontent.com/lupantech/ScienceQA/{REVISION}"
IMPORT_VERSION = "hint-image-v1"


def download_file(url: str, path: Path, *, max_bytes: int, attempts: int = 3,
                  opener=None, expected_bytes: int | None = None,
                  expected_sha256: str | None = None, reject_html: bool = False) -> dict:
    """Resume partial downloads; never replace a completed raw artifact silently."""
    receipt = path.with_name(path.name + ".download.json")
    if path.exists():
        digest = sha256_file(path)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["url"] != url or saved["sha256"] != digest:
                raise ValueError(f"raw download provenance/hash mismatch: {path}")
            if expected_bytes is not None and path.stat().st_size != expected_bytes:
                raise ValueError(f"downloaded file size mismatch: {path}")
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(f"publisher hash mismatch: {path}")
            return saved
        raise ValueError(f"raw file exists without download receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved["url"] == url and partial.stat().st_size == saved["bytes"]
                and sha256_file(partial) == saved["sha256"]
                and (expected_bytes is None or saved["bytes"] == expected_bytes)
                and (expected_sha256 is None or saved["sha256"] == expected_sha256)):
            partial.replace(path)
            return saved
    if partial.exists() and expected_bytes is not None and partial.stat().st_size == expected_bytes:
        # Recover a fully received file interrupted before receipt/rename, avoiding HTTP 416.
        digest = sha256_file(partial)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"publisher hash mismatch in complete partial: {path.name}")
        saved = {"url": url, "sha256": digest, "bytes": expected_bytes}
        write_json(receipt, saved)
        partial.replace(path)
        return saved
    last_error = "download_failed"
    for _ in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with (opener or urllib.request.urlopen)(request, timeout=60) as response:
                if reject_html and 'text/html' in response.headers.get('Content-Type','').lower():
                    raise ValueError(f"server returned an HTML page instead of data: {path.name}")
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("download range mismatch")
                start = offset if append else 0
                length = response.headers.get("Content-Length")
                expected = start + int(length) if length else None
                if expected is not None and expected > max_bytes:
                    raise ValueError(f"download exceeds byte budget: {path.name} ({expected} bytes)")
                if expected_bytes is not None and expected is not None and expected != expected_bytes:
                    raise ValueError(f"server file size differs from source listing: {path.name}")
                received = start
                with partial.open("ab" if append else "wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > max_bytes:
                            raise ValueError("download exceeds byte budget")
                        handle.write(chunk)
                if expected is not None and received != expected:
                    raise OSError("incomplete download")
                if expected_bytes is not None and received != expected_bytes:
                    raise OSError("incomplete source file")
            saved = {"url": url, "sha256": sha256_file(partial), "bytes": received}
            if expected_sha256 is not None and saved["sha256"] != expected_sha256:
                raise ValueError(f"publisher hash mismatch: {path.name}")
            # A receipt first allows recovery if interrupted immediately before rename.
            write_json(receipt, saved)
            partial.replace(path)
            return saved
        except (OSError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
    raise ValueError(f"download failed after {attempts} attempts: {path.name} ({last_error})")


def import_scienceqa(root: Path, *, split: str = "train", limit: int = 3,
                     seed: int = 0, max_download_mb: int = 1024,
                     offline: bool = False, source_ids: list[str] | None = None) -> dict:
    if split not in {"train", "val", "test"} or limit < 1 or max_download_mb < 1:
        raise ValueError("split must be train/val/test; limit and download budget must be positive")
    root = root.resolve()
    raw = root / "data/raw/scienceqa" / REVISION
    processed = root / "data/processed/scienceqa" / IMPORT_VERSION / f"{split}-n{limit}-seed{seed}"
    if source_ids is not None:
        if not source_ids or len(source_ids) != len(set(source_ids)) or len(source_ids) > limit:
            raise ValueError("explicit IDs must be nonempty, unique and within limit")
        processed = processed.parent / f"{split}-{stable_id('ids', *source_ids)}"
    sources = {}
    def local_source(url: str, path: Path) -> dict:
        receipt = path.with_name(path.name + ".download.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["url"] != url or saved["sha256"] != sha256_file(path):
                raise ValueError(f"raw download provenance/hash mismatch: {path}")
            return saved
        return {"url": url, "sha256": sha256_file(path), "offline": True}
    for name in ("problems.json", "pid_splits.json", "LICENSE-DATA"):
        url = f"{GITHUB}/{name if name == 'LICENSE-DATA' else 'data/scienceqa/' + name}"
        path = raw / name
        if not offline:
            sources[name] = download_file(url, path, max_bytes=64 * 1024 * 1024)
        else:
            sources[name] = local_source(url, path)
    problems = json.loads((raw / "problems.json").read_text())
    splits = json.loads((raw / "pid_splits.json").read_text())
    split_ids = [str(pid) for pid in splits[split]]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("duplicate IDs in official split")
    other_ids = {str(pid) for name in ("train", "val", "test") if name != split for pid in splits[name]}
    if set(split_ids) & other_ids:
        raise ValueError("overlapping official splits")
    decisions, candidates = [], []
    for pid in split_ids:
        if not re.fullmatch(r"[0-9]+", pid):
            raise ValueError("unsafe ScienceQA ID")
        problem = problems[pid]
        if problem["split"] != split:
            raise ValueError(f"split mismatch for {pid}")
        reasons = []
        if not isinstance(problem.get("hint"), str) or not problem["hint"].strip():
            reasons.append("empty_hint")
        if not problem.get("image"):
            reasons.append("no_image")
        decisions.append({"source_id": pid, "split": split, "reasons": reasons,
                          "status": "excluded" if reasons else "candidate"})
        if not reasons:
            candidates.append(pid)
    # Selection depends only on material presence and a fixed seed, never model scores.
    random.Random(seed).shuffle(candidates)
    if source_ids is not None and not set(source_ids) <= set(candidates):
        raise ValueError("explicit IDs must belong to this split's hint+image candidate pool")
    selected = set(source_ids if source_ids is not None else candidates[:limit])
    if not selected:
        raise ValueError("no hint+image candidates in the selected split")
    archive = raw / f"{split}.zip"
    archive_url = f"https://scienceqa.s3.us-west-1.amazonaws.com/images/{split}.zip"
    if not offline:
        sources[archive.name] = download_file(archive_url, archive, max_bytes=max_download_mb * 1024 * 1024)
    else:
        sources[archive.name] = local_source(archive_url, archive)
    materials = []
    with zipfile.ZipFile(archive) as images:
        names = set(images.namelist())
        for decision in decisions:
            pid = decision["source_id"]
            if pid not in selected:
                if decision["status"] == "candidate":
                    decision.update(status="not_selected", reasons=["explicit_diagnostic_selection" if source_ids is not None else "sample_limit"])
                continue
            filename = problems[pid]["image"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"unsafe image filename for {pid}")
            member = f"{split}/{pid}/{filename}"
            if member not in names:
                raise ValueError(f"image missing from official archive: {member}")
            info = images.getinfo(member)
            if info.file_size > 32 * 1024 * 1024:
                raise ValueError(f"oversized image: {member}")
            image = raw / "images" / member
            data = images.read(member)  # ZipFile validates the member CRC.
            if image.exists():
                if image.read_bytes() != data:
                    raise ValueError(f"raw image changed: {member}")
            else:
                image.parent.mkdir(parents=True, exist_ok=True)
                temporary = image.with_suffix(image.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(image)
            image_hash = sha256_file(image)
            row = MaterialRecord(
                material_id=f"scienceqa_{pid}", source_group_id=f"image_{image_hash}",
                split=split, text=problems[pid]["hint"].strip(),
                image_path=str(image.relative_to(root)), image_sha256=image_hash,
                provenance=f"ScienceQA@{REVISION}:{pid}", text_source="hint",
                notes="Candidate only; joint reasoning suitability and leakage require human review.",
            ).to_dict()
            row.update(source_id=pid, processing_version=IMPORT_VERSION, review_status="pending")
            materials.append(row)
            decision.update(status="selected_pending_review", reasons=["nonempty_hint_and_image"],
                            image_sha256=image_hash, source_group_id=row["source_group_id"])
    config = {"revision": REVISION, "processing_version": IMPORT_VERSION, "split": split,
              "limit": limit, "seed": seed, "text_source": "hint",
              "n_source": len(split_ids), "n_candidates": len(candidates),
              "n_selected": len(materials), "sources": sources,
              "excluded_input_fields": ["question", "choices", "answer", "lecture", "solution"],
              "review_status": "pending"}
    if source_ids is not None:
        config.update(selection_mode="explicit_ids_for_diagnostics", source_ids=source_ids)
    # Raw source hashes and selected inputs must agree on repeated import.
    material_file = processed / "materials.jsonl"
    if material_file.exists():
        from .records import read_jsonl
        if read_jsonl(material_file) != materials:
            raise ValueError("processed materials differ; investigate source changes before reuse")
    write_jsonl(material_file, materials)
    write_jsonl(processed / "selection.jsonl", decisions)
    write_json(processed / "manifest.json", config)
    return {"materials": str(material_file), "selected": len(materials),
            "candidates": len(candidates), "split": split, "review_status": "pending"}


def bundle_scienceqa_images(root: Path, source_id: str, *, split: str,
                           output_dir: Path, n_choices: int) -> dict:
    """Preserve material figures in I, without inventing visual-choice alignment.

    The standard single-image importer remains unchanged for historical runs.
    Added headers identify source roles, never describe image content or answers.
    """
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    if not re.fullmatch(r'[0-9]+', source_id) or split not in ('train', 'val', 'test'):
        raise ValueError('safe source ID and official split required')
    root=root.resolve();raw=root/'data/raw/scienceqa'/REVISION
    prefix=f'{split}/{source_id}/';assets=[]
    with zipfile.ZipFile(raw/f'{split}.zip') as archive:
        names=[n for n in archive.namelist() if n.startswith(prefix) and re.fullmatch(
            r'image(?:_[0-9]+)?\.png',n[len(prefix):])]
        excluded_choices=[n for n in archive.namelist() if n.startswith(prefix) and re.fullmatch(
            r'choice_[0-9]+\.png',n[len(prefix):])]
        def order(name):
            base=Path(name).stem
            return (0,0) if base=='image' else (1,int(base.split('_')[1]))
        names.sort(key=order)
        if not names:raise ValueError('missing original material figures')
        for name in names:
            data=archive.read(name);path=raw/'images'/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=data:raise ValueError('original image changed: '+name)
            if not path.exists():
                temporary=path.with_suffix('.png.part');temporary.write_bytes(data);temporary.replace(path)
            base=Path(name).stem
            role='Original figure' if base=='image' else 'Additional figure '+base.split('_')[1]
            image=Image.open(BytesIO(data)).convert('RGB')
            assets.append({'role':role,'archive_member':name,'path':str(path.relative_to(root)),
                           'sha256':sha256_file(path),'width':image.width,'height':image.height})
    choice_policy={'excluded_unmapped_choice_assets':sorted(excluded_choices),'n_text_choices':n_choices,
                   'choice_policy':'Canonical text choices stay in Q; archive choice_N indices are not assumed to match shuffled text options.'}
    if len(assets)==1:
        return {'image_path':assets[0]['path'],'image_sha256':assets[0]['sha256'],
                'image_assets':assets,'image_bundle_version':'material-figures-v1','layout':'original unchanged',**choice_policy}
    width=max(a['width'] for a in assets);height=max(a['height'] for a in assets);header=32
    columns=1 if len(assets)==2 else 2
    canvas=Image.new('RGB',(columns*width,((len(assets)+columns-1)//columns)*(height+header)),'white')
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=18)
    for i,a in enumerate(assets):
        x=(i%columns)*width;y=(i//columns)*(height+header)
        draw.text((x+4,y+4),a['role'],font=font,fill='black')
        image=Image.open(root/a['path']).convert('RGB')
        canvas.paste(image,(x+(width-image.width)//2,y+header))
    buffer=BytesIO();canvas.save(buffer,format='PNG');data=buffer.getvalue()
    target=output_dir/f'{source_id}.png';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=data:raise ValueError('image bundle changed; use a new output directory')
    if not target.exists():target.write_bytes(data)
    return {'image_path':str(target.relative_to(root)),'image_sha256':sha256_file(target),
            'image_assets':assets,'image_bundle_version':'material-figures-v1',**choice_policy,
            'layout':f'{columns} columns, original pixels, role headers only, no resize/crop/OCR'}


def normalize_materials(*, input_path: Path, output_path: Path, dataset_name: str,
                        id_field: str, text_field: str, image_field: str,
                        image_root: Path | None = None, split_field: str = 'split',
                        group_field: str | None = None, question_field: str | None = None,
                        questions_output: Path | None = None) -> dict:
    """Map flat JSONL fields to the shared material contract; no dataset-specific evaluation.

    Invalid rows fail before writing anything. Images are referenced, never copied;
    questions are exported only when explicitly mapped; unmapped answers and solutions are not copied.
    """
    from .records import read_jsonl
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', dataset_name):
        raise ValueError('dataset_name must be a safe nonempty identifier')
    source=input_path.resolve(); destination=output_path.resolve()
    if source==destination:raise ValueError('normalized output cannot overwrite raw input')
    if (question_field is None)!=(questions_output is None):raise ValueError('question_field and questions_output must be supplied together')
    qdestination=questions_output.resolve() if questions_output else None
    if qdestination in (source,destination):raise ValueError('question output must be separate from raw input and materials')
    source_rows=read_jsonl(source)
    if not source_rows:raise ValueError('input JSONL is empty or missing')
    images=(image_root or source.parent).resolve();seen=set();rows=[];questions=[]
    for index,r in enumerate(source_rows,1):
        sid=r.get(id_field);text=r.get(text_field);image_value=r.get(image_field)
        if sid is None or not str(sid).strip() or str(sid) in seen:raise ValueError(f'row {index}: missing/duplicate source ID')
        if not isinstance(text,str) or not text.strip():raise ValueError(f'row {index}: missing text')
        if not isinstance(image_value,str) or not image_value.strip():raise ValueError(f'row {index}: missing image path')
        seen.add(str(sid));image=(images/image_value).resolve()
        if not image.is_file():raise ValueError(f'row {index}: image file missing')
        digest=sha256_file(image)
        group=r.get(group_field) if group_field else None
        row=MaterialRecord(material_id=stable_id('material',dataset_name,sid),
            source_group_id=stable_id('source',dataset_name,group) if group is not None else 'image_'+digest,
            split=str(r.get(split_field,'unspecified')),text=text.strip(),image_path=str(image),image_sha256=digest,
            provenance=f'{dataset_name}:source_id={sid}',text_source=text_field).to_dict()
        row.update(dataset_name=dataset_name,source_id=str(sid),processing_version='flat-jsonl-v1')
        rows.append(row)
        if question_field:
            q=r.get(question_field)
            if q is not None and not isinstance(q,str):raise ValueError(f'row {index}: question must be text or null')
            if isinstance(q,str) and q.strip():
                questions.append({'material_id':row['material_id'],'question':q.strip(),'source':row['provenance']})
    if destination.exists() and read_jsonl(destination)!=rows:
        raise ValueError('normalized output already contains different materials; use another processing location')
    if qdestination and qdestination.exists() and read_jsonl(qdestination)!=questions:
        raise ValueError('question output already contains different questions; use another processing location')
    write_jsonl(destination,rows)
    if qdestination:write_jsonl(qdestination,questions)
    manifest={'dataset_name':dataset_name,'source_path':str(source),'source_sha256':sha256_file(source),'source_rows':len(source_rows),
              'image_root':str(images),
              'normalized_rows':len(rows),'field_map':{'id':id_field,'text':text_field,'image':image_field,
                                                    'split':split_field,'group':group_field,'question':question_field},
              'image_policy':'reference original file, SHA256 verified',
              'processing_version':'flat-jsonl-v1','excluded_metadata':'unmapped fields and source answers/solutions not copied'}
    if qdestination:
        manifest.update(questions_output=str(qdestination),provided_questions=len(questions),
                        rows_without_supplied_question=len(rows)-len(questions))
    write_json(destination.with_suffix('.manifest.json'),manifest)
    return {'materials':str(destination),'n_materials':len(rows),
            **({'questions':str(qdestination),'n_supplied_questions':len(questions)} if qdestination else {})}

