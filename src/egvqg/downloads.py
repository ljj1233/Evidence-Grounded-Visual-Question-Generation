"""Raw public dataset acquisition; separate from material filtering and model runs."""
from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from .dataset import download_file
from .records import sha256_file, utc_now, write_json

MMQA_REVISION='4dd14328c6d02a4daa357cc6032915a0b14602e3'
WEBQA_REVISION='81cc377aaddb2cf9de98cf082a2795ce53ccc1f4'
WEBQA_MIRROR_REVISION='5d2222ebafe1443d2e31f396a5eafb8803394146'


def _download_bundle(root: Path, files: dict[str,str], *, workers: int = 2, max_file_gb: int = 8) -> dict:
    root.mkdir(parents=True,exist_ok=True)
    receipts={};failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download_file,url,root/name,max_bytes=max_file_gb*1024**3):name for name,url in files.items()}
        for future in as_completed(futures):
            name=futures[future]
            try:
                receipts[name]=future.result()
                print(f'complete: {name} ({receipts[name]["bytes"]:,} bytes)',flush=True)
            except Exception as exc:
                failures[name]=f'{type(exc).__name__}: {exc}'
            write_json(root/'download_status.json',{'updated_at':utc_now(),'completed':receipts,'failures':failures})
    if failures:raise ValueError(f'downloads incomplete: {sorted(failures)}; see download_status.json')
    return receipts


def download_multimodalqa(project_root: str | Path='.', *, workers: int=2) -> dict:
    if not 1<=workers<=8:raise ValueError('workers must be between 1 and 8')
    root=Path(project_root).resolve()/'data/raw/multimodalqa'/MMQA_REVISION
    base=f'https://raw.githubusercontent.com/allenai/multimodalqa/{MMQA_REVISION}'
    names=('MMQA_train.jsonl.gz','MMQA_dev.jsonl.gz','MMQA_test.jsonl.gz',
           'MMQA_texts.jsonl.gz','MMQA_tables.jsonl.gz','MMQA_images.jsonl.gz')
    files={name:f'{base}/dataset/{name}' for name in names}
    files['README.upstream.md']=f'{base}/README.md'
    files['final_dataset_images.zip']='https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip'
    receipts=_download_bundle(root,files,workers=workers)
    counts={}
    for name in names:
        count=0
        with gzip.open(root/name,'rt',encoding='utf-8') as handle:
            for line in handle:
                if line.strip():json.loads(line);count+=1
        counts[name]=count
    image_paths=[]
    with gzip.open(root/'MMQA_images.jsonl.gz','rt',encoding='utf-8') as handle:
        for line in handle:
            if line.strip():image_paths.append(json.loads(line)['path'])
    with zipfile.ZipFile(root/'final_dataset_images.zip') as archive:
        for member in archive.infolist():
            relative=PurePosixPath(member.filename)
            if relative.is_absolute() or '..' in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('unsafe archive member')
            target=root.joinpath(*relative.parts)
            if member.is_dir():continue
            if target.exists():
                import zlib
                crc=0
                with target.open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
                if target.stat().st_size!=member.file_size or crc!=member.CRC:raise ValueError(f'extracted image changed: {relative}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.part')
            with archive.open(member) as source,temporary.open('wb') as output:
                shutil.copyfileobj(source,output,1024*1024)
            temporary.replace(target)
    missing=[path for path in image_paths if not (root/'final_dataset_images'/path).is_file()]
    if missing:raise ValueError(f'{len(missing)} MMQA metadata image paths missing after extraction')
    manifest={'dataset':'MultiModalQA','repository':'https://github.com/allenai/multimodalqa',
              'repository_revision':MMQA_REVISION,'downloaded_at':utc_now(),'status':'complete',
              'files':receipts,'record_counts':counts,'image_metadata_paths_verified':len(image_paths),
              'raw_data_policy':'Original files preserved. No filtering, normalization or model evaluation.'}
    write_json(root/'manifest.json',manifest)
    return {'root':str(root),'status':'complete','record_counts':counts,'images_verified':len(image_paths)}


def _public_opener(proxy=None):
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy,'http':proxy} if proxy else {}))


def _drive_listing(folder_id,opener):
    """Read a publicly shared folder. Embedded view covers the 50-item UI boundary."""
    import re
    import html
    rows={}
    url='https://drive.google.com/drive/folders/'+folder_id
    with opener.open(url,timeout=40) as response:page=response.read().decode('utf-8')
    match=re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'])*)'",page)
    if not match:raise ValueError('public Google Drive listing missing')
    payload=re.sub(r'\\x([0-9a-fA-F]{2})',lambda m:chr(int(m[1],16)),match[1])
    for item in json.loads(payload)[0]:
        rows[item[2]]={'id':item[0],'name':item[2],'mime':item[3],
                       'bytes':item[13] if isinstance(item[13],int) else None}
    if len(rows)>=50:
        with opener.open('https://drive.google.com/embeddedfolderview?id='+folder_id,timeout=40) as response:
            page=response.read().decode('utf-8')
        for href,body in re.findall(r'<a href="([^"]+)"[^>]*>(.*?)</a>',page,re.S):
            title=re.search(r'class="flip-entry-title">(.*?)</div>',body,re.S)
            fid=re.search(r'/file/d/([^/]+)',html.unescape(href))
            if title and fid:
                name=html.unescape(re.sub('<[^>]+>','',title[1]))
                rows.setdefault(name,{'id':fid[1],'name':name,'mime':'application/octet-stream','bytes':None})
    return rows


def _drive_url(file_id):
    return 'https://drive.usercontent.google.com/download?id='+file_id+'&export=download&confirm=t'


def _seven_zip(project_root,opener):
    """Use a pinned official standalone tool; no system/environment installation."""
    import tarfile
    tool_dir=Path(project_root).resolve()/'outputs/cache/tools/7zip-25.01'
    archive=tool_dir/'7z2501-linux-x64.tar.xz'
    download_file('https://github.com/ip7z/7zip/releases/download/25.01/7z2501-linux-x64.tar.xz',archive,
                  max_bytes=10*1024**2,opener=opener.open,expected_bytes=1571044,
                  expected_sha256='4ca3b7c6f2f67866b92622818b58233dc70367be2f36b498eb0bdeaaa44b53f4',reject_html=True)
    executable=tool_dir/'7zz'
    with tarfile.open(archive) as bundle:
        member=bundle.getmember('7zz')
        data=bundle.extractfile(member).read()
        if executable.exists() and executable.read_bytes()!=data:raise ValueError('cached 7zip executable changed')
        if not executable.exists():executable.write_bytes(data)
    executable.chmod(0o755)
    return executable


def _extract_7z(executable,archive,root,label):
    """7z validates CRCs during extraction; interrupted directories are never promoted."""
    import subprocess
    staging=root/(label+'.extracting')
    ready=root/label
    if ready.exists():return ready
    staging.mkdir(exist_ok=True)
    with (root/(label+'.extract.log')).open('w') as log:
        result=subprocess.run([str(executable),'x',str(archive),'-o'+str(staging),'-y','-mmt=2','-bsp0'],
                              stdout=log,stderr=subprocess.STDOUT,check=False)
    if result.returncode:raise ValueError(f'7zip extraction failed ({result.returncode}); see {label}.extract.log')
    staging.rename(ready)
    return ready


def download_webqa(project_root: str | Path='.', *, workers: int=3, proxy: str | None=None,
                   image_mirror: bool=False) -> dict:
    import re
    import urllib.request
    if not 1<=workers<=8:raise ValueError('workers must be between 1 and 8')
    root=Path(project_root).resolve()/'data/raw/webqa'/WEBQA_REVISION
    root.mkdir(parents=True,exist_ok=True)
    previous_manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else None
    opener=_public_opener(proxy)
    listing_path=root/'source_files.json'
    if listing_path.exists():sources=json.loads(listing_path.read_text())
    else:
        top=_drive_listing('1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ',opener)
        chunks=_drive_listing(top['WebQA_imgs_7z_chunks']['id'],opener)
        expected={f'imgs.7z.{i:03d}' for i in range(1,52)}
        if set(chunks)!=expected:raise ValueError(f'incomplete Drive volume listing: expected 51, got {len(chunks)}')
        sources={name:top[name] for name in ('WebQA_data_first_release.7z','imgs.lineidx')}
        sources.update(chunks)
        for name,item in sources.items():
            if Path(name).name!=name:raise ValueError('unsafe upstream filename')
            item['url']=_drive_url(item['id'])
            if item['bytes'] is None:
                request=urllib.request.Request(item['url'],headers={'Range':'bytes=0-0'})
                with opener.open(request,timeout=40) as response:
                    match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
                    if not match:raise ValueError('could not resolve Drive file size')
                    item['bytes']=int(match[1]);response.read(1)
        write_json(listing_path,sources)
    if not all((root/name).exists() for name in sources):
        required=sum(v['bytes'] for name,v in sources.items() if not (root/name).exists())+80*1024**3
        if shutil.disk_usage(root).free<required:raise ValueError('insufficient disk space for WebQA archives and extracted images')
    base=f'https://raw.githubusercontent.com/WebQnA/WebQA/{WEBQA_REVISION}'
    for remote,local in (('README.md','README.upstream.md'),('LICENSE','LICENSE.upstream')):
        download_file(f'{base}/{remote}',root/local,max_bytes=1024**2,opener=opener.open,reject_html=True)
    tool=_seven_zip(project_root,opener)
    image_sources={}
    if image_mirror:
        mirror_path=root/'image_mirror.json'
        if mirror_path.exists():mirror=json.loads(mirror_path.read_text())
        else:
            api=f'https://huggingface.co/api/datasets/Ancci/webqa_large/tree/{WEBQA_MIRROR_REVISION}'
            with urllib.request.urlopen(api,timeout=40) as response:tree=json.load(response)
            mirror={'repository':'Ancci/webqa_large','revision':WEBQA_MIRROR_REVISION,
                    'metadata_url':api,'files':{x['path']:{'bytes':x['size'],'sha256':x['lfs']['oid']} for x in tree
                                              if re.fullmatch(r'imgs\.7z\.\d{3}',x.get('path',''))}}
            write_json(mirror_path,mirror)
        for name,spec in mirror['files'].items():
            if name not in sources or spec['bytes']!=sources[name]['bytes']:raise ValueError('image mirror differs from official file listing')
            image_sources[name]={**spec,'url':f'https://hf-mirror.com/datasets/Ancci/webqa_large/resolve/{WEBQA_MIRROR_REVISION}/{name}?download=true',
                                 'official_url':sources[name]['url'],'source_kind':'community_mirror_with_sha256'}
        if len(image_sources)!=51:raise ValueError('image mirror must provide all 51 volumes')
    def fetch(name,item):
        mirror=image_sources.get(name)
        if mirror and (root/name).exists():
            # Keep the actual recorded origin of an already completed official file.
            receipt=json.loads((root/(name+'.download.json')).read_text())
            return download_file(receipt['url'],root/name,max_bytes=2*1024**3,expected_bytes=item['bytes'],expected_sha256=mirror['sha256'])
        return download_file(mirror['url'] if mirror else item['url'],root/name,max_bytes=2*1024**3,
                             expected_bytes=item['bytes'],expected_sha256=mirror['sha256'] if mirror else None,
                             opener=_public_opener().open if mirror else opener.open,reject_html=True)
    receipts={};failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(fetch,name,item):name for name,item in sources.items()}
        for job in as_completed(jobs):
            name=jobs[job]
            try:
                receipts[name]=job.result();print(f'complete: {name} ({receipts[name]["bytes"]:,} bytes)',flush=True)
            except Exception as exc:failures[name]=f'{type(exc).__name__}: {exc}'
            write_json(root/'download_status.json',{'updated_at':utc_now(),'completed':receipts,'failures':failures})
    if failures:raise ValueError(f'WebQA downloads incomplete: {sorted(failures)}')
    print('extracting WebQA annotations...',flush=True)
    metadata=_extract_7z(tool,root/'WebQA_data_first_release.7z',root,'annotations')
    print('extracting WebQA image TSV (two decoder threads)...',flush=True)
    images=_extract_7z(tool,root/'imgs.7z.001',root,'image_store')
    train_files=list(metadata.rglob('WebQA_train_val.json'));test_files=list(metadata.rglob('WebQA_test.json'))
    tsv_files=list(images.rglob('imgs.tsv'))
    if len(train_files)!=1 or len(test_files)!=1 or len(tsv_files)!=1:raise ValueError('unexpected WebQA archive layout')
    train=json.loads(train_files[0].read_text());test=json.loads(test_files[0].read_text())
    if not isinstance(train,dict) or not isinstance(test,dict) or set(train)&set(test):raise ValueError('invalid/overlapping WebQA QA splits')
    from collections import Counter
    splits=Counter(item.get('split','unspecified') for item in train.values())
    # Verify all byte offsets and image IDs without expanding base64 into duplicate files.
    tsv=tsv_files[0]
    print('validating WebQA image index and TSV...',flush=True)
    offsets=[int(line) for line in (root/'imgs.lineidx').read_text().splitlines() if line.strip()]
    if not offsets or offsets[0]!=0 or any(b<=a for a,b in zip(offsets,offsets[1:])):raise ValueError('invalid image offsets')
    digest=__import__('hashlib').sha256();image_ids=set();count=0
    with tsv.open('rb') as handle:
        while True:
            position=handle.tell();line=handle.readline()
            if not line:break
            if count>=len(offsets) or offsets[count]!=position:raise ValueError('image index offset mismatch')
            parts=line.rstrip(b'\r\n').split(b'\t')
            if len(parts)!=2:raise ValueError('invalid image TSV row')
            image_id=parts[0].decode('ascii')
            if image_id in image_ids:raise ValueError('duplicate image TSV ID')
            if int(image_id)%10000000!=count:raise ValueError('image ID/index mapping mismatch')
            image_ids.add(image_id);digest.update(line);count+=1
    if count!=len(offsets):raise ValueError('image index line count mismatch')
    requested=set()
    for record in list(train.values())+list(test.values()):
        for field in ('img_posFacts','img_negFacts','img_Facts'):
            for fact in record.get(field,[]):requested.add(str(fact['image_id']))
    if requested-image_ids:raise ValueError(f'{len(requested-image_ids)} referenced WebQA images missing')
    annotation_files={str(p.relative_to(root)):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in (train_files[0],test_files[0])}
    if previous_manifest and (previous_manifest['image_store']['sha256']!=digest.hexdigest()
                              or previous_manifest['annotation_files']!=annotation_files):
        raise ValueError('previously validated raw WebQA content changed')
    manifest={'dataset':'WebQA','repository':'https://github.com/WebQnA/WebQA','repository_revision':WEBQA_REVISION,
              'status':'complete','downloaded_at':previous_manifest['downloaded_at'] if previous_manifest else utc_now(),
              'validated_at':utc_now(),'sources':sources,'files':receipts,
              'image_mirror':image_sources or None,
              'question_counts':{**dict(splits),'test':len(test),'train_val_combined':len(train)},
              'image_count':count,'referenced_image_ids_verified':len(requested),
              'image_store':{'path':str(tsv.relative_to(root)),'bytes':tsv.stat().st_size,'sha256':digest.hexdigest()},
              'annotation_files':annotation_files,
              'raw_data_policy':'Original annotations, archives, line index and base64 image TSV preserved. No model evaluations or material filtering.'}
    write_json(root/'manifest.json',manifest)
    verify_webqa_references(project_root)
    return {'root':str(root),'status':'complete','question_counts':manifest['question_counts'],'images':count}


def verify_webqa_references(project_root: str | Path='.') -> dict:
    """Offline verification of every QA image lookup using the official index rule.

    The full TSV's byte hash and all offsets are checked during acquisition;
    this verifies ID-to-offset binding, including test's distinct img_Facts field.
    """
    root=Path(project_root).resolve()/'data/raw/webqa'/WEBQA_REVISION
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete':raise ValueError('WebQA acquisition is incomplete')
    tsv=root/manifest['image_store']['path']
    if tsv.stat().st_size!=manifest['image_store']['bytes']:raise ValueError('image TSV size changed')
    offsets=[int(x) for x in (root/'imgs.lineidx').read_text().splitlines() if x.strip()]
    per_file={};all_ids=set()
    for name,saved in manifest['annotation_files'].items():
        path=root/name
        if sha256_file(path)!=saved['sha256']:raise ValueError('raw annotation changed')
        records=json.loads(path.read_text());ids=set()
        for record in records.values():
            for field in ('img_posFacts','img_negFacts','img_Facts'):
                ids.update(str(fact['image_id']) for fact in record.get(field,[]))
        per_file[name]=len(ids);all_ids.update(ids)
    with tsv.open('rb') as stream:
        for image_id in sorted(all_ids,key=lambda x:int(x)%10000000):
            index=int(image_id)%10000000
            if not 0<=index<len(offsets):raise ValueError('referenced image index out of range')
            stream.seek(offsets[index]);prefix=stream.read(64)
            if b'\t' not in prefix or prefix.split(b'\t',1)[0].decode('ascii')!=image_id:
                raise ValueError('referenced image ID/offset mismatch')
    manifest.update(referenced_image_ids_verified=len(all_ids),
                    reference_ids_by_annotation_file=per_file,
                    reference_fields_verified=['img_posFacts','img_negFacts','img_Facts'],
                    reference_index_rule='int(image_id) % 10000000, from the official WebQA demo',
                    references_verified_at=utc_now())
    write_json(manifest_path,manifest)
    return {'all_reference_ids_verified':len(all_ids),'per_file':per_file}


def finish_download_tail(url: str, partial: Path, destination: Path, *, expected_bytes: int,
                         expected_sha256: str, proxy: str | None=None,
                         workers: int=4, block_bytes: int=4*1024**2) -> dict:
    """Bounded small-range fallback for a slow tail; writes a separate staging file.

    Existing download writers must be stopped before the caller promotes this
    staging file. No original data file is modified by this helper.
    """
    import time
    import urllib.request
    if not 1<=workers<=4:raise ValueError('tail workers must be between 1 and 4')
    offset=partial.stat().st_size
    if not 0<=offset<=expected_bytes:raise ValueError('invalid partial size')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('rb') as source,destination.open('wb') as target:
        remaining=offset
        while remaining:
            block=source.read(min(1024**2,remaining))
            if not block:raise ValueError('partial was truncated during snapshot')
            target.write(block);remaining-=len(block)
    ranges=[(start,min(start+block_bytes,expected_bytes)-1) for start in range(offset,expected_bytes,block_bytes)]
    def fetch(bounds):
        start,end=bounds
        for attempt in range(2):
            begin=time.monotonic()
            try:
                request=urllib.request.Request(url+('&' if '?' in url else '?')+f'range_start={start}',
                    headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'})
                with _public_opener(proxy).open(request,timeout=12) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{expected_bytes}':
                        raise ValueError('range response does not match request')
                    chunks=[];received=0
                    while received<end-start+1:
                        if time.monotonic()-begin>60:raise TimeoutError('range time budget exceeded')
                        chunk=response.read(min(64*1024,end-start+1-received))
                        if not chunk:raise OSError('incomplete range')
                        chunks.append(chunk);received+=len(chunk)
                return start,b''.join(chunks)
            except (OSError,ValueError):
                if attempt==1:raise
    failed=[];finished=0
    with destination.open('r+b') as target,ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,bounds):bounds for bounds in ranges}
        for future in as_completed(futures):
            try:
                start,data=future.result();target.seek(start);target.write(data);finished+=1
                print(f'tail block {finished}/{len(ranges)} verified',flush=True)
            except Exception as exc:failed.append({'range':futures[future],'error':type(exc).__name__})
    if failed:
        write_json(destination.with_name(destination.name+'.failures.json'),failed)
        raise ValueError(f'{len(failed)} tail ranges failed')
    if destination.stat().st_size!=expected_bytes or sha256_file(destination)!=expected_sha256:
        raise ValueError('assembled tail SHA256 mismatch')
    return {'url':url,'sha256':expected_sha256,'bytes':expected_bytes,
            'transport':'bounded-range-fallback','reused_prefix_bytes':offset,'range_blocks':len(ranges)}
