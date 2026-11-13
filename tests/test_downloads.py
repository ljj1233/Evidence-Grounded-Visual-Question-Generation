"""Offline regressions for complete public downloads and Drive's first-page limit."""
import gzip
import hashlib
import io
import json
import zipfile

import pytest

from egvqg.dataset import download_file
from egvqg.downloads import MMQA_REVISION, _drive_listing, download_multimodalqa


def test_download_rejects_html_and_recovers_complete_partial_without_http(tmp_path):
    class Response(io.BytesIO):
        status=200
        headers={'Content-Type':'text/html','Content-Length':'6'}
    path=tmp_path/'chunk.001'
    with pytest.raises(ValueError,match='HTML'):
        download_file('https://example.invalid/file',path,max_bytes=10,expected_bytes=6,
                      reject_html=True,opener=lambda *a,**k:Response(b'<html>'))
    assert not path.exists() and not path.with_name(path.name+'.part').exists()
    partial=path.with_name(path.name+'.part');partial.write_bytes(b'abcdef')
    with pytest.raises(ValueError,match='hash mismatch'):
        download_file('https://example.invalid/file',path,max_bytes=10,expected_bytes=6,expected_sha256='0'*64)
    def forbidden(*args,**kwargs):raise AssertionError('complete partial must not issue HTTP 416')
    receipt=download_file('https://example.invalid/file',path,max_bytes=10,expected_bytes=6,
                          expected_sha256=hashlib.sha256(b'abcdef').hexdigest(),opener=forbidden)
    assert path.read_bytes()==b'abcdef' and receipt['bytes']==6 and not partial.exists()
    assert download_file(receipt['url'],path,max_bytes=10,expected_bytes=6,expected_sha256=receipt['sha256'],opener=forbidden)==receipt


def test_drive_listing_includes_51st_volume_from_embedded_view():
    rows=[]
    for i in range(1,51):
        row=[None]*14;row[0]=f'id{i}';row[2]=f'imgs.7z.{i:03d}';row[3]='application/octet-stream';row[13]=1073741824
        rows.append(row)
    payload=json.dumps([rows]).replace('[',r'\x5b').replace(']',r'\x5d').replace('"',r'\x22')
    page="window['_DRIVE_ivd'] = '"+payload+"';"
    embedded='<a href="https://drive.google.com/file/d/id51/view"><div class="flip-entry-title">imgs.7z.051</div></a>'
    class Opener:
        def open(self,url,timeout):return io.BytesIO((embedded if 'embeddedfolderview' in url else page).encode())
    result=_drive_listing('folder',Opener())
    assert len(result)==51 and result['imgs.7z.051']['id']=='id51'
    assert result['imgs.7z.001']['bytes']==1073741824


def test_mmqa_download_validates_full_image_mapping_and_cached_crc(tmp_path,monkeypatch):
    def fixture(root,files,**kwargs):
        root.mkdir(parents=True,exist_ok=True)
        if (root/'final_dataset_images.zip').exists():return {}
        for name in files:
            if name.endswith('.gz'):
                row={'id':'x','path':'x.png'} if name=='MMQA_images.jsonl.gz' else {'id':'x'}
                with gzip.open(root/name,'wt') as stream:stream.write(json.dumps(row)+'\n')
        with zipfile.ZipFile(root/'final_dataset_images.zip','w') as z:z.writestr('final_dataset_images/x.png',b'fixture')
        return {}
    monkeypatch.setattr('egvqg.downloads._download_bundle',fixture)
    result=download_multimodalqa(tmp_path)
    assert result['images_verified']==1
    image=tmp_path/'data/raw/multimodalqa'/MMQA_REVISION/'final_dataset_images/x.png'
    image.write_bytes(b'changed')  # Same length; size-only cache checks would miss this.
    with pytest.raises(ValueError,match='extracted image changed'):download_multimodalqa(tmp_path)


def test_webqa_reference_verification_includes_test_field_and_checks_index_binding(tmp_path):
    from egvqg.downloads import WEBQA_REVISION, verify_webqa_references
    from egvqg.records import sha256_file, write_json
    root=tmp_path/'data/raw/webqa'/WEBQA_REVISION;root.mkdir(parents=True)
    first=b'30000000\tQUJD\n';second=b'30000001\tREVG\n'
    (root/'imgs.tsv').write_bytes(first+second)
    (root/'imgs.lineidx').write_text(f'0\n{len(first)}\n')
    write_json(root/'train.json',{'q':{'img_posFacts':[{'image_id':30000000}]}})
    write_json(root/'test.json',{'q2':{'img_Facts':[{'image_id':30000001}]}})
    write_json(root/'manifest.json',{'status':'complete','image_store':{'path':'imgs.tsv','bytes':len(first+second)},
              'annotation_files':{n:{'sha256':sha256_file(root/n)} for n in ('train.json','test.json')}})
    result=verify_webqa_references(tmp_path)
    assert result['all_reference_ids_verified']==2 and result['per_file']['test.json']==1
    (root/'imgs.tsv').write_bytes(second+first)
    with pytest.raises(ValueError,match='ID/offset mismatch'):verify_webqa_references(tmp_path)


def test_tail_ranges_reuse_prefix_and_verify_full_hash(tmp_path,monkeypatch):
    from egvqg.downloads import finish_download_tail
    import re
    content=b'abcdefghij';partial=tmp_path/'original.part';partial.write_bytes(content[:3])
    class Response(io.BytesIO):
        status=206
        def __init__(self,start,end):
            super().__init__(content[start:end+1])
            self.headers={'Content-Range':f'bytes {start}-{end}/{len(content)}'}
    class Opener:
        def open(self,request,timeout):
            match=re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range'))
            return Response(int(match[1]),int(match[2]))
    monkeypatch.setattr('egvqg.downloads._public_opener',lambda proxy=None:Opener())
    dest=tmp_path/'staged'
    result=finish_download_tail('https://example.invalid/data',partial,dest,expected_bytes=len(content),
                                expected_sha256=hashlib.sha256(content).hexdigest(),workers=2,block_bytes=2)
    assert result['reused_prefix_bytes']==3 and dest.read_bytes()==content
    assert partial.read_bytes()==content[:3]
