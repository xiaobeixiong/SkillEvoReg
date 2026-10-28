"""Fetch and verify the public tokenizer vocabulary for token-budget checks."""
import hashlib
import os
from pathlib import Path
import urllib.request

URL='https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
EXPECTED='446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'
ROOT=Path(__file__).resolve().parents[1]

def main():
    cache=Path(os.environ.get('TIKTOKEN_CACHE_DIR',str(ROOT/'.cache/tiktoken')))
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/hashlib.sha1(URL.encode()).hexdigest()
    data=path.read_bytes() if path.exists() else urllib.request.urlopen(URL,timeout=45).read()
    if hashlib.sha256(data).hexdigest()!=EXPECTED:
        raise RuntimeError('Tokenizer vocabulary checksum mismatch')
    if not path.exists():
        path.write_bytes(data)
    print('Verified o200k_base vocabulary:',path)

if __name__=='__main__':main()
