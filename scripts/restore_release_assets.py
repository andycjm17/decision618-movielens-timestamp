"""Restore private Release assets to their original project paths, with hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(path,item):
    return path.is_file() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256']

def restore(check_only=False):
    manifest=json.loads((ROOT/'release_assets.json').read_text())
    if not check_only and shutil.which('gh') is None:
        raise RuntimeError('GitHub CLI gh is required; sign in to the account that can access this private repository.')
    failed=[]
    for item in manifest['assets']:
        destination=(ROOT/item['path']).resolve()
        if ROOT not in destination.parents:raise ValueError('Invalid destination outside the repository.')
        if verify(destination,item):
            print('Verified:',item['path']);continue
        if destination.exists():raise RuntimeError('Refusing to overwrite different content: '+str(destination))
        if check_only:
            failed.append(item['path']);continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='release-restore-',dir=destination.parent) as folder:
            subprocess.run(['gh','release','download',manifest['release_tag'],'--repo',manifest['repository'],
                            '--pattern',item['asset'],'--dir',folder],check=True)
            downloaded=Path(folder)/item['asset']
            if not verify(downloaded,item):raise RuntimeError('Asset size/hash mismatch: '+item['asset'])
            shutil.move(str(downloaded),str(destination))
        print('Restored:',item['path'])
    if failed:raise RuntimeError('Missing assets: '+', '.join(failed))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();restore(check_only=args.check_only)
