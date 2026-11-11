"""Copy recorded model files from the owner's Drive for advisor review.
The output directory can be shared separately; model bytes are not added to GitHub.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
OLD_ROOT=Path('/content/drive/MyDrive/BUSI_EXPERIMENTS')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def export(inventory,experiments_root,destination):
    records=json.loads(inventory.read_text())['records'];destination.mkdir(parents=True,exist_ok=True);result=[];copied={}
    for r in records:
        relative=Path(r['path']).relative_to(OLD_ROOT)
        if '..' in relative.parts:raise ValueError('Unsafe relative path')
        source=experiments_root/relative;row=dict(r)
        if not source.is_file():row.update(status='MISSING');result.append(row);continue
        actual=sha(source)
        if actual!=r['sha256']:row.update(status='HASH_MISMATCH',observed_sha256=actual);result.append(row);continue
        name=actual+'.keras';target=destination/name
        if target.exists():
            if sha(target)!=actual:raise ValueError('Existing destination differs: '+name)
        else:
            shutil.copy2(source,target)
            if sha(target)!=actual:raise ValueError("Copied checkpoint hash differs: "+name)
        copied[actual]=name;row.update(status='MATCH',exported_file=name);result.append(row)
    payload={'records':result,'expected_records':len(records),'matching_records':sum(r['status']=='MATCH' for r in result),'unique_exported_files':len(copied)}
    (destination/'exported_manifest.json').write_text(json.dumps(payload,indent=2));return payload
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--inventory',type=Path,required=True);a.add_argument('--experiments-root',type=Path,default=OLD_ROOT);a.add_argument('--destination',type=Path,required=True);o=a.parse_args();r=export(o.inventory,o.experiments_root,o.destination)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2));print('Export:',o.destination)
    if r['matching_records']!=r['expected_records']:raise SystemExit('Some source files are missing or differ; inspect exported_manifest.json.')
