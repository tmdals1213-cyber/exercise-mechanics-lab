"""Transport the curated Git index to connected GitHub tools in bounded chunks."""
from pathlib import Path
import json,subprocess,base64,hashlib
from jump_monitor_data import ROOT,atomic

def main():
    source=ROOT/'github_release';out=ROOT/'deliverables/github_release/tree_transport';out.mkdir(parents=True,exist_ok=True);paths=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode('utf-8').split('\0');texts=[];binary=[]
    for rel in paths:
        if not rel:continue
        p=source/rel;data=p.read_bytes()
        try:content=data.decode('utf-8')
        except UnicodeDecodeError:
            assert p.suffix.lower()in ['.png','.jpg','.jpeg'],rel;binary.append(dict(path=rel,content=base64.b64encode(data).decode('ascii'),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)));continue
        texts.append(dict(path=rel,mode='100644',type='blob',content=content))
    payload=json.dumps(dict(text=texts,binary=binary),ensure_ascii=True,separators=(',',':'));size=24000;chunks=[]
    for i,start in enumerate(range(0,len(payload),size)):
        part=payload[start:start+size];name=f'part_{i:04d}.txt';(out/name).write_text(part,encoding='ascii');chunks.append(dict(name=name,characters=len(part),sha256=hashlib.sha256(part.encode('ascii')).hexdigest()))
    atomic(out/'manifest.json',dict(text_files=len(texts),binary_files=len(binary),characters=len(payload),payload_sha256=hashlib.sha256(payload.encode('ascii')).hexdigest(),chunks=chunks));print('GITHUB_TREE_TRANSPORT_READY',len(texts),len(binary),len(chunks),len(payload),flush=True)

if __name__=='__main__':main()
