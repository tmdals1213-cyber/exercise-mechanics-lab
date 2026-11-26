"""Fetch a bounded, provenance-tracked subset of Calisti et al. 2025.

CC BY 4.0 source; subject selected by control/male/near-reference body size,
not by an outcome. The archive is read via HTTP ranges, not downloaded whole.
"""
import io,json,hashlib,urllib.request,zipfile
from pathlib import Path
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/cmj_calisti2025'
API='https://api.figshare.com/v2/articles/28890545'

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ExerciseMechanicsResearch/1.0'}),timeout=40) as r:return r.read()

class RemoteFile(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        if n>64*1024*1024:raise ValueError('Unexpectedly large single read')
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','User-Agent':'ExerciseMechanicsResearch/1.0'})
        with urllib.request.urlopen(req,timeout=40) as r:
            if r.status!=206:raise RuntimeError('Server ignored Range; refusing whole-archive download')
            data=r.read(n+1)
        if len(data)!=n:raise RuntimeError(('Range length mismatch',n,len(data)))
        self.pos+=n;self.transferred+=n;return data

def main():
    source=OUT/'source';source.mkdir(parents=True,exist_ok=True);meta=json.loads(get(API));(source/'figshare_metadata.json').write_text(json.dumps(meta,indent=2));manifest=[]
    small=['participant_log.xlsx','labeling_CMJ.xlsx','catelli_3dof_model.osim','setup_IK_default.xml','setup_scaling_default.xml']
    for name in small:
        f=next(x for x in meta['files'] if x['name']==name);path=source/name
        if not path.exists() or path.stat().st_size!=f['size']:path.write_bytes(get(f['download_url']))
        assert hashlib.md5(path.read_bytes()).hexdigest()==f['computed_md5'],name
        manifest.append(dict(file=name,url=f['download_url'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_md5=f['computed_md5']))
    rows=list(openpyxl.load_workbook(source/'participant_log.xlsx',read_only=True,data_only=True).active.values);r=next(r for r in rows if r[0]=='sub10');assert r[1]==2 and r[2]==2
    label=list(openpyxl.load_workbook(source/'labeling_CMJ.xlsx',read_only=True,data_only=True).active.values);valid=[r for r in label[1:] if r[1]==10 and r[4]==0];assert len(valid)==3 and all(r[2]==1 and r[8]==0 for r in valid)
    subject=dict(id='sub10',sex='male',group='healthy control',height_m=float(r[4]),mass_kg=float(r[5]),selection='First control male closest to 180 cm / 90 kg reference (180 cm / 88 kg); no outcome-based selection.',metadata_note='participant_log encodes controls as 2; labeling_CMJ encodes controls as 1. Both checked explicitly.',trials=['s10_CMJ_t1','s10_CMJ_t2','s10_CMJ_t3'],fatigued=False)
    # Do not reuse the comprehension variable r (Python comprehension is scoped).
    (OUT/'subject.json').write_text(json.dumps(subject,indent=2))
    f=next(x for x in meta['files'] if x['name']=='Kinematic_data.zip');remote=RemoteFile(f['download_url'],f['size']);prefix='Kinematic_data/Raw_c3d_files/sub10/'
    with zipfile.ZipFile(remote) as archive:
        for name in ['static.c3d']+[t+'.c3d' for t in subject['trials']]:
            entry=archive.getinfo(prefix+name);path=source/name
            if not path.exists() or path.stat().st_size!=entry.file_size:path.write_bytes(archive.read(entry))
            manifest.append(dict(file=name,archive=f['download_url'],archive_entry=entry.filename,archive_offset=entry.header_offset,uncompressed_size=entry.file_size,crc32=f'{entry.CRC:08x}',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('FETCHED_TRIAL',name,path.stat().st_size,flush=True)
    provenance=dict(schema='exercise-mechanics.measured-cmj/1',authors='Calisti, Mohr and Federolf',paper='https://doi.org/10.1038/s41597-025-05934-5',dataset_doi=meta['doi'],dataset_version=meta['version'],license=meta['license'],source_files=manifest,archive_bytes_transferred_this_run=remote.transferred,source_data_modified=False,scope='Quiet static trial and three non-fatigued bilateral CMJs; no tendon forces measured.',metadata_group_code_discrepancy_handled=True)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));print('CMJ_SOURCE_READY',json.dumps(subject),flush=True)

if __name__=='__main__':main()
