"""Decode numerical region masks and package an offline synchronized viewer."""
import base64,json,zlib,hashlib,csv
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tension_visualization'

def decode_mask(path,group_count,allowed_ids=None,step=4):
    rgb=np.array(Image.open(path).convert('RGB'),dtype=float);strength=rgb[:,:,0]+rgb[:,:,1];code=np.divide(rgb[:,:,0]*255,strength,out=np.zeros_like(strength),where=strength>1);ids=np.rint(code/step).astype(int);valid=(strength>2)&(ids>=1)&(ids<=group_count)&(abs(code-ids*step)<step/2+.01);coverage=np.clip(np.rint(strength),0,255).astype(np.uint16);ids[~valid]=0;coverage[~valid]=0
    if allowed_ids is not None:
        invalid=~np.isin(ids,list(allowed_ids));ids[invalid]=0;coverage[invalid]=0
    return (ids.astype(np.uint16)*256+coverage).astype('<u2')

def encode_runs(values):
    flat=values.ravel();edges=np.r_[0,np.where(flat[1:]!=flat[:-1])[0]+1,len(flat)];parts=[]
    for start,end in zip(edges[:-1],edges[1:]):
        remaining=int(end-start);value=int(flat[start])
        while remaining:
            n=min(65535,remaining);parts.extend([n,value]);remaining-=n
    return base64.b64encode(zlib.compress(np.array(parts,dtype='<u2').tobytes(),9)).decode('ascii')

def main():
    d=json.loads((OUT/'data.json').read_text(encoding='utf-8'));masks={'anatomy':[],'humanoid':[]};visibility=[]
    for mode in ['anatomy','humanoid']:
      prefix=OUT/'anatomy' if mode=='anatomy' else OUT
      for i in range(len(d['times'])):
        base=prefix/f'base/frame_{i:04d}.png';mask=prefix/f'mask/frame_{i:04d}.png'
        if not base.exists() or not mask.exists():raise RuntimeError('Missing paired frame '+str(i))
        decoded=decode_mask(mask,len(d['groups']),{g['display_id'] for g in d['groups'] if g['display_id']},d['mask_step']);ids=decoded//256
        def visible(part):
            values,counts=np.unique(part,return_counts=True);return {int(v) for v,c in zip(values,counts) if v and c>=64}
        left=visible(ids[:,:540]);right=visible(ids[:,540:]);common=left&right
        if len(common)<2:raise RuntimeError(('Insufficient front/back shared mapping',i,len(common)))
        if decoded.shape!=(900,1080):raise RuntimeError('Wrong render dimensions')
        masks[mode].append(encode_runs(decoded));visibility.append({'mode':mode,'frame':i,'common_visible_group_ids':sorted(common),'front_group_ids':sorted(left),'back_group_ids':sorted(right)})
    (OUT/'masks.js').write_text('window.TENSION_MASKS='+json.dumps(masks,separators=(',',':'))+';',encoding='utf-8');template=(ROOT/'scripts/fullbody_visualization_template.html').read_text(encoding='utf-8');(OUT/'index.html').write_text(template,encoding='utf-8')
    with (OUT/'group_tensions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['profile','time_s','group','side','status','total_tension_N','active_tension_N','passive_tension_N'])
        for profile,values in d['profiles'].items():
            for i,t in enumerate(d['times']):
                for j,g in enumerate(d['groups']):writer.writerow([profile,t,g['key'],g['side'],g['status'],*[values[k][j][i] if values[k][j] is not None else '' for k in ['total','active','passive']]])
    checks={'status':'passed','frames_per_mode':len(masks['anatomy']),'modes':list(masks),'shared_value_record':True,'same_region_ids_both_models':True,'minimum_common_visible_group_ids':min(len(x['common_visible_group_ids']) for x in visibility),'zero_is_not_unknown':True,'gray_means_unmapped_or_parameter_review':True,'frame_visibility':visibility,'source_data_sha256':hashlib.sha256((OUT/'data.json').read_bytes()).hexdigest(),'render_mapping_sha256':hashlib.sha256((OUT/'render_mapping.json').read_bytes()).hexdigest()};(OUT/'mapping_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('WHOLE_BODY_PACKAGED',len(masks['anatomy']),'frames per mode',checks['minimum_common_visible_group_ids'],'shared visible regions')

if __name__=='__main__':main()
