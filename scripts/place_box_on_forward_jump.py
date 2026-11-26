"""Place an apparatus on an already-audited jump; never change its mechanics.

Geometry stops at first contact. A sole envelope plus sampled lower-limb
capsules are checked; no landing reaction, stability, or full skin collision is
claimed. Box height is an output of this placement search.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def sole_envelope(points):
    strips=[]
    for offset in [0,6]:
        for first in [0,2]:
            for u in np.linspace(0,1,6):
                left=(1-u)*points[:,offset+first]+u*points[:,offset+first+2]
                right=(1-u)*points[:,offset+first+1]+u*points[:,offset+first+3]
                for v in [0,.5,1]:strips.append((1-v)*left+v*right)
    return np.stack(strips,axis=1)

def search(data,setup,requested_height=None):
    air=data['mode']==2;t=data['times'][air];raw=data['material_contact_positions_m'][air];p=sole_envelope(raw)
    initial=data['material_contact_positions_m'][0];initial_front=float(initial[:,1].min());cx=float(initial[:,0].mean())
    names=setup['body_names'];body=data['body_xpos'][air];limbs=[];radii=[]
    for side in ['r','l']:
        for a,b,radius in [('femur','tibia',.055),('tibia','talus',.035)]:
            A=body[:,names.index(a+'_'+side)];B=body[:,names.index(b+'_'+side)]
            for u in np.linspace(0,1,18):limbs.append((1-u)*A+u*B);radii.append(radius)
    limb=np.stack(limbs,axis=1);radii=np.array(radii);results=[];forefoot_results=[];best_first=None
    maximum=float(p[:,:,2].min(1).max());heights=[requested_height]if requested_height is not None else np.arange(np.floor(maximum*100)/100,.09,-.01)
    for near in initial_front-np.arange(.03,.341,.02):
        bounds=np.array([[cx-.4,near-.65,0.],[cx+.4,near,0.]])
        for height in heights:
            if height<=0:continue
            dz=p[:-1,:,2]-p[1:,:,2];crossing=(p[:-1,:,2]>height)&(p[1:,:,2]<=height)&(dz>1e-9)
            ii,jj=np.where(crossing)
            if not len(ii):continue
            u=(p[ii,jj,2]-height)/dz[ii,jj];hit=p[ii,jj]+u[:,None]*(p[ii+1,jj]-p[ii,jj])
            inside=np.all((hit[:,:2]>=bounds[0,:2])&(hit[:,:2]<=bounds[1,:2]),axis=1)
            if not inside.any():continue
            ii,jj,u,hit=ii[inside],jj[inside],u[inside],hit[inside];contact_t=t[ii]+u*(t[ii+1]-t[ii]);pick=int(contact_t.argmin())
            k=int(ii[pick]);fraction=float(u[pick]);ct=float(contact_t[pick]);before=np.arange(len(t))<=k
            # A point entering the front face below the top is a side collision,
            # not a successful top contact. Check segment intersections in time.
            dy=p[:-1,:,1]-p[1:,:,1];edge=(p[:-1,:,1]>near)&(p[1:,:,1]<=near)&(dy>1e-9)
            ei,ej=np.where(edge);eu=(p[ei,ej,1]-near)/dy[ei,ej]
            ep=p[ei,ej]+eu[:,None]*(p[ei+1,ej]-p[ei,ej]);et=t[ei]+eu*(t[ei+1]-t[ei])
            side=(et<ct-1e-6)&(ep[:,0]>=bounds[0,0])&(ep[:,0]<=bounds[1,0])&(ep[:,2]<height-.0001)
            if side.any():continue
            # Discrete sole envelope occupancy supplements the swept front edge.
            q=p[before];inside_xy=np.all((q[:,:,:2]>=bounds[0,:2])&(q[:,:,:2]<=bounds[1,:2]),axis=2)
            if np.any(inside_xy&(q[:,:,2]<height-.0002)):continue
            # Sampled capsules are a declared geometric envelope, not tissue FEM.
            bounds[1,2]=height;lp=limb[before];dist=np.linalg.norm(np.maximum(np.maximum(bounds[0]-lp,lp-bounds[1]),0),axis=2)
            if np.any(dist<radii[None]-.001):continue
            at_contact=raw[k]+fraction*(raw[k+1]-raw[k]);whole=bool(np.all((at_contact[:,:2]>=bounds[0,:2]+.003)&(at_contact[:,:2]<=bounds[1,:2]-.003)))
            fi=[2,3,4,5,8,9,10,11];front=at_contact[fi]
            fore_inside=bool(np.all((front[:,:2]>=bounds[0,:2]+.003)&(front[:,:2]<=bounds[1,:2]-.003)))
            gaps=[float(at_contact[j:j+4,2].min()-height)for j in [2,8]]
            descending=all(float((raw[k+1,j:j+4,2]-raw[k,j:j+4,2]).mean())<0 for j in [2,8])
            record=dict(height_m=float(height),near_face_y_m=float(near),far_face_y_m=float(near-.65),x_bounds_m=bounds[:,0].tolist(),first_contact_s=ct,
                contact_sole_world_m=hit[pick].tolist(),both_foot_footprints_inside=whole,
                lower_limb_capsule_min_clearance_m=float((dist-radii[None]).min()),first_contact_sole_point_index=int(jj[pick]))
            record.update(both_forefoot_footprints_inside=fore_inside,both_forefeet_height_gaps_m=gaps)
            if fore_inside and descending and max(gaps)<.03 and min(gaps)>-.002:forefoot_results.append(record)
            if best_first is None or record['height_m']>best_first['height_m']:best_first=record
            if whole:results.append(record);break
    best=max(results,key=lambda x:(x['height_m'],x['lower_limb_capsule_min_clearance_m']),default=None)
    best_forefoot=max(forefoot_results,key=lambda x:(x['height_m'],x['lower_limb_capsule_min_clearance_m']),default=None)
    return dict(status='precontact_geometry_demonstrated'if best else'no_valid_precontact_placement_in_search',best_complete_footprint_placement=best,
        best_first_top_contact_placement=best_first,maximum_minimum_sole_height_m=maximum,
        best_bilateral_forefoot_placement=best_forefoot,
        forefoot_interpretation='Both forefoot projections fit on top and descend to within 3 cm of first contact; heel overhang is possible. This supplementary result does not satisfy the complete-footprint criterion or demonstrate a stable landing.',
        requested_height_m=requested_height,search_near_faces_m=(initial_front-np.arange(.03,.341,.02)).tolist(),
        box_width_m=.8,box_depth_m=.65,height_resolution_m=.01,sole_envelope_points=p.shape[1],
        lower_limb_capsule_samples=limb.shape[1],landing_force_computed=False,stable_landing_verified=False,full_surface_mesh_collision=False,
        interpretation='Geometry before first contact only; body dynamics and forces are unchanged by the apparatus search.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--height-m',type=float);a=ap.parse_args();p=ROOT/a.folder
    audit=json.loads((p/'verification.json').read_text());assert all(audit['gates'].values()),'Rejected dynamics cannot produce a released box comparison'
    z=np.load(p/'motion.npz');s=json.loads((p/'setup.json').read_text());report=search(z,s,a.height_m)
    report.update(source_motion_sha256=hashlib.sha256((p/'motion.npz').read_bytes()).hexdigest(),mechanics_status=audit['status'],method='Solve jump first, place box afterwards; no fixed foot landing target.')
    filename='box_placement.json'if a.height_m is None else f'box_placement_{a.height_m*100:.0f}cm.json';(p/filename).write_text(json.dumps(report,indent=2))
    best=report['best_complete_footprint_placement'];air=z['mode']==2;tt=z['times'];C=z['com'];minsole=z['material_contact_positions_m'][:,:,2].min(1)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10});fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ax[0].plot(-C[air,1],C[air,2],label='Whole-body COM',lw=2)
    for j in [0,4,6,10]:ax[0].plot(-z['material_contact_positions_m'][air,j,1],z['material_contact_positions_m'][air,j,2],alpha=.6)
    if best:
        from matplotlib.patches import Rectangle
        ax[0].add_patch(Rectangle((-best['near_face_y_m'],0),.65,best['height_m'],fc='#8095a8',alpha=.3,ec='#425568'))
        ax[0].set_title(f'Post-hoc box: {100*best["height_m"]:.0f} cm / precontact only')
    else:ax[0].set_title('No valid box placement demonstrated')
    ax[0].set(xlabel='Forward position [m]',ylabel='Height above floor [m]');ax[0].set_aspect('equal',adjustable='datalim');ax[0].legend();ax[0].grid(alpha=.2)
    ax[1].plot(tt,C[:,2],label='COM height');ax[1].plot(tt,minsole,label='Lowest sole-envelope station')
    ax[1].axvline(s['takeoff_s'],ls='--',c='grey',label='Takeoff')
    if best:ax[1].axhline(best['height_m'],ls=':',c='#b17739',label='Placed box top')
    ax[1].set(xlabel='Physical time [s]',ylabel='Height [m]',title='No box force acts during free flight');ax[1].legend();ax[1].grid(alpha=.2)
    fig.savefig(p/filename.replace('.json','.png'),dpi=160);plt.close(fig)
    print('POSTHOC_BOX',json.dumps(report),flush=True)

if __name__=='__main__':main()
