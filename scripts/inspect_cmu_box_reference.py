"""Inspect a CMU action label before treating it as an elevated-platform jump."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'datasets/cmu_75_12'
def load():
 text=(FOLDER/'75.asf').read_text();bones={};block={};inside=False
 for line in text.split(':bonedata')[1].split(':hierarchy')[0].splitlines():
  x=line.split()
  if not x:continue
  if x[0]=='begin':block={};inside=True
  elif x[0]=='end':bones[block['name']]=block;inside=False
  elif inside:
   if x[0]=='name':block['name']=x[1]
   elif x[0]in ['direction','axis']:block[x[0]]=np.array(list(map(float,x[1:4])))
   elif x[0]=='length':block['length']=float(x[1])
   elif x[0]=='dof':block['dof']=x[1:]
 hierarchy={}
 for line in text.split(':hierarchy')[1].splitlines():
  x=line.split()
  if x and x[0]not in ['begin','end']:hierarchy[x[0]]=x[1:]
 frames=[];current=None
 for line in (FOLDER/'75_12.amc').read_text().splitlines():
  x=line.split()
  if not x or line.startswith(('#',':')):continue
  if x[0].isdigit():
   if current is not None:frames.append(current)
   current={}
  else:current[x[0]]=np.array(list(map(float,x[1:])))
 if current is not None:frames.append(current)
 names=['root']+list(bones);ids={n:i for i,n in enumerate(names)};parents=np.full(len(names),-1)
 for parent,children in hierarchy.items():
  for child in children:parents[ids[child]]=ids[parent]
 scale=.0254/.45;C=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
 positions=[]
 for frame in frames:
  pos={'root':frame['root'][:3]*scale};rots={'root':Rotation.from_euler('xyz',frame['root'][3:],degrees=True).as_matrix()}
  def visit(parent):
   for name in hierarchy.get(parent,[]):
    bone=bones[name];axis=Rotation.from_euler('xyz',bone['axis'],degrees=True).as_matrix();angles=np.zeros(3)
    for dof,value in zip(bone.get('dof',[]),frame.get(name,[])):angles['xyz'.index(dof[-1])]=value
    rot=axis@Rotation.from_euler('xyz',angles,degrees=True).as_matrix()@axis.T;rots[name]=rots[parent]@rot
    pos[name]=pos[parent]+rots[name]@bone['direction']*bone['length']*scale;visit(name)
  visit('root');positions.append(np.array([C@pos[n]for n in names]))
 return np.arange(len(frames))/60,np.array(positions),names,parents
def main():
 t,P,names,parents=load();foot=P[:,[names.index(n)for n in ['lfoot','ltoes','rfoot','rtoes']],2].min(1)
 np.savez_compressed(FOLDER/'source_joint_trajectories.npz',times=t,joint_positions_m=P,joint_names=names,parent_indices=parents)
 fig=plt.figure(figsize=(15,8),layout='constrained');gs=fig.add_gridspec(2,6)
 lo=P.min((0,1));hi=P.max((0,1));indices=np.linspace(0,len(t)-1,12).astype(int)
 for k,i in enumerate(indices):
  ax=fig.add_subplot(gs[k//6,k%6],projection='3d')
  for j,parent in enumerate(parents):
   if parent>=0:ax.plot(*P[i,[parent,j]].T,color='#2175a5',lw=2)
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=f'{t[i]:.2f}s');ax.set_box_aspect(np.maximum(hi-lo,.2));ax.view_init(15,-65);ax.set_axis_off()
 fig.savefig(FOLDER/'source_motion_contact_sheet.png',dpi=110);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');axs[0].plot(t,P[:,0,2],label='Pelvis');axs[0].plot(t,foot,label='Lowest foot endpoint');axs[0].legend();axs[0].set(xlabel='Time [s]',ylabel='Height [m]')
 axs[1].plot(P[:,0,0],P[:,0,1]);axs[1].set(xlabel='World X [m]',ylabel='World Y [m]',title='Pelvis horizontal path');axs[1].axis('equal');fig.savefig(FOLDER/'source_trajectory_check.png',dpi=130);plt.close(fig)
 report=dict(source='CMU Graphics Lab motion capture 75_12',catalog_label='box jump',source_page='http://mocap.cs.cmu.edu/search.php?subjectnumber=75',fps=60,frames=len(t),duration_s=float(t[-1]),source_units='ASF length 0.45, inch conversion / 0.45',world_conversion='CMU Y-up to right-handed Z-up',source_SHA256={f:hashlib.sha256((FOLDER/f).read_bytes()).hexdigest()for f in ['75.asf','75_12.amc']},classification='pending_visual_review_elevated_box_not_assumed',force_data_available=False,force_analysis_allowed=False)
 (FOLDER/'source_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
