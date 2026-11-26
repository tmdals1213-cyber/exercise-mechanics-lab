"""Shared force-backed group/log display. Numerical force records are unchanged."""
import json,numpy as np
MUSCLE_UPPER_N=30000.;EXPOSURE_UPPER_NS=30000.;JOINT_UPPER_N=50000.
COLORS=np.array([[43,97,142],[48,163,169],[240,183,78],[238,82,69]])
def rgb(v,upper=MUSCLE_UPPER_N):
 x=np.log1p(np.maximum(v,0));stops=np.log1p([0,100,1000,upper]);return np.stack([np.interp(x,stops,COLORS[:,i]) for i in range(3)],axis=-1).astype(np.uint8)
def joint_rgb(v):
 u=np.clip(np.asarray(v)/JOINT_UPPER_N,0,1);return np.stack([8+237*u,8+22*u,10+20*u],axis=-1).astype(np.uint8)
def setup(folder,anatomy,skin,bones,scale):
 import case_tension_render as base
 base.rgb=rgb;base.joint_rgb=joint_rgb;fn=base.setup(folder,anatomy,skin,bones,scale)
 p=folder/'tension_render_mapping.json';a=json.loads(p.read_text());a.update(palette='Common absolute per-side muscle-group scalar force, logarithmic. Separate joint resultant, linear black/red.',palette_upper_N=MUSCLE_UPPER_N,joint_upper_N=JOINT_UPPER_N,forces_changed=False);p.write_text(json.dumps(a,indent=2));return fn
def setup_exposure(folder,anatomy,skin):
 import bpy
 from atlas_mapping import map_atlas
 from case_tension_render import linear
 v=json.loads((folder/'analysis/visualization/data.json').read_text());e=json.loads((folder/'analysis/exposure.json').read_text());E=np.array(e['element_integral_Ns']);groups=v['groups'];audit=[]
 for ob in anatomy:
  if ob==skin or not any(x.type=='ARMATURE' for x in ob.modifiers):continue
  ids=map_atlas(ob.name,v['names']);gi=next((i for i,g in enumerate(groups) if ids and set(ids).issubset(g['indices']) and g.get('released',True)),None)
  if gi is None:continue
  val=float(E[groups[gi]['indices']].sum());ma=bpy.data.materials.new('Group exposure '+ob.name);ma.use_nodes=True;bs=ma.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*linear(rgb(val,EXPOSURE_UPPER_NS)),1);bs.inputs['Roughness'].default_value=.6;ob.data.materials.clear();ob.data.materials.append(ma);audit.append(dict(mesh=ob.name,group=groups[gi]['key'],side=groups[gi]['side'],integral_Ns=val))
 (folder/'exposure_render_mapping.json').write_text(json.dumps(dict(common_upper_Ns=EXPOSURE_UPPER_NS,metric='One-cycle scalar muscle-group force-time integral; logarithmic display',meshes=audit,forces_changed=False),indent=2));return lambda t:None
def setup_joint_peak(folder,anatomy,skin):
 import bpy
 from case_tension_render import linear
 v=json.loads((folder/'analysis/visualization/data.json').read_text());e=json.loads((folder/'analysis/exposure.json').read_text());t=np.array(v['times']);J=np.array(v['joint_force_N']);mask=(t>=e['start_s'])&(t<=e['end_s']);peak=J[:,mask].max(1)
 for ob in anatomy:
  if ob!=skin and any(x.type=='ARMATURE' for x in ob.modifiers):ob.hide_render=True
 for j,item in enumerate(v['joints']):
  ma=bpy.data.materials.get('Joint reaction '+item['key']);assert ma is not None;ma.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(*linear(joint_rgb(peak[j])),1)
 (folder/'joint_peak_render_values.json').write_text(json.dumps(dict(metric='Per-joint peak resultant magnitude across the same active window; maxima occur at different times.',window_s=[e['start_s'],e['end_s']],joint_upper_N=JOINT_UPPER_N,values_N={g['key']:float(x) for g,x in zip(v['joints'],peak)},force_data_unchanged=True),indent=2));return lambda t:None
