"""Verify actual evaluated Blender surfaces, not only intended coordinates."""
from pathlib import Path
import json,sys,bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from bar_geometry import shaft_gap
OUT=ROOT/'results/grip_comparison';reports=[]
for grip in [65,80]:
 case=OUT/f'grip_{grip}/cycle/physiology';motion=np.load(case/'fullbody_motion.npz');meta=json.loads((case/'fullbody_metadata.json').read_text())
 bpy.ops.wm.open_mainfile(filepath=str(case/'fullbody_scene.blend'));scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
 bar=bpy.data.objects['100kg barbell'];origin=np.array(bar.matrix_world.translation);surface={}
 for ob in scene.objects:
  if ob.type!='MESH' or ob.hide_render or not any(k in ob.name.lower() for k in ['pectoralis major','deltoid','bone clavicle']):continue
  ev=ob.evaluated_get(deps);v=np.array([list(ev.matrix_world@p.co) for p in ev.data.vertices]);ev.data.calc_loop_triangles()
  surface[ob.name]=shaft_gap(v,[list(p.vertices) for p in ev.data.loop_triangles],origin)
 contact=min(value for name,value in surface.items() if 'pectoralis' in name.lower() or 'clavicle' in name.lower())
 assert -.0001<contact<.003,('Rack gap outside tolerance',grip,contact)
 assert min(surface.values())>-.0001,('Shaft intersects shoulder/chest surface',grip,surface)
 skin=bpy.data.objects['Skin'];head=[]
 for vertex in skin.data.vertices:
  head.append(sum(g.weight for g in vertex.groups if skin.vertex_groups[g.group].name=='head')>.35)
 ev=skin.evaluated_get(deps);v=np.array([list(ev.matrix_world@p.co) for p in ev.data.vertices]);ev.data.calc_loop_triangles()
 faces=[list(p.vertices) for p in ev.data.loop_triangles if all(head[i] for i in p.vertices)]
 head_index=meta['body_names'].index('head');assert abs(motion['body_xpos'][:,head_index]-motion['body_xpos'][0,head_index]).max()<1e-8
 assert abs(motion['body_xmat'][:,head_index]-motion['body_xmat'][0,head_index]).max()<1e-8
 head_gaps=[shaft_gap(v,faces,p) for p in motion['physical_grips'].mean(1)]
 assert min(head_gaps)>.002,('Face clearance failed',grip,min(head_gaps))
 assert abs(motion['physical_grips'][0]-motion['physical_grips'][-1]).max()<1e-8
 elbow=motion['physical_elbow_centres'][0];hands=motion['physical_grips'][0]
 assert np.all(elbow[:,2]<hands[:,2]-.1)
 assert np.all(elbow[:,1]<hands[:,1]+.01)
 report={'status':'passed','grip_cm':grip,'actual_start_surface_gap_m':contact,'minimum_head_clearance_m':min(head_gaps),
  'start_surface_gaps_m':surface,'bar_start_world_m':origin.tolist(),'head_gaps_per_frame_m':head_gaps,
  'elbow_flexion_max_deg':meta['elbow_flexion_max_deg'],
  'limits':'Geometric tangency and clearance, not deformable tissue contact. Bar weight remains transmitted through both hands.'}
 (case/'bar_geometry_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report)
(OUT/'bar_geometry_validation.json').write_text(json.dumps({'status':'passed','cases':reports},indent=2),encoding='utf-8')
print('RACK_CONTACT_AND_FACE_CLEARANCE_PASSED',[(r['grip_cm'],r['actual_start_surface_gap_m'],r['minimum_head_clearance_m']) for r in reports],flush=True)
