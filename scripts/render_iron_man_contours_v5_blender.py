"""Neutral anatomical front-oblique / side views of checked squat motions."""
import sys,json,math
from pathlib import Path
import bpy,bmesh
import numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from torso_surfaces import add_torso_surfaces,fit_lower_rectus_depth,fit_lat_insertion
from display_bar_grip import install as install_display_grip


def matrix(p,R):
    out=Matrix.Identity(4);out.translation=Vector(p)
    for i in range(3):
        for j in range(3):out[i][j]=float(R[i,j])
    return out


def material(name,color,metal=0,rough=.5):
    ma=bpy.data.materials.new(name);ma.use_nodes=True;shader=ma.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(*color,1);shader.inputs['Roughness'].default_value=rough;shader.inputs['Metallic'].default_value=metal
    return ma


def audit_grip(skin,front,motion,idx,scale,folder):
    """Audit the displayed skin, not only an assumed wrist reference point."""
    deps=bpy.context.evaluated_depsgraph_get();ev=skin.evaluated_get(deps);mesh=ev.to_mesh()
    world=[ev.matrix_world@v.co for v in mesh.vertices];faces=[tuple(p.vertices) for p in mesh.polygons]
    # Export exact linear-skinning inputs for the numerical grasp fitter.
    weights=np.zeros((len(skin.data.vertices),len(idx)));hand_weight=np.zeros((len(skin.data.vertices),2))
    hand_prefix=('lunate','scaphoid','pisiform','triquetrum','capitate','trapezium','firstmc','proximal_thumb','distal_thumb','trapezoid','hamate','secondmc','2proxph','midph2','distph2','thirdmc','3proxph','midph3','distph3','fourthmc','4proxph','midph4','distph4','fifthmc','5proxph','midph5','distph5')
    for i,v in enumerate(skin.data.vertices):
        for g in v.groups:
            name=skin.vertex_groups[g.group].name;name='sacrum' if name=='pelvis' else name
            if name in idx:weights[i,idx[name]]+=g.weight
            for side_id,side in enumerate(['r','l']):
                if name in [p+'_'+side for p in hand_prefix]:hand_weight[i,side_id]+=g.weight
    chosen_faces=[f for f in faces if min(hand_weight[list(f)].sum(1))>.25]
    chosen=sorted({i for f in chosen_faces for i in f});lookup={v:i for i,v in enumerate(chosen)}
    bind_matrix=front.matrix_world.inverted()@skin.matrix_world
    bind=np.array([tuple(bind_matrix@skin.data.vertices[i].co) for i in chosen])*scale
    rendered=np.array([tuple(front.matrix_world.inverted()@world[i]) for i in chosen])*scale
    np.savez_compressed(folder/'skin_bind.npz',bind_positions_m=bind,weights=weights[chosen],faces=np.array([[lookup[i] for i in f] for f in chosen_faces]),hand_weight=hand_weight[chosen],rendered_positions_m=rendered,reference_qpos=motion['qpos'][0])
    result=[]
    for side in ['r','l']:
        j=idx['lunate_'+side];W=front.matrix_world@matrix(motion['body_xpos'][0,j]/scale,motion['body_xmat'][0,j].reshape(3,3));inv=W.inverted()
        verts=[inv@p for p in world];tree=BVHTree.FromPolygons(verts,faces,all_triangles=False)
        native_hand_R=motion['body_xmat'][0,j].reshape(3,3)
        target=motion['bar_position_m'][0]+(-.48 if side=='r' else .48)*motion['bar_rotation'][0,:,0]
        centre=native_hand_R.T@(target-motion['body_xpos'][0,j])/scale;radius=.014/scale
        axis=np.array(W.to_3x3().inverted()@(front.matrix_world.to_3x3()@Vector(motion['bar_rotation'][0,:,0])))
        axis/=np.linalg.norm(axis)
        def signed_distance(p):
            co,normal,face,dist=tree.find_nearest(Vector(p))
            return float(dist if (Vector(p)-co).dot(normal)>=0 else -dist)
        hand_vertices=np.array([tuple(inv@world[i]) for i in np.flatnonzero(hand_weight[:,0 if side=='r' else 1]>.7)])
        along=(hand_vertices-centre)@axis
        offsets=np.arange(along.min()-.005,along.max()+.005,.0005)
        distances=np.array([signed_distance(centre+a*axis) for a in offsets])
        # Keep candidate centres in the curled-finger cavity neighbourhood.
        candidates=[]
        for y in np.arange(-.106,-.072,.002):
            for z0 in np.arange(-.042,-.002,.002):
                z=z0 if side=='r' else -z0;cc=np.array([0,y,z])
                ds=[signed_distance(cc+a*axis) for a in np.linspace(-.035,.035,29)]
                candidates.append((min(ds),float(np.linalg.norm(cc-centre)),cc.tolist()))
        feasible=[c for c in candidates if c[0]>=radius+.0005]
        recommended=min(feasible,key=lambda z:z[1]) if feasible else max(candidates,key=lambda z:z[0])
        result.append(dict(side=side,bar_radius_m=.014,skin_axis_min_signed_distance_m=float(distances.min()*scale),
                           bar_skin_min_clearance_m=float((distances.min()-radius)*scale),
                           inside_axis_samples=int((distances<0).sum()),proposed_grasp_centre_unscaled_m=centre.tolist(),
                           recommended_grasp_centre_unscaled_m=recommended[2],recommended_axis_clearance_m=float((recommended[0]-radius)*scale),
                           axis_sample_spacing_m=.0005*scale,clearance_lower_bound_with_sampling_m=float((distances.min()-radius-.00025)*scale),
                           condition='Skin nearest-surface signed-distance audit across the full displayed hand, sampled every 0.5 mm along the bar axis. Not a skin deformation or contact-pressure model.'))
    ev.to_mesh_clear();(folder/'grip_surface_audit.json').write_text(json.dumps(result,indent=2))
    print('GRIP_SURFACE_AUDIT',json.dumps(result),flush=True)


def main():
    args=sys.argv[sys.argv.index('--')+1:];kg=int(args[args.index('--kg')+1]);folder=Path(args[args.index('--folder')+1]) if '--folder' in args else ROOT/f'results/power_comparison/squat_{kg}'
    study_view='--study' in args or '--analysis-study' in args
    d=dict(np.load(folder/'motion.npz'));rest=dict(np.load(folder/'reference_pose.npz'));meta=json.loads((folder/'summary.json').read_text());scale=meta['scale'];idx={n:i for i,n in enumerate(meta['body_names'])};is_box=meta.get('movement')=='box_jump_model'
    is_jump=meta.get('movement') in ['cmj_replay','box_jump_reference_replay'];no_bar=is_box or is_jump
    summary_standing='--summary-standing' in args or '--case-exposure' in args or '--joint-peak' in args
    if summary_standing:
        d=dict(np.load(folder/'standing_pose.npz' if '--case-exposure' in args else ROOT/'deliverables/instagram_launch/standing_pose.npz'));no_bar=True
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(ROOT/'assets/fullbody/fullbody_muscles_skin.glb'))
    rig=next(x for x in bpy.data.objects if x.type=='ARMATURE');add_torso_surfaces(ROOT,rig)
    skin=bpy.data.objects['Skin'];keep=[]
    for v in skin.data.vertices:
        w={skin.vertex_groups[g.group].name:g.weight for g in v.groups};p=skin.matrix_world@v.co
        face=w.get('head',0)>.35 and (p.z>1.64 or (p.z>1.58 and p.y<.19))
        ends=sum(v for n,v in w.items() if any(k in n for k in ['ph_','proxph','midph','distph','thumb','firstmc','lunate','calcn','toes','talus']))>.65
        keep.append(face or ends)
    bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free()
    muscle=material('Neutral anatomy',(.62,.69,.73));endsmat=material('Head hands feet',(.43,.49,.54));bone_mat=material('Bone',(.31,.37,.42));shoe_mat=material('Rigid shoe sole',(.045,.065,.085));steel=material('Bar steel',(.26,.32,.38),.75,.25);plate_mat=material('Weight plates',(.075,.14,.18),.2)
    anatomy=[]
    for ob in list(bpy.data.objects):
        if ob.type!='MESH':continue
        if not any(m.type=='ARMATURE' for m in ob.modifiers) or 'platysma' in ob.name.lower():ob.hide_render=True;continue
        if ob!=skin:fit_lower_rectus_depth(ob);fit_lat_insertion(ob,ROOT,ROOT/'results/ohp_fullbody')
        ob.data.materials.clear();ob.data.materials.append(endsmat if ob==skin else muscle)
        for p in ob.data.polygons:p.use_smooth=True
        anatomy.append(ob)
    front=bpy.data.objects.new('Front view',None);bpy.context.collection.objects.link(front)
    for ob in list(bpy.data.objects):
        if ob!=front and ob.parent is None:ob.parent=front
    bones=[];old=json.loads((ROOT/'results/ohp_fullbody/fullbody_metadata.json').read_text())
    for item in json.loads((ROOT/'results/ohp_fullbody/bone_meshes.json').read_text()):
        if item['body'] not in idx or item['body'] in old.get('hand_display_body_names',[]):continue
        mesh=bpy.data.meshes.new(item['name']);mesh.from_pydata(item['vertices'],[],item['faces']);mesh.materials.append(bone_mat);mesh.update()
        ob=bpy.data.objects.new('Bone '+item['name'],mesh);bpy.context.collection.objects.link(ob);ob.parent=front;bones.append((ob,item['body']));anatomy.append(ob)
    update_tensions=None
    if '--tension' in args:
        from group_contour_style_v5 import setup as setup_tension_render
        update_tensions=setup_tension_render(folder,anatomy,skin,bones,scale)
    if '--study' in args and not summary_standing:
        from launch_relative_render import setup as setup_relative_render
        relative_update=setup_relative_render(ROOT,folder,anatomy,skin)
        absolute_update=update_tensions
        def update_tensions(t):
            if absolute_update:absolute_update(t)
            relative_update(t)
    if False and '--analysis-study' in args and '--tension' in args:
        from case_absolute_render import setup as setup_absolute
        anatomical_update=setup_absolute(folder,anatomy,skin)
        group_update=update_tensions
        def update_tensions(t):
            group_update(t);anatomical_update(t)
    if summary_standing:
        if '--joint-peak' in args:
            from group_contour_style_v5 import setup_joint_peak
            update_tensions=setup_joint_peak(folder,anatomy,skin)
        elif '--case-exposure' in args:
            from group_contour_style_v5 import setup_exposure
            update_tensions=setup_exposure(folder,anatomy,skin)
        else:
            from launch_relative_render import setup_exposure
            update_tensions=setup_exposure(ROOT,folder,anatomy,skin)
    if '--conditional-muscles' in args:
        from case_absolute_render import setup as setup_conditional
        update_tensions=setup_conditional(folder,anatomy,skin)
    bar=bpy.data.objects.new('Bar centre',None);bpy.context.collection.objects.link(bar);bar.parent=front
    for kind,x,rad,depth in ([] if no_bar else [('shaft',0,.014,2.2),('plate',-.84,.225,.025+(kg-20)*.001),('plate',.84,.225,.025+(kg-20)*.001)]):
        bpy.ops.mesh.primitive_cylinder_add(vertices=40,radius=rad/scale,depth=depth/scale)
        ob=bpy.context.object;ob['bar_kind']=kind;ob.parent=bar;ob.location=(x/scale,0,0);ob.rotation_euler[1]=math.pi/2;ob.data.materials.append(steel if kind=='shaft' else plate_mat)
        for p in ob.data.polygons:p.use_smooth=True
    if not no_bar and '--native-grip-audit' not in args:
        display_grip=install_display_grip(skin,bar,d,idx,scale,endsmat)
        (folder/'display_grip.json').write_text(json.dumps(display_grip,indent=2))
    # Display the explicitly assumed 25 mm rigid shoe wedge.
    for ids in ([] if no_bar else [range(4),range(4,8)]):
        p=d['contact_positions_m'][0,list(ids)].copy();v=[]
        for z in [0,1]:
            for i in range(4):
                x=p[i].copy();x[2]=-.003 if z==0 else (.052 if i>=2 else .028);v.append((x/scale).tolist())
        mesh=bpy.data.meshes.new('Rigid sole');mesh.from_pydata(v,[],[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]);mesh.materials.append(shoe_mat)
        ob=bpy.data.objects.new('Rigid shoe sole',mesh);bpy.context.collection.objects.link(ob);ob.parent=front
    if (is_box or meta.get('display_platform_only',False)) and not summary_standing:
        geom=meta['box_geometry'];H=geom['top_height_m'];box_mat=material('Box surface',(.16,.24,.29))
        bpy.ops.mesh.primitive_cube_add(size=1);ob=bpy.context.object;ob.parent=front
        xb=geom.get('x_bounds_m',[-.4,.4]);yb=geom['y_bounds_m']
        ob.location=(np.mean(xb)/scale,np.mean(yb)/scale,H/2/scale);ob.scale=((xb[1]-xb[0])/scale,(yb[1]-yb[0])/scale,H/scale);ob.data.materials.append(box_mat)
        bevel=ob.modifiers.new('Rounded box edges','BEVEL');bevel.width=.008/scale;bevel.segments=2
        if '--wide-review' in args or '--outline-platform' in args:
            wire=ob.modifiers.new('Visible motion through platform outline','WIREFRAME');wire.thickness=.008/scale
    if '--floor-review' in args:
        floor_mat=material('Fixed ground reference',(.12,.20,.25))
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5,y_subdivisions=9,size=2)
        ob=bpy.context.object;ob.name='Fixed ground reference';ob.parent=front;ob.location=(0,-.35/scale,-.008/scale);ob.scale=(.6/scale,1.1/scale,1.);ob.data.materials.append(floor_mat)
        wire=ob.modifiers.new('Ground grid lines','WIREFRAME');wire.thickness=.002/scale
    yaw=math.radians(25);right=Vector((math.cos(yaw),math.sin(yaw),0));front.location=-.62*right;front.scale=(scale,)*3
    side=bpy.data.objects.new('Side view',None);bpy.context.collection.objects.link(side)
    pivot=Vector(d['body_xpos'][0,idx['pelvis']]);pivot.z=0
    side.matrix_world=Matrix.Translation(.85*right+pivot)@Matrix.Rotation(math.radians(115),4,'Z')@Matrix.Translation(-pivot)@Matrix.Scale(scale,4)
    rig2=rig.copy();rig2.data=rig.data.copy();bpy.context.collection.objects.link(rig2);rig2.parent=side;rig2.matrix_basis=rig.matrix_basis.copy()
    copies=[]
    for ob in list(bpy.data.objects):
        if ob in [front,side,rig,rig2] or ob.type!='MESH' or ob.hide_render:continue
        cp=ob.copy();cp.name=ob.name+' side';bpy.context.collection.objects.link(cp)
        cp.parent=rig2 if ob.parent==rig else side
        for mod in cp.modifiers:
            if mod.type=='ARMATURE':mod.object=rig2
        if ob.parent==bar:
            cp.parent=side
            if ob.get('bar_kind')=='plate':cp.hide_render=True
        copies.append((ob,cp))
    ref={n:matrix(rest['body_xpos'][j]/scale,rest['body_xmat'][j].reshape(3,3)) for n,j in idx.items()};br={pb.name:pb.matrix_local.copy() for pb in rig.data.bones}
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=16
    scene.render.resolution_x=1080;scene.render.resolution_y=900;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.film_transparent=True
    scene.world=bpy.data.worlds.new('Studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.025,.035,.05,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    def aim(ob,target):ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()
    for loc,power,size in [((3,-4,5),900,4),((-3,-1,3),650,3),((0,4,4),1100,3)]:
        light=bpy.data.lights.new('Softbox','AREA');light.energy=power;light.size=size;ob=bpy.data.objects.new('Softbox',light);bpy.context.collection.objects.link(ob);ob.location=loc;aim(ob,(0,0,1))
    camdata=bpy.data.cameras.new('Matched views');cam=bpy.data.objects.new('Matched views',camdata);bpy.context.collection.objects.link(cam);scene.camera=cam;camdata.type='ORTHO';camdata.ortho_scale=2.95
    center=Vector((0,.08,.97))-.10*right;elev=math.radians(15);cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
    if is_box:
        center=Vector((0,-.08,1.13));camdata.ortho_scale=3.1;cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
    if '--single' in args:
        front.location=(0,0,0);side.hide_render=True
        for _,cp in copies:cp.hide_render=True
        scene.render.resolution_x=1080;scene.render.resolution_y=620
        camdata.ortho_scale=4.15;center=Vector((-.025,.04,1.10))
        cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
    if study_view:
        front.location=-1.3*right
        side.matrix_world=Matrix.Translation(1.3*right+pivot)@Matrix.Rotation(math.radians(115),4,'Z')@Matrix.Translation(-pivot)@Matrix.Scale(scale,4)
        scene.render.resolution_x=1600;scene.render.resolution_y=860
        scene.eevee.taa_render_samples=24
        camdata.ortho_scale=5.6;center=Vector((0,.04,1.13))
        cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
        if summary_standing:
            side.matrix_world=Matrix.Translation(1.3*right+pivot)@Matrix.Rotation(math.radians(180),4,'Z')@Matrix.Translation(-pivot)@Matrix.Scale(scale,4)
        if '--wide-review' in args:
            front.location=-1.55*right
            side.matrix_world=Matrix.Translation(1.55*right+pivot)@Matrix.Rotation(math.radians(115),4,'Z')@Matrix.Translation(-pivot)@Matrix.Scale(scale,4)
            camdata.ortho_scale=7.4;center=Vector((0,.04,1.4))
            cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center)
    detail_side=args[args.index('--grip-hand')+1] if '--grip-hand' in args else None
    if '--grip' in args or detail_side:
        side.hide_render=True
        for _,cp in copies:cp.hide_render=True
        center=Vector((-.60*right.x,.08-.60*right.y,1.48));cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;camdata.ortho_scale=1.2;aim(cam,center)
        if detail_side:
            assert detail_side in ['r','l']
            # Isolate the complete hand skin and shaft for inspection; this
            # does not alter the skin surface used by the collision audit.
            for ob in anatomy:
                if ob!=skin:ob.hide_render=True
            for ob in bar.children:
                if ob.get('bar_kind')=='plate':ob.hide_render=True
                if ob.name.startswith('Display grip '+('l' if detail_side=='r' else 'r')):ob.hide_render=True
            palm_keep=[]
            for v in skin.data.vertices:
                weight=sum(g.weight for g in v.groups if skin.vertex_groups[g.group].name.endswith('_'+detail_side) and any(p in skin.vertex_groups[g.group].name for p in ['mc','lunate','scaph','pisiform','triquetrum','capitate','trapez','hamate','thumb']))
                palm_keep.append(weight>.25)
            bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[v for v in bm.verts if not palm_keep[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free()
            bpy.context.view_layer.update()
            point=d['bar_position_m'][0]+(-.48 if detail_side=='r' else .48)*d['bar_rotation'][0,:,0]
            center=front.matrix_world@Vector(point/scale)
            hand_R=d['body_xmat'][0,idx['lunate_'+detail_side]].reshape(3,3)
            normal=(-1 if detail_side=='r' else 1)*hand_R[:,2]
            direction=normal+(-.2 if detail_side=='r' else .2)*d['bar_rotation'][0,:,0]
            cam.location=center+Vector(direction)*4
            camdata.ortho_scale=.27;aim(cam,center)
    matched_person='--matched-person-scale' in args
    if matched_person:
        # A common orthographic scale fixes pixels/metre. Only camera target
        # translation follows the body; crouching does not enlarge the person.
        dt=float(np.median(np.diff(d['times']))) if len(d['times'])>1 else 1/60;centres=d['body_xpos'][:,idx['pelvis']].copy();footlow=d['contact_positions_m'][:,:,2].min(1);headtop=d['body_xpos'][:,idx['head'],2]+.13;centres[:,2]=(footlow+headtop)/2;sigma=.065/dt;radius=max(1,int(np.ceil(3*sigma)));kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum();centres=np.column_stack([np.convolve(np.pad(centres[:,j],(radius,radius),mode='edge'),kernel,mode='valid') for j in range(3)]);camdata.ortho_scale=5.6
        tracking_records=[]
    render_dir='joint_peak_render' if '--joint-peak' in args else 'review_render' if '--wide-review' in args else 'exposure_render' if summary_standing else 'study_render' if study_view else 'render'
    folder.joinpath(render_dir).mkdir(exist_ok=True)
    render_fps=int(args[args.index('--fps')+1]) if '--fps' in args else 60 if study_view else 30
    ts=np.arange(round(d['times'][-1]*render_fps)+1)/render_fps
    if '--poster' in args and meta.get('movement')=='box_jump_reference_replay':frames=np.linspace(0,len(ts)-1,5).round().astype(int).tolist()
    else:frames=[0,int(np.argmin(abs(ts-meta['phase_times_s']['bottom']))),int(np.argmin(abs(ts-(meta['phase_times_s']['start_ascent']+.3))))] if '--poster' in args else range(len(ts))
    if '--poster' in args and 'COM_power_W' in d:
        peak_time=float(d['times'][int(np.argmax(d['COM_power_W']))]);frames=sorted(set([*frames,int(np.argmin(abs(ts-peak_time)))]))
    if '--poster' in args and is_box:
        frames=sorted(set([0,int(np.argmin(abs(ts-meta['phase_times_s']['bottom']))),int(np.argmin(abs(ts-meta['phase_times_s']['end_ascent']))),int(np.argmin(abs(ts-(meta['phase_times_s']['end_ascent']+meta['takeoff_velocity_m_s'][2]/9.80665)))),len(ts)-1]))
    if '--sample-times' in args:
        frames=sorted(set(int(np.argmin(abs(ts-float(t)))) for t in args[args.index('--sample-times')+1].split(',')))
    if detail_side:frames=[0]
    if not summary_standing:ts=np.linspace(0,float(d['times'][-1]),121);frames=range(121)
    if '--frame-start' in args:frames=[k for k in frames if k>=int(args[args.index('--frame-start')+1])]
    if '--frame-end' in args:frames=[k for k in frames if k<=int(args[args.index('--frame-end')+1])]
    for k in frames:
        if '--skip-existing' in args:
            existing=folder/render_dir/f'frame_{k:04d}.png'
            if existing.exists() and existing.read_bytes()[-12:]==b'\x00\x00\x00\x00IEND\xaeB`\x82':continue
        i=int(np.argmin(abs(d['times']-ts[k])));desired={}
        if update_tensions:update_tensions(float(min(ts[k],d['times'][-1])))
        for pb in rig.pose.bones:
            name='sacrum' if pb.name=='pelvis' else pb.name
            if name in idx:
                j=idx[name];desired[pb.name]=matrix(d['body_xpos'][i,j]/scale,d['body_xmat'][i,j].reshape(3,3))@ref[name].inverted()@br[pb.name]
        for pb in rig.pose.bones:
            if pb.name not in desired:continue
            pb.matrix_basis=br[pb.name].inverted()@br[pb.parent.name]@desired[pb.parent.name].inverted()@desired[pb.name] if pb.parent and pb.parent.name in desired else br[pb.name].inverted()@desired[pb.name]
            rig2.pose.bones[pb.name].matrix_basis=pb.matrix_basis.copy()
        for ob,name in bones:
            j=idx[name];ob.matrix_basis=matrix(d['body_xpos'][i,j]/scale,d['body_xmat'][i,j].reshape(3,3))@ref[name].inverted()
        if not no_bar:bar.matrix_basis=matrix(d['bar_position_m'][i]/scale,d['bar_rotation'][i])
        for ob,cp in copies:
            cp.matrix_basis=bar.matrix_basis@ob.matrix_basis if ob.parent==bar else ob.matrix_basis.copy()
        output=f'render/grip_{detail_side}.png' if detail_side else 'render/native_grip_reference.png' if '--native-grip-audit' in args else f'{render_dir}/frame_{k:04d}.png'
        if matched_person:
            target=Vector(centres[i]);xy=Vector((target.x,target.y,0.));front.location=-1.3*right-xy;Rs=Matrix.Rotation(math.radians(180 if summary_standing else 115),4,'Z');side.matrix_world=Matrix.Translation(1.3*right-(Rs.to_3x3()@xy))@Rs@Matrix.Scale(scale,4);center=Vector((0.,0.,target.z));cam.location=center+Vector((math.sin(yaw)*math.cos(elev),-math.cos(yaw)*math.cos(elev),math.sin(elev)))*7;aim(cam,center);tracking_records.append(dict(frame=int(k),time_s=float(ts[k]),body_center_m=list(target),ortho_width_m=5.6))
        bpy.context.view_layer.update();scene.render.filepath=str(folder/output);bpy.ops.render.render(write_still=True)
        if '--grip' in args and not detail_side and '--native-grip-audit' in args:audit_grip(skin,front,d,idx,scale,folder)
        print('SQUAT_RENDER',kg,k,len(ts),flush=True)
    if matched_person:(folder/('matched_camera.json' if not summary_standing else render_dir+'_camera.json')).write_text(json.dumps(dict(ortho_width_m=5.6,raw_pixels_per_m=1600/5.6,body_target_filter_s=.065,zoom_is_pose_independent=True,frames=tracking_records),indent=2))
    if not detail_side:
        setup_name='exposure_render_setup.json' if summary_standing else 'study_render_setup.json' if study_view else 'render_setup.json'
        (folder/setup_name).write_text(json.dumps(dict(fps=render_fps,frames=len(ts),main_yaw_deg=25,main_elevation_deg=15,
            inset='rear three-quarter' if summary_standing else 'true side' if '--single' not in args else None,
            display='integrated tension on common standing display pose' if summary_standing else 'conditional anatomical muscle tension' if '--tension' in args or '--study' in args else 'neutral anatomy',
            bar_support=None if no_bar else 'upper back',side_shaft='full length; plates omitted' if not no_bar else None,bottom_reserved_by_composer_px=384),indent=2))


if __name__=='__main__':main()
