from pathlib import Path
p=Path(__file__).parent;s=(p/'render_control_jump_blender.py').read_text()
anchor="    render_dir='joint_peak_render'"
assert anchor in s
insert='''    world_side='--world-side' in args
    if world_side:
        # One immutable world camera. Native body translation is preserved.
        front.location=(0.,0.,0.);front.scale=(scale,)*3
        rig2.hide_render=True
        for ob,cp in copies:cp.hide_render=True
        toe_y=float(d['contact_positions_m'][0,:,1].min())
        center_forward=1.;camera_z=1.15;width_m=3.3
        center=Vector((0.,toe_y-center_forward,camera_z));cam.location=center+Vector((-6.,0.,0.));aim(cam,center);camdata.ortho_scale=width_m
        scene.render.resolution_x=1080;scene.render.resolution_y=1050
        for ob in bpy.data.objects:
            if ob.name.startswith('Fixed ground reference'):ob.hide_render=True
        bpy.ops.mesh.primitive_plane_add(size=1,location=(0,toe_y-.85,-.012));ground=bpy.context.object;ground.name='Fixed side ground';ground.scale=(2.,4.,1.);ground.data.materials.append(material('Side ground',(.055,.075,.09)))
        (folder/'fixed_side_camera_v2.json').write_text(json.dumps(dict(camera_tracks_body=False,body_translation_preserved=True,true_side=True,view_from_negative_X=True,forward_is_screen_right=True,elevation_deg=0,ortho_width_m=width_m,resolution=[1080,1050],camera_center_forward_m=center_forward,camera_center_z_m=camera_z,initial_foremost_sole_y_m=toe_y,world_camera_location_m=list(cam.location)),indent=2))
'''
s=s.replace(anchor,insert+anchor)
s=s.replace("else 'study_render' if study_view else 'render'", "else 'fixed_side_render_v2' if world_side else 'study_render' if study_view else 'render'")
(p/'render_fixed_side_jump_blender.py').write_text(s);print('FIXED_SIDE_RENDERER_READY')
