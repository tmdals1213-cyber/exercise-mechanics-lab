from pathlib import Path
p=Path(__file__).parent
s=(p/'render_iron_man_contours_v5_blender.py').read_text()
s=s.replace("    bar=bpy.data.objects.new('Bar centre',None);", "    bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,radius=.024/scale)\n    com_marker=bpy.context.object;com_marker.name='Whole body COM';com_marker.parent=front;com_marker.data.materials.append(material('COM orange',(.95,.35,.08)));com_marker.location=d['com'][0]/scale\n    bar=bpy.data.objects.new('Bar centre',None);")
s=s.replace("if not summary_standing:ts=np.linspace(0,float(d['times'][-1]),121);frames=range(121)", "if not summary_standing:\n        count=round(float(d['times'][-1])*30)+1;ts=np.linspace(0,float(d['times'][-1]),count);frames=range(count)")
s=s.replace("        for ob,cp in copies:\n            cp.matrix_basis=", "        com_marker.location=d['com'][i]/scale\n        for ob,cp in copies:\n            cp.matrix_basis=")
(p/'render_control_jump_blender.py').write_text(s)
print('CONTROL_RENDERER_READY')
