"""Explicit display-only fingers around a bar; no hand-force inference.

The native palm/wrist anchor follows the mechanical rig. Original finger
surfaces are replaced by smooth swept geometry. This presentation asset does
not change rigid-body inertias, forces or joint coordinates.
"""
import math
import bpy,bmesh
import numpy as np


def install(skin,bar,motion,idx,scale,material):
    removed=[]
    for v in skin.data.vertices:
        w=sum(g.weight for g in v.groups if any(k in skin.vertex_groups[g.group].name for k in ['proxph','midph','distph','proximal_thumb','distal_thumb']))
        removed.append(w>.45)
    bm=bmesh.new();bm.from_mesh(skin.data);bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm,geom=[v for v in bm.verts if removed[v.index]],context='VERTS');bm.to_mesh(skin.data);bm.free()
    R=motion['bar_rotation'][0];origin=motion['bar_position_m'][0]
    local=lambda name:(motion['body_xpos'][0,idx[name]]-origin)@R
    records=[]
    for side in ['r','l']:
        sign=1.
        for digit,L in [(2,.092),(3,.100),(4,.095),(5,.080),(1,.072)]:
            thumb=digit==1
            start=local(('proximal_thumb' if thumb else f'{digit}proxph')+'_'+side)
            radius=.008 if thumb else .0068 if digit<4 else .0062
            circle=.014+radius+.0012
            angle=math.atan2(start[2],start[1]);r0=max(float(np.linalg.norm(start[1:])),circle)
            if not thumb:
                direction=local(f'midph{digit}_'+side)-start
                s=float(start[1]*direction[2]-start[2]*direction[1])
                sign=1. if s>=0 else -1.
            turn=(-sign*2.8 if thumb else sign*max(3.2,min(3.7,(L-(r0-circle))/circle)))
            points=[]
            for u in np.linspace(0,1,49):
                rr=circle+(r0-circle)*(1-min(1,u/.24))**2
                a=angle+turn*u
                x=start[0]+(-.012 if side=='r' else .012)*u if thumb else start[0]
                points.append(np.array([x,rr*math.cos(a),rr*math.sin(a)]))
            points=np.array(points);verts=[];faces=[];sides=16
            for k,p in enumerate(points):
                tangent=points[min(k+1,len(points)-1)]-points[max(k-1,0)];tangent/=np.linalg.norm(tangent)
                a=np.array([1.,0,0]);a-=tangent*(a@tangent);a/=np.linalg.norm(a);c=np.cross(tangent,a)
                for phi in np.linspace(0,2*math.pi,sides,endpoint=False):verts.append(((p+radius*(a*math.cos(phi)+c*math.sin(phi)))/scale).tolist())
                if k:
                    for j in range(sides):faces.append(((k-1)*sides+j,(k-1)*sides+(j+1)%sides,k*sides+(j+1)%sides,k*sides+j))
            faces.extend([tuple(range(sides-1,-1,-1)),tuple(range((len(points)-1)*sides,len(points)*sides))])
            mesh=bpy.data.meshes.new(f'Display grip {side} {digit}');mesh.from_pydata(verts,[],faces);mesh.materials.append(material)
            ob=bpy.data.objects.new(mesh.name,mesh);bpy.context.collection.objects.link(ob);ob.parent=bar;ob['display_finger']=True
            for p in mesh.polygons:p.use_smooth=True
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=radius/scale)
            cap=bpy.context.object;cap.name=mesh.name+' tip';cap.parent=bar;cap.location=points[-1]/scale;cap.data.materials.append(material)
            for p in cap.data.polygons:p.use_smooth=True
            # The swept centre never comes closer than rod + finger radius
            # + 1.2 mm. Account conservatively for chord interpolation.
            delta=abs(turn)/48;bound=circle*math.cos(delta/2)-radius-.014
            records.append(dict(side=side,digit=digit,wrap_angle_deg=abs(turn)*180/math.pi,shaft_clearance_lower_bound_m=bound))
    return dict(status='display_geometry_only',native_finger_surfaces_replaced=True,hand_forces_recomputed=False,
                inertias_changed=False,palm_and_wrist_follow_mechanical_rig=True,fingers=records)
