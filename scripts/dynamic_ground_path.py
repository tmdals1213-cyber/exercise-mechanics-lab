"""Contact/acceleration QP: trajectory targets are soft, floating-root EOM hard.

Joint control is a motion-design assumption. Contact and range constraints
remain physical constraints; neither prescribed H nor a root motor is used.
"""
import json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from build_contact_jump import ContactJump
from centroidal_jump import Polynomial

def solve(r,template,com,velocity,acceleration,points,bottom,toe,tip,to,folder,hz=1000,balance_weight=1e7,reference_acceleration_scale=1.,task_profile=None):
    b,m,d=r.b,r.m,r.d;n=r.n
    times=np.unique(np.round(np.r_[np.arange(0,to,1/hz),bottom,toe,tip,to],10))
    samples=[];trunk=[];bid=m.body('torso').id
    for t in times:
        q=template(float(t));b.pose(q);samples.append(q);trunk.append(d.xmat[bid].reshape(3,3).copy())
    refs=CubicSpline(times,np.array(samples));trunk_ref=RotationSpline(times,Rotation.from_matrix(trunk))
    def reference(t):
        q=refs(t);q[3:7]/=np.linalg.norm(q[3:7]);return q
    for i,name in enumerate(b.ind_names):
        if name.startswith('shoulder_elv_'):m.jnt_range[b.independent[i],1]=min(m.jnt_range[b.independent[i],1],1.85)
    task=ContactJump.__new__(ContactJump);task.rig=r;task.b=b;task.m=m;task.d=d;task.n=n;task.W=float(m.body_mass.sum())*9.80665
    task.debug_folder=folder
    q=reference(0);q[:3]=0;b.pose(q);q[:3]+=com(0)-d.subtree_com[1];q=b.complete(q);u=np.zeros(n);r.set_state(q,u)
    task.p=dict(points=[p[:4]for p in points],ground=0.,feet={});task.stand=q.copy();task.c0=d.subtree_com[1].copy()
    task.contact_frame_reference={bid:(d.xmat[bid].reshape(3,3).copy(),d.xpos[bid].copy()) for bid,local,world,fore in points}
    task.bottom=bottom;task.takeoff=to;task.toe=toe;task.forward_T=bottom-.25;task.forward=Polynomial(task.c0[1],0,0,task.c0[1],0,0,task.forward_T)
    task.theta_stand=0.;task.Rupright=d.xmat[bid].reshape(3,3).copy();task.trunk_bid=bid;task.flight_feet=None
    task.reference=reference;task.trunk_target_rotation=lambda t:trunk_ref(t).as_matrix()
    task.horizontal_target=lambda t:(com(t)[1],velocity(t)[1],acceleration(t)[1])
    task.com_acceleration_target=lambda t:acceleration(t)+80*(com(t)-d.subtree_com[1])+18*(velocity(t)-d.subtree_linvel[1])
    task.soft_ground_momentum_target=True;task.ground_momentum_weight=balance_weight;task.polish_equalities=True
    task.reference_acceleration_scale=reference_acceleration_scale
    task.adaptive_toe_release=bool((task_profile or{}).get('release_zero_load_points',(task_profile or{}).get('adaptive_contacts',False)));task.toe_release_events=[];task.released_toe_points=set()
    task.posture_weight_by_joint={name:1. for name in b.ind_names if name.startswith(('elv_angle','shoulder_','elbow_'))}
    if task_profile:
        task.soft_ground_momentum_target=task_profile.get('soft_ground_momentum_target',True)
        task.joint_range_barrier_rate=task_profile.get('joint_range_barrier_rate',40.)
        task.trunk_tracking_gain=task_profile.get('trunk_tracking_gain',100.)
        if task_profile.get('track_trunk_feedforward'):
            task.trunk_target_kinematics=lambda t:(trunk_ref(t).as_matrix(),trunk_ref(t).apply(trunk_ref(t,1)),trunk_ref(t).apply(trunk_ref(t,2)))
        if task_profile.get('launch_angular_momentum') is not None:
            task.hard_momentum_start_s=task_profile['hard_momentum_start_s'];momentum_path={}
            def momentum_target(t):
                if t<task.hard_momentum_start_s:return -8*d.subtree_angmom[1]
                if not momentum_path:momentum_path['poly']=Polynomial(d.subtree_angmom[1].copy(),0,0,np.array(task_profile['launch_angular_momentum']),0,0,to-task.hard_momentum_start_s)
                poly=momentum_path['poly'];elapsed=t-task.hard_momentum_start_s
                return poly.value(elapsed,1)+8*(poly.value(elapsed)-d.subtree_angmom[1])
            task.ground_momentum_target=momentum_target
        task.locked_joint_targets=task_profile.get('locked_joint_targets',{})
        task.posture_weight_by_joint.update(task_profile.get('posture_weight_by_joint',{}))
        task.com_tracking_weight=task_profile.get('com_tracking_weight',5000.)
        task.ground_trunk_weight=task_profile.get('ground_trunk_weight',2000.)
        task.contact_stabilization_tolerance=task_profile.get('contact_stabilization_tolerance',1e-5)
        task.rigid_contact_frames=task_profile.get('rigid_contact_frames',False)
        task.symmetric_vertical_reactions=task_profile.get('symmetric_vertical_reactions',True)
        task.ground_momentum_axis_weights=task_profile.get('ground_momentum_axis_weights',[1.,1.,1.])
        if 'hard_com_tracking_axes' in task_profile:task.hard_com_tracking_axes=task_profile['hard_com_tracking_axes']
        if 'prescribed_joint_motion' in task_profile:task.prescribed_joint_motion=task_profile['prescribed_joint_motion']
        if 'hard_com_axes_after_start' in task_profile:
            task.hard_com_axes_after_start=task_profile['hard_com_axes_after_start'];task.hard_com_start_s=bottom
        if task_profile.get('horizontal_impulse_goal_m_s') is not None:
            original_control=task.com_acceleration_target;impulse={};G=9.80665
            def impulse_control(t):
                value=original_control(t)
                if t<bottom:return value
                if not impulse:
                    available=float(velocity(to)[2]-d.subtree_linvel[1,2]+G*(to-bottom))
                    impulse['ratio']=(task_profile['horizontal_impulse_goal_m_s']-float(d.subtree_linvel[1,1]))/available
                    impulse['latched_velocity_m_s']=d.subtree_linvel[1].tolist()
                    if abs(impulse['ratio'])>=.8:raise RuntimeError('Horizontal impulse request exceeds friction cone')
                value[1]=impulse['ratio']*(value[2]+G);return value
            task.com_acceleration_target=impulse_control
        if task_profile.get('latch_propulsion_state'):
            original_control=task.com_acceleration_target;launch={};G=9.80665;T=to-bottom
            def latched_control(t):
                value=original_control(t)
                if t<bottom:return value
                if not launch:
                    launch.update(z=float(d.subtree_com[1,2]),v=float(d.subtree_linvel[1,2]))
                s=t-bottom;u=s/T;v0=launch['v'];pulse=np.pi/(2*T)*(task_profile['launch_vz_m_s']-v0+G*T/2)
                z=launch['z']+v0*s+pulse*T*T*(u/np.pi-np.sin(np.pi*u)/np.pi**2)-G*T*T*u**3/6
                v=v0+pulse*T*(1-np.cos(np.pi*u))/np.pi-G*T*u*u/2
                acc=pulse*np.sin(np.pi*u)-G*u
                value[2]=acc+80*(z-d.subtree_com[1,2])+18*(v-d.subtree_linvel[1,2]);return value
            task.com_acceleration_target=latched_control
    states=[q.copy()];velocities=[u.copy()];accelerations=[];forces=[]
    def step(q,u,t):
        if hasattr(task,'hard_momentum_start_s') and t>=task.hard_momentum_start_s:task.soft_ground_momentum_target=False
        # Toe patch shrinkage follows unilateral reactions. Do not forcibly
        # remove the metatarsal edge if the toe tips have already unloaded.
        if getattr(task,'flight_mode_selected',False):return task.solve(q,u,t,'flight')
        if (task_profile or{}).get('reactivate_closed_contacts') and task.released_toe_points:
            r.set_state(q,u);returned=[]
            for i in list(task.released_toe_points):
                bid,local,anchor,_=points[i];J,p=b.jacobian(bid,local)
                # Restore only a still-closed, stationary zero-load contact.
                # A point that slid or rolled cannot be pinned back to its
                # old anchor. Actual re-impact requires its own contact solve.
                if abs(p[2])<1e-5 and np.linalg.norm(J@d.qvel)<.001 and np.linalg.norm(p-anchor)<2e-5:
                    task.released_toe_points.remove(i);returned.append(i)
            if returned:task.toe_release_events.append(dict(time_s=float(t),reactivate_point_indices=returned))
        mode='toe'if t>toe or getattr(task,'toe_mode_selected',False)else'plant'
        try:
            result=task.solve(q,u,t,mode)
            if mode=='plant' and t>bottom and (task_profile or{}).get('release_unloaded_heels'):
                heel=[i for i,p in enumerate(points) if not p[3]]
                if float(result[-1][heel,2].sum())<1e-3:
                    saved=set(task.released_toe_points);event_count=len(task.toe_release_events)
                    try:
                        proposed=task.solve(q,u,t,'toe');r.set_state(q,u)
                        for i in heel:
                            J,p=b.jacobian(points[i][0],points[i][1])
                            if p[2]<-.0001 or (J@d.qvel)[2]<-.002:raise RuntimeError('Heel release is not separating')
                        task.toe_mode_selected=True;task.toe_release_events.append(dict(time_s=float(t),point_indices=heel,mode_selection='Heel reaction is zero; toe support has a nonpenetrating solution.'))
                        return proposed
                    except RuntimeError:
                        task.released_toe_points=saved;del task.toe_release_events[event_count:];r.set_state(q,u)
            return result
        except RuntimeError as error:
            if mode=='plant' and (task_profile or{}).get('adaptive_contacts'):
                try:
                    result=task.solve(q,u,t,'toe');r.set_state(q,u)
                    for point in points:
                        if point[3]:continue
                        J,p=b.jacobian(point[0],point[1])
                        if p[2]<-.002 or(J@d.qvel)[2]<-.02:raise error
                    task.toe_mode_selected=True
                    task.toe_release_events.append(dict(time_s=float(t),point_indices=[i for i,p in enumerate(points)if not p[3]],mode_selection='Unilateral heel release feasible under root dynamics and friction.'))
                    return result
                except RuntimeError:raise error
            if mode=='plant':raise
            if (task_profile or{}).get('search_forefoot_contact_modes'):
                # A flat forefoot can become infeasible before its fixed-patch
                # solve assigns zero force to an edge. Test the unilateral
                # tip/ball alternatives before incorrectly declaring flight.
                saved=set(task.released_toe_points);count=len(task.toe_release_events)
                candidates=[('toe_tip_line',[2,3,8,9]),('metatarsal_line',[4,5,10,11]),('medial_points',[2,4,5,9,10,11]),('lateral_points',[3,4,5,8,10,11])]
                for name,removed in candidates:
                    task.released_toe_points=saved|set(removed);del task.toe_release_events[count:]
                    retained=[i for i,p in enumerate(points) if p[3] and i not in task.released_toe_points]
                    if not any(i<6 for i in retained) or not any(i>=6 for i in retained):continue
                    try:
                        proposed=task.solve(q,u,t,'toe');r.set_state(q,u)
                        for i in set(removed)-saved:
                            J,p=b.jacobian(points[i][0],points[i][1])
                            if p[2]<-.0001 or (J@d.qvel)[2]<-.002:raise RuntimeError('Contact-mode release is not separating')
                        task.toe_mode_selected=True;task.toe_release_events.append(dict(time_s=float(t),point_indices=sorted(set(removed)-saved),mode_selection='nonpenetrating '+name+' support after fixed forefoot infeasibility'))
                        return proposed
                    except RuntimeError:pass
                task.released_toe_points=saved;del task.toe_release_events[count:];r.set_state(q,u)
            if (task_profile or{}).get('minimum_launch_speed_m_s') is not None:
                r.set_state(q,u)
                if float(d.subtree_linvel[1,2])<task_profile['minimum_launch_speed_m_s']:raise error
            # Test the other unilateral-contact mode. A free solution is only
            # admissible with no support force and nonpenetrating foot motion;
            # ContactJump enforces those acceleration inequalities in flight.
            saved=set(task.released_toe_points)
            try:result=task.solve(q,u,t,'flight')
            except Exception:
                task.released_toe_points=saved;raise error
            r.set_state(q,u)
            for point in points:
                J,p=b.jacobian(point[0],point[1])
                if p[2]<-.002 or (p[2]<.002 and (J@d.qvel)[2]<-.02):raise error
            task.flight_mode_selected=True;task.released_toe_points=set(range(len(points)))
            task.toe_release_events.append(dict(time_s=float(t),point_indices=list(range(len(points))),mode_selection='unactuated free flight feasible; nonpenetrating sole motion'))
            return result
    actual_to=to
    for k,(t,tn)in enumerate(zip(times[:-1],times[1:])):
        dt=float(tn-t)
        try:
            result=step(q,u,float(t));a,B,curve=result[:3]
            mid=q.copy();mujoco.mj_integratePos(m,mid,B@u,dt/2);mid=b.complete(mid);um=u+dt*a/2
            rm=step(mid,um,float((t+tn)/2));am,Bm=rm[:2]
            nq=q.copy();mujoco.mj_integratePos(m,nq,Bm@um,dt);q=b.complete(nq);u+=dt*am
        except Exception as e:
            np.savez_compressed(folder/'dynamic_ground_partial.npz',times=times[:len(states)],qpos=np.array(states),u=np.array(velocities))
            (folder/'dynamic_ground_failure.json').write_text(json.dumps(dict(time_s=float(t),reason=str(e),completed=False),indent=2));raise
        states.append(q.copy());velocities.append(u.copy());accelerations.append(a);forces.append(result[-1])
        if k%200==0:print('DYNAMIC_GROUND',k,len(times),round(float(tn),3),flush=True)
        if float(tn)>bottom and float(rm[-1][:,2].sum())<1e-4:
            actual_to=float(tn);print('DYNAMIC_LIFTOFF',actual_to,flush=True);break
    times=times[:len(states)]
    np.savez_compressed(folder/'dynamic_ground_integrated.npz',times=times,qpos=np.array(states),u=np.array(velocities),udot=np.array(accelerations),contact_forces_N=np.array(forces))
    r.set_state(q,u)
    report=dict(status='contact_acceleration_QP_complete',COM_target_is_soft=not bool(getattr(task,'hard_com_tracking_axes',[])),hard_COM_tracking_axes=getattr(task,'hard_com_tracking_axes',[]),ground_momentum_objective_weight=task.ground_momentum_weight,root_dynamics_exact=True,no_root_motor=True,takeoff_velocity_m_s=d.subtree_linvel[1].tolist(),takeoff_H=d.subtree_angmom[1].tolist(),integration_hz=hz,human_motion_validated=False,actual_takeoff_s=actual_to,release_events=task.toe_release_events)
    report.update(maximum_redundant_contact_stabilization_residual=float(getattr(task,'maximum_contact_stabilization_residual',0.)),contact_stabilization_tolerance=float(getattr(task,'contact_stabilization_tolerance',1e-5)))
    report['hard_COM_tracking_axes_after_start']=getattr(task,'hard_com_axes_after_start',[])
    if (task_profile or{}).get('horizontal_impulse_goal_m_s') is not None:report['horizontal_impulse']=impulse
    (folder/'dynamic_ground_summary.json').write_text(json.dumps(report,indent=2))
    return times,np.array(states),report
