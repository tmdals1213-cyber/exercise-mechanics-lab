"""Isolated 30 cm airborne-COM CMJ candidate; forces are EOM outputs."""
from pathlib import Path
import argparse,json,hashlib,traceback
import numpy as np
from scipy.spatial.transform import Rotation
from build_contact_jump import ContactJump
from centroidal_jump import Polynomial
from retarget_measured_jump import MeasuredJumpRig

ROOT=Path(__file__).resolve().parents[1]

class AdultCMJ(ContactJump):
    def __init__(self,rig,hz,requested_vz,push_duration=.25,crouch_drop=.15,exact_com=False):
        super().__init__(rig,hz)
        self.bottom=.85;self.toe=.94;self.takeoff=self.bottom+push_duration
        self.requested_launch_vz_m_s=requested_vz;self.crouch_drop_m=crouch_drop
        self.bottom_trunk_lean_rad=.50
        self.forward_T=self.toe-.25
        toe_y=float(np.mean([p[2][1] for p in self.p['points'] if p[3]]))
        self.forward=Polynomial(self.c0[1],0,0,toe_y+.005,0,0,self.forward_T)
        self.flight_trunk_weight=1e5;self.flight_pelvis_orientation_weight=1e4
        self.flight_posture_weight_by_joint={}
        # Preserve a recognizable hands-on-hips upper-body configuration.
        source=np.load(ROOT/'results/power_comparison/cmj_analysis/motion.npz')['qpos'][0]
        for q in [self.stand,self.low,self.ext,self.air]:
            for n in self.b.ind_names:
                if n.startswith(('elv_angle_','shoulder_','elbow_','pro_sup_','wrist_')):
                    j=self.b.joints[n];q[self.m.jnt_qposadr[j]]=source[self.m.jnt_qposadr[j]]
            q[:]=self.b.complete(q)
        self.b.pose(self.stand);self.c0=self.d.subtree_com[1].copy()
        self.forward=Polynomial(self.c0[1],0,0,toe_y+.005,0,0,self.forward_T)
        self.locked_joint_targets={n:float(self.stand[self.m.jnt_qposadr[self.b.joints[n]]]) for n in self.b.ind_names
            if n.startswith(('elv_angle_','shoulder_','elbow_','pro_sup_','wrist_'))}
        for side in ['r','l']:
            j=self.b.joints['shoulder_elv_'+side];self.m.jnt_range[j,1]=min(self.m.jnt_range[j,1],1.86)
        if exact_com:
            self.reduce_equalities=False;self.polish_equalities=True
            self.bottom_trunk_lean_rad=.50
            self.soft_ground_momentum_target=True;self.ground_momentum_weight=1e9
            self.ground_momentum_target=lambda t:-20*self.d.subtree_angmom[1]
            self.release_start_s=.25
            self.release_acceleration_target=self.com_acceleration_goal
            self.release_tracking_axes=[2]
            self.planned_takeoff_s=self.takeoff
            self.posture_weight_by_joint={}
            for name in ['flex_extension','lat_bending','axial_rotation']:
                self.locked_joint_targets[name]=float(self.stand[self.m.jnt_qposadr[self.b.joints[name]]])
            for side in ['r','l']:
                self.low[self.m.jnt_qposadr[self.b.joints['knee_angle_'+side]]]=1.25+(.4/.07)*(crouch_drop-.15)
                self.low[self.m.jnt_qposadr[self.b.joints['hip_flexion_'+side]]]=.85+(.4/.07)*(crouch_drop-.15)
                self.posture_weight_by_joint['knee_angle_'+side]=100.
                self.posture_weight_by_joint['hip_flexion_'+side]=50.

    def com_acceleration_goal(self,t):
        ss=np.clip(t-.25,0,self.forward_T)
        cy,vy,ay=[float(self.forward.value(ss,k)[0]) for k in range(3)]
        zmin=self.c0[2]-self.crouch_drop_m
        if t<self.bottom:
            p=Polynomial(self.c0[2],0,0,zmin,0,0,self.bottom-.25)
            cz,vz,az=[float(p.value(t-.25,k)[0]) for k in range(3)]
        else:
            T=self.takeoff-self.bottom;ss=t-self.bottom;u=np.clip(ss/T,0,1);g=9.80665
            C=np.pi/2*(self.requested_launch_vz_m_s/(g*T)+.5)
            cz=zmin+g*T*T*(-u**3/6+C*(u/np.pi-np.sin(np.pi*u)/np.pi**2))
            vz=g*T*(-u*u/2+C*(1-np.cos(np.pi*u))/np.pi);az=g*(-u+C*np.sin(np.pi*u))
            if ss>T:
                extra=ss-T;cz+=vz*extra-.5*g*extra*extra;vz-=g*extra;az=-g
        target=np.array([self.c0[0],cy,cz]);velocity=np.array([0.,vy,vz])
        return np.array([0.,ay,az])+100*(target-self.d.subtree_com[1])+20*(velocity-self.d.subtree_linvel[1])

    def configure_flight(self,q,u,qd,qdd):
        vz=float(self.d.subtree_linvel[1,2]);self.flight_T=max(.25,2*vz/9.80665*.93)
        self.air_transition_T=self.flight_T
        endpoint=self.air.copy();endpoint[:7]=q[:7]
        self.air_poly=Polynomial(q,qd,qdd,endpoint,0,0,self.air_transition_T)
        self.flight_feet=None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=250)
    ap.add_argument('--requested-vz',type=float,default=3.1);ap.add_argument('--label',default='pilot_250hz')
    ap.add_argument('--push-duration',type=float,default=.25);ap.add_argument('--crouch-drop',type=float,default=.15)
    ap.add_argument('--exact-com',action='store_true')
    a=ap.parse_args();out=ROOT/'results/power_comparison/cmj_adult_target'/a.label;out.mkdir(parents=True,exist_ok=True)
    rig=MeasuredJumpRig();motion=AdultCMJ(rig,a.hz,a.requested_vz,a.push_duration,a.crouch_drop,a.exact_com)
    try:data=motion.run()
    except Exception:
        trace=getattr(motion,'trace',{})
        if trace:np.savez_compressed(out/'rejected_trace.npz',**{k:np.asarray(v) for k,v in trace.items()})
        (out/'failure.txt').write_text(traceback.format_exc());raise
    np.savez_compressed(out/'motion.npz',**data)
    flight=data['mode']==2;k=int(np.flatnonzero(flight)[0]);mass=float(rig.m.body_mass.sum())
    chest=data['body_xmat'][:,rig.m.body('torso').id].reshape(-1,3,3)
    tilt=np.rad2deg(np.arccos(np.clip(chest[flight,:,2]@chest[0,:,2],-1,1)))
    report=dict(status='candidate_not_published',target_airborne_COM_height_m=.30,requested_launch_vz_m_s=a.requested_vz,
        requested_push_duration_s=a.push_duration,crouch_drop_m=a.crouch_drop,
        COM_trajectory_is_prescribed=a.exact_com,
        actual_takeoff_s=float(data['times'][k]),actual_takeoff_velocity_m_s=data['velocity'][k].tolist(),
        ballistic_height_from_v_m=float(data['velocity'][k,2]**2/(2*9.80665)),
        COM_rise_after_takeoff_m=float(data['com'][flight,2].max()-data['com'][k,2]),
        standing_to_apex_COM_rise_m=float(data['com'][:,2].max()-data['com'][0,2]),
        peak_vertical_GRF_N=float(data['force_world_N'][:,2].max()),
        peak_GRF_bodyweights=float(data['force_world_N'][:,2].max()/(mass*9.80665)),
        takeoff_angular_momentum_norm=float(np.linalg.norm(data['angular_momentum'][k])),
        flight_angular_momentum_max_drift=float(np.linalg.norm(data['angular_momentum'][flight]-data['angular_momentum'][k],axis=1).max()),
        maximum_flight_thorax_tilt_deg=float(tilt.max()),flight_ground_force_max_N=float(abs(data['contact_forces_N'][flight]).max()),
        mass_kg=mass,source_motion_sha256=hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest(),
        source_data_scaled=False,muscle_forces_solved=False,source_human_motion=False)
    (out/'candidate_summary.json').write_text(json.dumps(report,indent=2))
    (out/'setup.json').write_text(json.dumps(dict(profile=rig.profile,rig_type='measured_cmj_standard',physics_hz=a.hz,
        coordinate_names=rig.b.coordinate_names,body_names=[rig.m.body(i).name for i in range(rig.m.nbody)],mass_kg=mass,height_m=1.8,
        takeoff_s=motion.actual_takeoff,countermovement_end_s=motion.bottom,heel_rise_start_s=motion.toe,
        scope='Prescribed adult-reference CMJ; solved contacts; no landing or measured recruitment.',
        ground_reaction_and_COP_are_inputs=False,posture_goals_are_inputs=True,human_motion_validated=False),indent=2))
    (out/'contact_definition.json').write_text(json.dumps(dict(points=[dict(body_id=bid,body=rig.m.body(bid).name,
        local_m=local.tolist(),initial_world_m=p.tolist(),forefoot=fore) for bid,local,p,fore in motion.p['points']],
        mass_kg=mass,gravity_m_s2=9.80665,ground_z_m=motion.p['ground']),indent=2))
    print('ADULT_CMJ_CANDIDATE',json.dumps(report),flush=True)

if __name__=='__main__':main()
