"""Contact-driven box jump with a measured CMJ posture seed.

The seed is a kinematic reference, not measured box-jump data. Ground forces
are solved from the floating-base equations. A prescribed separating release
is allowed, with no velocity reset or external flight support.
"""
import argparse,json,hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from build_box_jump_power import BoxJump,build,ROOT
from centroidal_jump import Polynomial,s5

class SeededBoxJump(BoxJump):
    def __init__(self,rig,hz,height):
        source=ROOT/'results/measured_jump_analysis/s10_CMJ_t1'
        z=np.load(source/'motion.npz');setup=json.loads((source/'setup.json').read_text())
        rig.contact_stand_override=z['qpos'][0].copy();super().__init__(rig,hz,height);self.tempo=.80
        self.source_q=CubicSpline(z['times'],z['qpos'],axis=0);self.source_com=CubicSpline(z['times'],z['com'],axis=0)
        self.source_rot=RotationSpline(z['times'],Rotation.from_quat(z['qpos'][:,[4,5,6,3]]))
        torso=setup['body_names'].index('torso');self.source_torso=RotationSpline(z['times'],Rotation.from_matrix(z['body_xmat'][:,torso].reshape(-1,3,3)))
        self.source_zero=z['qpos'][0];self.source_com_zero=z['com'][0];self.root_zero=Rotation.from_quat(self.stand[[4,5,6,3]])
        self.bottom=setup['countermovement_end_s']*self.tempo;self.toe=setup['heel_rise_start_s']*self.tempo
        self.takeoff=(setup['takeoff_s']-.008)*self.tempo;self.planned_takeoff_s=self.takeoff;self.start_descent=.25*self.tempo
        self.soft_ground_momentum_target=True;self.ground_momentum_weight=1000.;self.adaptive_toe_release=False
        self.quiet_warp=Polynomial(0,0,0,.25/self.tempo,1/self.tempo,0,.25)
        self.forward_shift_before=Polynomial(0,0,0,-.05,-.20,0,self.bottom-.25)
        self.forward_shift=Polynomial(-.05,-.20,0,-.12,-.65,0,self.takeoff-self.bottom)
        self.release_start_s=self.takeoff-.06;self.release_boundary_force=None
        momentum=[]
        for q,v in zip(z['qpos'],z['qvel']):
            rig.set_state(q,np.r_[v[:6],v[rig.dofs]]);momentum.append(rig.d.subtree_angmom[1].copy())
        self.source_momentum=CubicSpline(z['times'],momentum,axis=0)
        self.landing_above_box_m=.75-.25*height;self.range_based_box_placement=True
        self.flight_knee_position_weight=1000.;self.flight_orientation_weight=1000000.
        self.requested_launch_vz_m_s=float(self.source_com(setup['takeoff_s'],1)[2]/self.tempo)
        self.profile=dict(kind='measured CMJ posture seed plus contact dynamics',source_trial='s10_CMJ_t1',source_motion_sha256=hashlib.sha256((source/'motion.npz').read_bytes()).hexdigest(),time_ratio=self.tempo,box_data_measured=False,release='C2 COM acceleration transition to gravity; separating release with no velocity reset',unweighting_duration_s=.06)
        self.profile['box_placement']='static platform placed from common launch range and chosen pre-impact pose; not independently measured'

    def source_time(self,t):
        if t<0:return 0.,0.,0.
        if t<.25:return tuple(float(self.quiet_warp.value(t,k)[0]) for k in range(3))
        return t/self.tempo,1/self.tempo,0.

    def reference(self,t):
        if self.air_poly is not None:return self.air_poly.value(min(t-self.actual_takeoff,self.air_transition_T))
        ts,_,_=self.source_time(t);source=self.source_q(ts);q=self.stand.copy();q[self.rig.ids]=source[self.rig.ids]
        R=self.source_rot(ts)*self.source_rot(0).inv()*self.root_zero;q[3:7]=np.roll(R.as_quat(),1)
        for side in ['r','l']:
            q[self.m.jnt_qposadr[self.b.joints['shoulder_elv_'+side]]]=np.clip(q[self.m.jnt_qposadr[self.b.joints['shoulder_elv_'+side]]],.001,1.85)
            q[self.m.jnt_qposadr[self.b.joints['mtp_angle_'+side]]]=-.35*s5((t-self.toe)/(self.takeoff-self.toe))
        # Reference queries must be pure: complete() updates MuJoCo geometry.
        # Only independent coordinates enter the posture controller here.
        return q

    def com_acceleration_target(self,t):
        ts,rate,rate_dot=self.source_time(t);target=self.c0+self.source_com(ts)-self.source_com_zero
        velocity=self.source_com(ts,1)*rate;acceleration=self.source_com(ts,2)*rate**2+self.source_com(ts,1)*rate_dot
        if t>=.25:
            poly=self.forward_shift_before if t<self.bottom else self.forward_shift;tt=t-.25 if t<self.bottom else t-self.bottom
            target[1]+=poly.value(tt)[0];velocity[1]+=poly.value(tt,1)[0];acceleration[1]+=poly.value(tt,2)[0]
        return acceleration+100*(target-self.d.subtree_com[1])+20*(velocity-self.d.subtree_linvel[1])

    def release_acceleration_target(self,t):
        if self.release_boundary_force is None:self.release_boundary_force=self.previous_force_world_N.copy()
        u=np.clip((t-self.release_start_s)/(self.takeoff-self.release_start_s),0,1)
        # This specifies a smooth kinematic boundary condition. The QP must
        # still find nonnegative, friction-feasible forces and joint motion;
        # no force is clipped and no root support is applied.
        return self.release_boundary_force*(1-s5(u))/self.m.body_mass.sum()+self.m.opt.gravity

    def ground_momentum_target(self,t):
        ts,rate,rate_dot=self.source_time(t);desired=self.source_momentum(ts)*rate
        derivative=self.source_momentum(ts,1)*rate**2+self.source_momentum(ts)*rate_dot
        return derivative+8*(desired-self.d.subtree_angmom[1])

    def trunk_target_rotation(self,t):
        if self.actual_takeoff is not None:return self.Rupright
        return self.source_torso(self.source_time(t)[0]).as_matrix()@self.source_torso(0).as_matrix().T@self.Rstand

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--height-cm',type=int,choices=[40,60,80],default=60);p.add_argument('--hz',type=int,default=600);p.add_argument('--reuse-ground',action='store_true');a=p.parse_args()
    build(a.height_cm,a.hz,reuse_ground=a.reuse_ground,controller_type=SeededBoxJump,output_suffix='seeded')
