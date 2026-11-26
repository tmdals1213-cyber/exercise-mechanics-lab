"""Isolated low-box feasibility trial; no accepted comparison is overwritten.

Use the previously upright contact controller with zero centroidal momentum,
adding a modest forward motion target. All contact forces remain EOM outputs.
This is a prescribed task, not measured box-jump data or a height-power law.
"""
import argparse
import numpy as np
from build_box_jump_power import BoxJump,build
from build_contact_jump import ContactJump
from centroidal_jump import Polynomial,s5


class LowBoxJump(BoxJump):
    def __init__(self,rig,hz,height):
        ContactJump.__init__(self,rig,hz)
        self.box_height=height;self.bottom=.85;self.toe=.97;self.takeoff=1.03
        self.requested_launch_vz_m_s=2.7;self.crouch_drop_m=.20
        self.soft_ground_momentum_target=False;self.adaptive_toe_release=False
        self.toe_release_events=[];self.box_rear_y=-.12;self.box_front_y=-.77
        self.forward_T=self.toe-.25
        toe_y=float(min(p[2][1] for p in self.p['points']))
        self.forward=Polynomial(self.c0[1],0,0,toe_y-.06,-.55,0,self.forward_T)
        self.flight_point_weight=1e7;self.flight_orientation_weight=1e5
        self.flight_knee_position_weight=1000.;self.flight_knee_half_width_m=.12
        self.flight_pelvis_orientation_weight=1e7
        self.flight_trunk_weight=1e7
        self.flight_posture_weight_by_joint={}
        for side in ['r','l']:
            shoulder=self.b.joints['shoulder_elv_'+side]
            self.m.jnt_range[shoulder,1]=min(self.m.jnt_range[shoulder,1],1.86)
            for name,value in [('hip_flexion',1.9),('knee_angle',2.05),('mtp_angle',0.)]:
                self.air[self.m.jnt_qposadr[self.b.joints[name+'_'+side]]]=value
            self.flight_posture_weight_by_joint['hip_flexion_'+side]=20.
            self.flight_posture_weight_by_joint['mtp_angle_'+side]=100.
        self.landing_above_box_m=.60;self.range_based_box_placement=True
        self.profile=dict(kind='low-box feasibility with upright contact controller',
                          measured=False,common_launch_target_for_heights=True,
                          requested_forward_speed_m_s=-.55,requested_vertical_speed_m_s=2.7,
                          ascent_target_duration_s=.18,
                          landing_included=False)

    def com_acceleration_target(self,t):
        ss=np.clip(t-.25,0,self.forward_T)
        cy,vy,ay=[float(self.forward.value(ss,k)[0]) for k in range(3)]
        if t>self.toe:cy+=vy*(t-self.toe)
        if t<.25:cy,vy,ay=self.c0[1],0.,0.
        zmin=self.c0[2]-self.crouch_drop_m
        if t<.25:cz,vz,az=self.c0[2],0.,0.
        elif t<self.bottom:
            p=Polynomial(self.c0[2],0,0,zmin,0,0,self.bottom-.25)
            cz,vz,az=[float(p.value(t-.25,k)[0]) for k in range(3)]
        else:
            T=self.takeoff-self.bottom;ss=t-self.bottom;u=np.clip(ss/T,0,1);G=9.80665
            C=np.pi/2*(self.requested_launch_vz_m_s/(G*T)+.5)
            cz=zmin+G*T*T*(-u**3/6+C*(u/np.pi-np.sin(np.pi*u)/np.pi**2))
            vz=G*T*(-u*u/2+C*(1-np.cos(np.pi*u))/np.pi);az=G*(-u+C*np.sin(np.pi*u))
            if ss>T:
                extra=ss-T;cz+=vz*extra-.5*G*extra*extra;vz-=G*extra;az=-G
        target=np.array([self.c0[0],cy,cz]);velocity=np.array([0.,vy,vz])
        return np.array([0.,ay,az])+100*(target-self.d.subtree_com[1])+20*(velocity-self.d.subtree_linvel[1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--height-cm',type=int,choices=[40,60,80],default=40)
    p.add_argument('--hz',type=int,default=250);args=p.parse_args()
    build(args.height_cm,args.hz,controller_type=LowBoxJump,output_suffix='low_trial')
