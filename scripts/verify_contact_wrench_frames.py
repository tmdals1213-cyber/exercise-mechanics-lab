from pathlib import Path
import json,time,numpy as np,mujoco
from native_rigid_upper_landing import actual_contact_wrench
ROOT=Path(__file__).resolve().parents[1]
def main():
 results=[]
 for kind in ['plane','box']:
  floor='<geom type="plane" size="2 2 .1"/>'if kind=='plane'else'<geom type="box" pos="0 0 -.1" size="2 2 .1"/>'
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+floor+'<body pos="0 0 .049"><freejoint/><geom type="sphere" size=".05" mass="1" friction=".8 .005 .0001"/></body></worldbody></mujoco>');d=mujoco.MjData(m);d.qvel[:3]=[.4,.3,0];mujoco.mj_forward(m,d);F,M,COP=actual_contact_wrench(m,d);mujoco.mj_rnePostConstraint(m,d);nativeF=d.cfrc_ext[:,3:].sum(0);nativeM=d.cfrc_ext[:,:3].sum(0);C=d.subtree_com[1];ferr=float(abs(F-nativeF).max());merr=float(abs(M-np.cross(C,F)-nativeM).max());assert d.ncon>0 and ferr<1e-9 and merr<1e-9;results.append(dict(kind=kind,static_world_is_geom2=bool(m.geom_bodyid[d.contact[0].geom2]==0),force_error_N=ferr,moment_error_Nm=merr,force_world_N=F.tolist()))
 report=dict(status='passed',tests=results,native_reference='mj_rnePostConstraint cfrc_ext summed independently over dynamic bodies',change='Vectorized aggregation; full force and torque sign reversed when world is geom2.');(ROOT/'results/contact_wrench_frame_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
