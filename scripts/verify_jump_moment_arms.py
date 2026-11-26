"""Independent central-difference check of reduced muscle moment arms.

Numerical agreement checks derivatives in this converted model, not fidelity to
measured human moment arms or to the original OpenSim model.
"""
import json
import numpy as np
from jump_rig import JumpRig,OUT

def main():
 rig=JumpRig();b=rig.b;m=b.m;folder=OUT/'vertical/acceleration_2400hz';z=dict(np.load(folder/'motion.npz'));checks=[];eps=1e-5
 if not np.allclose(m.actuator_gear[:,0],1):raise ValueError('This length-derivative check requires unit tendon gears')
 for requested in [0.,.55,.92,1.25,1.58]:
  index=int(np.argmin(abs(z['times']-requested)));q=z['qpos'][index];b.pose(q);B,_=rig.coordinates(q,np.zeros(rig.n));R=B.T@b.moments()
  for name in ['flex_extension','shoulder_elv_r','elbow_flexion_r','hip_flexion_r','knee_angle_r','ankle_angle_r']:
   jid=b.joints[name];coordinate=6+b.independent.index(jid);adr=m.jnt_qposadr[jid];plus=q.copy();minus=q.copy();plus[adr]+=eps;minus[adr]-=eps;b.pose(plus);lp=b.d.actuator_length.copy();b.pose(minus);lm=b.d.actuator_length.copy();finite_difference=-(lp-lm)/(2*eps);error=abs(finite_difference-R[coordinate]);j=int(error.argmax())
   checks.append({'time_s':float(z['times'][index]),'coordinate':name,'maximum_error_m':float(error[j]),'worst_element':b.names[j]})
 maximum=max(c['maximum_error_m'] for c in checks);report={'status':'passed' if maximum<5e-5 else 'failed','tests':len(checks),'elements_per_test':m.nu,'maximum_moment_arm_error_m':maximum,'step_rad':eps,'threshold_m':5e-5,'checks':checks,'scope':'Derivative consistency of the converted physical-SI model including coordinate couplers; not anatomical or experimental validation.'};(folder/'moment_arm_derivative_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ['status','tests','elements_per_test','maximum_moment_arm_error_m']}))
 if report['status']!='passed':raise RuntimeError('Muscle moment-arm derivative check failed')

if __name__=='__main__':main()
