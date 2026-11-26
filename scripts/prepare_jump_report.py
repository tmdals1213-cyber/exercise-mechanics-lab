"""Self-contained, offline diagnostic report; not a production video."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/jump_validation'

def main():
 cases={}
 for kind,hz in [('vertical',2400),('long',600)]:
  folder=OUT/kind/f'acceleration_{hz}hz';path=folder/'motion.npz';complete=path.exists();z=dict(np.load(path if complete else folder/'partial_motion.npz'));setup=json.loads((folder/'setup.json').read_text());takeoff=setup['takeoff_s'];landing=setup['landing_s'];t=z['times'];ix=np.unique(np.r_[np.arange(0,len(t),max(1,hz//120)),len(t)-1]);com=z['com'];force=z['force_world_N'];mass=90.;g=9.80665
  snapshots=[{'key':k,'time':float(t[np.argmin(abs(t-s))])} for k,s in [('ready',0),('push',.92),('apex',takeoff+.25),('landing',landing+.07)]]
  verification=json.loads((folder/'verification.json').read_text()) if (folder/'verification.json').exists() else None
  failure=json.loads((folder/'failure.json').read_text()) if not complete else None
  cases[kind]={'title':'Vertical jump' if kind=='vertical' else 'Standing long jump','folder':f'{kind}/acceleration_{hz}hz','complete':complete,'hz':hz,'takeoff':takeoff,'landing':landing,'t':t[ix].tolist(),'com_z':com[ix,2].tolist(),'com_forward':(-com[ix,1]+com[0,1]).tolist(),'Fz_BW':(force[ix,2]/(mass*g)).tolist(),'Fy_BW':(-force[ix,1]/(mass*g)).tolist(),'snapshots':snapshots,'verification':verification,'failure':failure,'max_COM_rise_m':float(com[:,2].max()-com[0,2]),'peak_ground_force_BW':float(force[:,2].max()/(mass*g))}
 convergence=[]
 for hz in [600,1200,2400]:
  p=OUT/f'vertical/acceleration_{hz}hz/verification.json'
  if p.exists():
   v=json.loads(p.read_text());convergence.append({k:v[k] for k in ['physics_hz','energy_work_error_J','ballistic_COM_max_error_m','maximum_sticking_contact_speed_m_s','maximum_net_joint_moment_Nm','maximum_net_joint_power_W','numerical_consistency_passed']})
 readiness=OUT/'muscle_prediction_readiness.json'
 payload={'cases':cases,'convergence':convergence,'muscle_prediction_readiness':json.loads(readiness.read_text()) if readiness.exists() else None};(OUT/'report_data.json').write_text(json.dumps(payload,indent=2))
 template=(ROOT/'scripts/jump_report_template.html').read_text(encoding='utf-8');(OUT/'overview.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload)),encoding='utf-8')
 print('JUMP_DIAGNOSTIC_REPORT',OUT/'overview.html')

if __name__=='__main__':main()
