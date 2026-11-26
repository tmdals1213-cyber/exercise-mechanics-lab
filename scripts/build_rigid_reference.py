"""Regenerate the diagnostic jump under the explicit rigid inertial profile.

Prescribed GRF/COP and posture targets remain inputs. This is a conditional
mechanical reference, not a measured/validated human jump or muscle solution.
"""
import argparse,json,hashlib,numpy as np
from rigid_baseline_profile import RigidBaselineRig
from jump_acceleration import AccelerationMotion
from centroidal_jump import force_plan
from jump_rig import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hz',type=int,default=600);args=ap.parse_args();rig=RigidBaselineRig()
    # Include the IK dependency in future pose-cache keys. Legacy pose caches
    # keyed only the wrapper and subject builder, not the solver implementation.
    deps=['wholebody_ik.py','jump_rig.py','parametric_body.py','rigid_baseline_profile.py'];cache_signature=hashlib.sha256((rig.b.report['implementation_sha256']+''.join(hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in deps)).encode()).hexdigest();rig.b.report['implementation_sha256']=cache_signature
    plan=force_plan(rig,'vertical',args.hz);motion=AccelerationMotion(rig,plan,'vertical',args.hz);dest=ROOT/f'results/mechanics_foundation/rigid_reference_{args.hz}hz';dest.mkdir(parents=True,exist_ok=True);error=None
    try:data=motion.run()
    except RuntimeError as e:error=str(e);data={k:np.asarray(v) for k,v in motion.trace.items()}
    np.savez_compressed(dest/('partial_motion.npz' if error else 'motion.npz'),**data)
    source=['rigid_baseline_profile.py','newton_euler_reference.py','jump_acceleration.py','centroidal_jump.py','jump_rig.py','parametric_body.py','wholebody_ik.py','build_rigid_reference.py']
    setup={'kind':'vertical_rigid_profile_diagnostic','physics_hz':args.hz,'profile':rig.profile,'coordinate_names':rig.b.coordinate_names,'body_names':[rig.m.body(i).name for i in range(rig.m.nbody)],'takeoff_s':plan['takeoff'],'landing_s':motion.t_land,'duration_s':motion.duration,'absorption_end_s':motion.t_absorb,'recovery_end_s':motion.t_recover,'GRF_and_COP_are_inputs':True,'human_motion_validated':False,'muscle_solution_available':False,'completed':error is None,'source_sha256':{n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in source}}
    setup['pose_cache_signature_sha256']=cache_signature
    (dest/'setup.json').write_text(json.dumps(setup,indent=2),encoding='utf-8');definition={'points':[{'body_id':bid,'body':rig.m.body(bid).name,'local_m':p.tolist(),'initial_world_m':world.tolist(),'forefoot':fore} for bid,p,world,fore in plan['points']],'mass_kg':float(rig.m.body_mass.sum()),'gravity_m_s2':9.80665,'ground_z_m':plan['ground']};(dest/'contact_definition.json').write_text(json.dumps(definition,indent=2),encoding='utf-8')
    if error:(dest/'failure.json').write_text(json.dumps({'status':'failed','error':error,'last_valid_time_s':float(data['times'][-1])},indent=2),encoding='utf-8');raise RuntimeError(error)
    print('RIGID_REFERENCE_COMPLETE',dest,flush=True)
if __name__=='__main__':main()
