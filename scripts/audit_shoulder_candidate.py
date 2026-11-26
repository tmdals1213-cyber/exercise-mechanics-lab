"""Executable thoracoscapular COMPONENT assessment, not a whole-body solver."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import opensim as osim

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,default=Path('D:/Exercise_mechanics/models/foundation/TSM_subject_noWeight.osim')); args=ap.parse_args()
    original=ROOT/'models/rmr/TSM_subject_noWeight.osim'
    assert hashlib.sha256(args.model.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest(), 'Candidate model differs from recorded source'
    dest=ROOT/'results/model_foundation/shoulder_candidate'; dest.mkdir(parents=True,exist_ok=True)
    osim.Logger.setLevelString('error'); model=osim.Model(str(args.model)); model.setUseVisualizer(False)
    state=model.initSystem(); model.realizePosition(state)
    coordinates=list(model.getCoordinateSet()); muscles=list(model.getMuscles())
    initial={c.getName():c.getValue(state) for c in coordinates}
    # An assembled geometry probe only. Fix trunk/arm coordinates and the
    # requested scapular upward rotation is a weighted assembly goal; allow
    # shoulder-girdle coordinates to satisfy the native AC point constraint.
    fixed=[c for c in coordinates if c.getName().startswith('ground_') or c.getName() in ['plane_elv','shoulder_elv','axial_rot','elbow_flexion','pro_sup']]
    target=model.getCoordinateSet().get('scapula_upward_rot')
    all_names=[c.getName() for c in coordinates]
    offsets=np.linspace(-10.,30.,41)
    states=[]; lengths=[]; arm_coefficients=[]; qerr=[]; rotations=[]
    for degrees in offsets:
        s=osim.State(state)
        for c in fixed: c.setLocked(s,True)
        target.setValue(s,initial[target.getName()]+np.deg2rad(degrees),False)
        model.assemble(s,target,100.); model.realizePosition(s)
        states.append([c.getValue(s) for c in coordinates])
        lengths.append([muscle.getLength(s) for muscle in muscles])
        qerr.append([s.getQErr().get(i) for i in range(s.getQErr().size())])
        rotations.append(target.getValue(s))
        # These coefficients use OpenSim's constraint-aware MomentArmSolver.
        probe=osim.State(s); model.realizePosition(probe)
        arm_coefficients.append([muscle.computeMomentArm(probe,target) for muscle in muscles])
    lengths=np.asarray(lengths); arms=np.asarray(arm_coefficients); states=np.asarray(states)
    assert np.isfinite(lengths).all() and np.all(lengths>0) and np.isfinite(arms).all()
    closure=float(abs(np.asarray(qerr)).max()); assert closure<1e-6, closure
    scapular=[i for i,m in enumerate(muscles) if m.getName().startswith(('Trapezius','Serratus','Rhomboideus','Levator'))]
    assert len(scapular)==10 and np.max(abs(arms[:,scapular]))>.001
    np.savez_compressed(dest/'geometry_probe.npz',requested_offset_deg=offsets,actual_upward_rotation_rad=rotations,coordinate_positions=states,musculotendon_length_m=lengths,upward_rotation_moment_arm_m=arms,AC_constraint_error_m=qerr)
    muscle_info=[{'name':m.getName(),'muscle_type':m.getConcreteClassName(),'maximum_isometric_force_parameter_N':m.get_max_isometric_force(),'optimal_fiber_length_m':m.get_optimal_fiber_length(),'tendon_slack_length_m':m.get_tendon_slack_length(),'pennation_at_optimal_rad':m.get_pennation_angle_at_optimal(),'ignore_tendon_compliance':m.get_ignore_tendon_compliance()} for m in muscles]
    result={'status':'component_geometry_probe_passed','whole_body_integrated':False,'OHP_force_solution':False,'model_name':model.getName(),'muscle_elements':len(muscles),'scapular_muscle_elements':[muscles[i].getName() for i in scapular],
            'coordinates':all_names,'coordinate_count':len(coordinates),'generalized_speeds':state.getNU(),'constraint_position_equations':state.getNQErr(),'initial_coordinate_values':initial,
            'AC_closure_max_error_m':closure,'probe':'41 assembled positions, scapular upward rotation goal offset -10 to +30 degrees from source default, weight=100; actual assembled values saved separately; trunk and GH/elbow/forearm held. Mechanism probe, not a measured shoulder rhythm or prescribed OHP.',
            'moment_arm_definition':'Native OpenSim constraint-aware target-coordinate moment arm. Not the total derivative along this weighted assembly probe, where other shoulder-girdle coordinates vary. No moment-arm validation claim for a human task.',
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_model_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'source_reference':'https://doi.org/10.3389/fnbot.2019.00090','source_variant':'https://github.com/ComputationalBiomechanicsLab/rmr-solver/tree/v1.1',
            'axes':'Native OpenSim shoulder frame: Y up. NOT the Z-up MyoSim world frame. Registration required before integration.','muscles':muscle_info,
            'integration_gates':['Source is a scaled unilateral shoulder model, not the 180cm/90kg full-body subject. Re-estimate segment scale, mass, COM and inertia before merging.','Replace the current arm and scapular couplers; do not append duplicate bones or double-count muscles.','Model lacks active wrist DOFs and full elbow extensor coverage (only triceps long head). Preserve/merge validated distal-arm elements deliberately.','Native elbow range ends at 130 degrees; current rack reaches about 140. Establish a supported rack posture or independently validate any range extension.','Source RMR model sets ignore_tendon_compliance=true. This audit does not add tendon elasticity.','Map native Y-up shoulder frame and left-right anatomy; validate attachments, moment arms and closed-loop constraint reactions.','Register hand/bar contact, shoulder-girdle position, GH stability and 2-hand/ground external loads, then rerun whole-body dynamics.','No physiological force, fatigue, injury or microdamage claims from this geometry probe.']}
    (dest/'assessment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SHOULDER_COMPONENT_CHECKED',len(muscles),'muscles',closure,'AC closure error',flush=True)


if __name__=='__main__': main()
