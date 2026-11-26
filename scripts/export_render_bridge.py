"""Physical poses + force channels for an independent presentation renderer.

No mesh is used to compute force. New beauty rigs must be registered to this
packet; this file does not certify that an arbitrary mesh fits the analysis rig.
"""
import json,hashlib
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    for grip in [65,80]:
        dest=ROOT/f'results/model_foundation/grip_{grip}'
        source=ROOT/f'results/grip_comparison/grip_{grip}/cycle/physiology'
        z=np.load(dest/'fullbody_motion.npz'); m=json.loads((source/'fullbody_metadata.json').read_text('utf-8'))
        scale=m['physics_length_scale']; rotations=z['body_xmat'].reshape(len(z['times']),-1,3,3)
        positions=z['body_xpos']*scale; rest_position=z['rest_body_xpos']*scale; rest_rotation=z['rest_body_xmat'].reshape(-1,3,3)
        # Delta maps an anatomical reference point already in physical world
        # coordinates into the current pose. New render rigs use these frames
        # with their own registered bind transforms, not artist geometry as physics.
        delta_rotation=rotations@np.swapaxes(rest_rotation,-1,-2)[None]
        delta_translation=positions-np.einsum('tbij,bj->tbi',delta_rotation,rest_position)
        error=float(abs(np.swapaxes(rotations,-1,-2)@rotations-np.eye(3)).max()); assert error<1e-8
        np.savez_compressed(dest/'render_packet.npz',times=z['times'],body_position_world_m=positions,body_rotation_world=rotations,rest_body_position_world_m=rest_position,rest_body_rotation_world=rest_rotation,
                            delta_rotation_world=delta_rotation,delta_translation_world_m=delta_translation,force_N=z['force_N'],active_force_N=z['active_force_N'],passive_force_N=z['passive_force_N'],reference_force_parameter_N=m['reference_force_N'])
        mapping=json.loads((source/'atlas_mapping.json').read_text('utf-8'))
        for row in mapping:
            row['force_representation']='mapped_group' if row['actuator_indices'] else 'unknown_no_force_representation'
        report={'schema':'exercise-mechanics.render-bridge/1.0','body_names':m['body_names'],'actuator_names':m['muscle_names'],'surface_mapping':mapping,
                'packet_sha256':hashlib.sha256((dest/'render_packet.npz').read_bytes()).hexdigest(),'source_motion_sha256':hashlib.sha256((dest/'fullbody_motion.npz').read_bytes()).hexdigest(),
                'coordinate_frame':'Native right-handed MyoSim world, Z up. All packet positions are physical meters, rotations are 3x3 row-major arrays.',
                'reference_delta':'p_current = delta_rotation_world @ p_reference + delta_translation_world_m, for points rigidly bound to one analysis body. Skinning is the renderer responsibility.',
                'force_mapping':'Mapped surfaces may share actuator groups; never sum force over surface meshes to obtain anatomical muscle or body totals. Empty mapping means unknown, not zero force.',
                'color_options':['total force N','active force N','passive force N','force / declared reference parameter (unbounded, not activation or percent capacity)'],
                'registration_checks_required':['hand-to-bar and foot-to-ground anchors','proximal humerus latissimus insertion','shoulder-girdle and muscle attachment registration','new surface penetrations / silhouettes in front, side and back','major anatomical group coverage including unactuated groups'],
                'rotation_orthogonality_error':error,'new_beauty_surface_registered':False,'render_frame_rate_controls_physics':False,'raw_peak_source':'telemetry.npz, not resampled image frames'}
        (dest/'render_bridge.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('RENDER_BRIDGE_READY',grip,len(m['body_names']),'bodies',flush=True)


if __name__=='__main__': main()
