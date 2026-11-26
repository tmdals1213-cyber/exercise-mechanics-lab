"""English copy generated from the selected force histories, not assumed winners."""
from pathlib import Path
import json
import numpy as np
from launch_sources import analysis_path,load
from case_contours import overview

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/instagram_launch'

def main():
    bio='AI agents + biomechanics\nMuscle tension & joint loads, visualized\nModel-based movement analysis\n'
    intro=('Using AI agents and established biomechanics models to explore how muscles and joints share the load during exercise.\n\n'
           'Visualizing how movement, speed and body proportions change the forces involved, with the assumptions and references included.\n')
    exposure=json.loads((OUT/'tension_exposure.json').read_text())['cases'];labels=['Quads','Glutes','Hamstrings','Calves'];peaks={}
    for key in ['cmj','squat_100']:
        v=load(ROOT,key);tt=np.array(v['times']);w=exposure[key];mask=(tt>=w['start_s'])&(tt<=w['end_s'])
        peaks[key]=overview(v)[[1,0,2,3]][:,mask].max(1)
    peak_lines=[];total_lines=[]
    for i,label in enumerate(labels):
        j,s=peaks['cmj'][i],peaks['squat_100'][i];winner='jump' if j>s else 'squat'
        peak_lines.append(f'{label}: {winner} +{100*(max(j,s)/min(j,s)-1):.0f}%')
        j=exposure['cmj']['group_integral_mean_per_side_Ns'][i];s=exposure['squat_100']['group_integral_mean_per_side_Ns'][i]
        total_lines.append(f'{label}: {"jump" if j>s else "squat"} {max(j,s)/min(j,s):.2f}x')
    caption=('Can a bodyweight jump rival a 100 kg squat?\n\n'
        '30 cm jump. 100 kg high-bar squat. Same 180 cm, 88 kg model.\n\n'
        'PEAK tension\n'+' | '.join(peak_lines)+'\n\n'
        'TOTAL tension x time\n'+' | '.join(total_lines)+'\n\n'
        'The result depends on the muscle and the metric. A short peak and sustained loading tell different stories.\n\n'
        f'TOTAL = integral of tension over physical time, not time-averaged force. One movement: jump {exposure["cmj"]["duration_s"]:.2f} s before takeoff; squat {exposure["squat_100"]["duration_s"]:.2f} s, including a 0.90 s ascent. Replays do not add to the total.\n\n'
        '30 cm means center-of-mass rise from takeoff to apex, not jump-and-reach or box height.\n\n'
        'AI-agent-assisted analysis using adapted MyoSim/MuJoCo mechanics and reference motion. The jump was prescribed and recomputed, not newly measured. Projected foot contacts and a disclosed rotational residual remain approximations.\n\n'
        'Conditional muscle-tendon tension (active + passive), mean per side. Not measured activation, energy, fatigue or growth. Muscle-force accuracy is not independently validated.\n\n'
        'Models/data: MyoHub/myo_sim; Seth 2019 (10.3389/fnbot.2019.00090); Calisti 2025 (10.1038/s41597-025-05934-5). Height reference: Kjaer 2016 (10.1186/s13102-016-0059-4). Force context: Goodman 2023; Wade 2018.\n\n'
        'Anatomy adapted: BP3D/DBCLS via MyoHub, CC BY 4.0; Z-Anatomy/Kervyn, Zielinski & Vinent, CC BY-SA 4.0; supplementary BP3D, CC BY-SA 2.1 JP.\n\n'
        '#Biomechanics #CountermovementJump #Squat #MusculoskeletalModel\n')
    assert len(bio.strip())<=150 and len(caption.strip())<=2200,(len(bio),len(caption))
    for name,value in [('profile_bio_en.txt',bio),('intro_post_en.txt',intro),('reel_caption_en.txt',caption)]:
        assert value.isascii();(OUT/name).write_text(value,encoding='utf-8')
    audit=json.loads((analysis_path(ROOT,'cmj')/'verification.json').read_text());c=audit['checks']
    methods=('HEIGHT / MODEL SCOPE\n\n'
        f'Airborne COM rise: {100*c["airborne_COM_height_m"]:.3f} cm.\n'
        f'Quiet standing to apex COM rise: {100*c["standing_to_apex_COM_rise_m"]:.3f} cm.\n'
        f'Estimated toe-tip clearance at COM apex: {100*c["toe_tip_clearance_at_COM_apex_m"]:.3f} cm, dependent on foot/knee pose.\n'
        f'Takeoff speed: {c["takeoff_vertical_velocity_m_s"]:.6f} m/s. Peak GRF: {c["peak_GRF_N"]:.3f} N.\n\n'
        '30cm is an illustrative target informed by the force-platform, hands-on-hips male sample mean of 29.3cm (SD 7.2cm), ages 20-64.9, Kjaer et al. 2016. It is not a universal average. Their wall jump-and-reach mean (38.0cm, SD 8.5cm) used a different protocol.\n'
        'https://doi.org/10.1186/s13102-016-0059-4\n\n'
        'The original Calisti s10_CMJ_t1 posture path was retimed during propulsion. COM propulsion was prescribed smoothly for the new launch speed. Flight uses gravity and conserved angular momentum. All muscle tensions were recalculated; source force amplitudes were not simply multiplied.\n\n'
        'Support forces are solved within projected foot regions with unilateral contact and Coulomb friction (mu=0.8). This is a prescribed inverse-dynamics illustration, not an exact sticking-contact forward simulation. Remaining rotational imbalance is not represented as an invented support torque.\n'
        f'Stance rotational residual RMS: {c["stance_moment_RMS_Nm"]:.2f} Nm ({100*c["stance_moment_RMS_BW_height"]:.2f}% BW x height); maximum: {c["stance_moment_max_Nm"]:.2f} Nm ({100*c["stance_moment_max_BW_height"]:.2f}% BW x height).\n'
        f'Force-weighted contact projection gap RMS: {1000*c["force_weighted_projected_contact_gap_RMS_m"]:.2f} mm. Strict foot sticking is not claimed.\n'
        f'250 vs 500Hz peak GRF difference: {100*c["sample_rate_peak_GRF_relative_difference"]:.4f}%.\n\n'
        'Muscle forces are conditional on generic anatomy, passive curves and force sharing. These mechanics checks do not establish 10% human muscle-force accuracy. Rectus abdominis passive tension is not evidence of continuous voluntary activation.\n'
        'TOTAL integrates scalar muscle-tendon tension over one physical movement (Ns); MEAN divides by that movement duration (N). Neither is work or metabolic energy. Three display replays are counted once.\n')
    (OUT/'jump_height_method_review.txt').write_text(methods,encoding='utf-8')
    print('ENGLISH_COPY_READY',len(caption.strip()),flush=True)

if __name__=='__main__':main()
