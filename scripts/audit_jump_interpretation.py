"""Audit the jump's interpretation, not just its equation residuals.

Force sums, net joint power and COM energy rate are different quantities.
Constraint ablations diagnose sensitivity; they are not replacement physiology.
"""
import json,csv,hashlib,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from tension_reference import solve_causal_history,reaction_cone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tension_visualization'

def integrate_window(t,y,a,b):
    sel=(t>a)&(t<b);tt=np.r_[a,t[sel],b];yy=np.r_[np.interp(a,t,y),y[sel],np.interp(b,t,y)];return float(np.trapezoid(yy,tt))

def peak(t,y):
    i=int(np.argmax(abs(y)));return {'time_s':float(t[i]),'value':float(y[i])}

def main():
    p=ROOT/'results/jump_validation/vertical/acceleration_2400hz';motion=dict(np.load(p/'motion.npz'));loads=dict(np.load(p/'loads.npz'));setup=json.loads((p/'setup.json').read_text('utf-8'));data=json.loads((OUT/'data.json').read_text('utf-8'));ops=dict(np.load(OUT/'operators.npz'));raw=dict(np.load(OUT/'standard.npz'));names=setup['coordinate_names'][6:];t=motion['times'];vel=motion['u'][:,6:];acc=motion['udot'][:,6:];tau=loads['net_joint_moment_Nm'];cp=tau*vel;tt=raw['times']
    region_indices={'trunk':[i for i,n in enumerate(names) if n in ['flex_extension','lat_bending','axial_rotation']], 'arms':[i for i,n in enumerate(names) if any(n.startswith(k) for k in ['elv_angle','shoulder_elv','shoulder_rot','elbow','pro_sup','deviation','flexion_'])], 'legs':[i for i,n in enumerate(names) if n.startswith(('hip_','knee_','ankle_','subtalar_','mtp_'))]}
    assert sorted(i for ids in region_indices.values() for i in ids)==list(range(len(names)))
    power={r:cp[:,ids].sum(axis=1) for r,ids in region_indices.items()};positive={r:np.maximum(cp[:,ids],0).sum(axis=1) for r,ids in region_indices.items()};negative={r:np.minimum(cp[:,ids],0).sum(axis=1) for r,ids in region_indices.items()}
    force=loads['contact_force_world_N'].sum(axis=1);com_rate=(force*motion['velocity']).sum(axis=1)
    phases={'ground_before_takeoff':(0,setup['takeoff_s']),'propulsion':(.85,setup['takeoff_s']),'flight':(setup['takeoff_s'],setup['landing_s']),'landing_and_recovery':(setup['landing_s'],float(t[-1]))}
    report={'status':'interpretation_rejected_for_human_load_comparison','numerical_balance_is_not_movement_validation':True,'motion_generated_by_muscles_in_forward_dynamics':False,'ground_reaction_is_prescribed_input':True,'muscle_forces_from_length_only':False,'peak_net_region_power_W':{r:peak(t,v) for r,v in power.items()},'peak_COM_mechanical_energy_rate_W':peak(t,com_rate),'phase_work_J':{},'coordinate_peaks':[],'ablations':{},'back_coverage':{'force_paths_missing':['Trapezius','Serratus anterior','Rhomboids','Levator scapulae'],'parameter_review':['Latissimus'],'modeled':['Lumbar spinal extensors','Quadratus lumborum'],'neutral_color_means_zero':False},'work_definition':'Generalized-coordinate tau*qdot. Positive work sums positive coordinate powers; not muscle metabolic work or a causal attribution of jump height. F_GRF dot v_COM is COM translational+gravitational energy rate, not work at a stationary foot contact.'}
    for phase,(a,b) in phases.items():report['phase_work_J'][phase]={r:{'net':integrate_window(t,power[r],a,b),'positive_coordinate_sum':integrate_window(t,positive[r],a,b),'negative_coordinate_sum':integrate_window(t,negative[r],a,b)} for r in power}
    for i,n in enumerate(names):report['coordinate_peaks'].append({'coordinate':n,'speed_rad_s':peak(t,vel[:,i]),'acceleration_rad_s2':peak(t,acc[:,i]),'moment_Nm':peak(t,tau[:,i]),'power_W':peak(t,cp[:,i])})
    report['coordinate_peaks'].sort(key=lambda v:-abs(v['moment_Nm']['value']))
    upper={'deltoid','pectoralis','triceps','biceps','rotator_cuff','arm_accessory','forearm','hand_intrinsics'};lower={'gluteals','quadriceps','hamstrings','gastrocnemius','soleus','hip_flexors','adductors','hip_accessory','tibialis_anterior','tibialis_posterior','fibularis','toe_flexors','toe_extensors'}
    ix={r:sorted(i for g in data['groups'] if g['key'] in keys for i in g['indices']) for r,keys in [('arms',upper),('legs',lower)]};ref=ops['reference'];stability=[reaction_cone(S,o,a) for S,o,a in zip(ops['S'],ops['offset'],ops['axes'])]
    cases={'current':raw,'no_GH_direction_cone':None,'no_temporal_regularizer':None,'no_GH_or_temporal_regularizer':None}
    for case in cases:
        if case!='current':
            try:cases[case]=solve_causal_history(tt,ops['R'],ops['target'],ops['gain'],ops['passive'],ref,None if 'no_GH' in case else stability,0 if 'temporal' in case else .01,1.)
            except RuntimeError as e:report['ablations'][case]={'failed':str(e)};continue
        z=cases[case];F=z['force_N'];P=z['passive_force_N'];item={}
        for r,ids in ix.items():
            s=F[:,ids].sum(axis=1);j=int(np.argmax(s));item[r]={'peak_sum_tension_N':float(s[j]),'peak_time_s':float(tt[j]),'passive_sum_at_peak_N':float(P[j,ids].sum()),'mean_sum_tension_N':float(np.trapezoid(s,tt)/(tt[-1]-tt[0]))}
        report['ablations'][case]=item;print('ABLATION',case,json.dumps(item),flush=True)
    report['ablations_are_diagnostics_not_replacement_models']=True
    report['input_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [p/'motion.npz',p/'loads.npz',OUT/'operators.npz',OUT/'standard.npz',OUT/'data.json']}
    (OUT/'interpretation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'interpretation_timeseries.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['time_s','ground_vertical_force_N','COM_energy_rate_W','legs_net_generalized_power_W','arms_net_generalized_power_W','trunk_net_generalized_power_W']);writer.writerows(zip(t,force[:,2],com_rate,power['legs'],power['arms'],power['trunk']))
    # Standalone diagnostic figure, shared seconds and signed kW axes.
    im=Image.new('RGB',(1500,1120),'#101c2b');draw=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
    draw.text((75,28),'JUMP AUDIT / REFERENCE MOTION REJECTED',font=font(35),fill='#f1c18d');draw.text((75,78),'Equation consistency does not establish realistic movement or muscle recruitment.',font=font(24),fill='#b5c7d8')
    charts=[('Vertical ground reaction / kN',[('GRF',force[:,2]/1000,'#8ce2d5')]),('Net generalized power / kW',[('Legs',power['legs']/1000,'#8ce2d5'),('Arms',power['arms']/1000,'#efac78'),('Trunk',power['trunk']/1000,'#a8afff')]),('COM translational + gravitational energy rate / kW',[('COM',com_rate/1000,'#84bdf4')])]
    for k,(title,series) in enumerate(charts):
        x0,x1=150,1420;y0=195+k*285;y1=y0+185;vals=np.concatenate([s for _,s,_ in series]);lo=min(0,float(vals.min()));hi=max(.1,float(vals.max()));pad=.08*(hi-lo);lo-=pad;hi+=pad;X=lambda s:x0+(x1-x0)*s/t[-1];Y=lambda v:y1-(y1-y0)*(v-lo)/(hi-lo)
        draw.text((75,y0-48),title,font=font(26),fill='#edf3fb')
        for v in np.linspace(lo,hi,4):draw.line((x0,Y(v),x1,Y(v)),fill='#344b60');draw.text((x0-12,Y(v)),f'{v:.1f}',font=font(21),fill='#b5c7d8',anchor='rm')
        for label,s,color in series:
            ids=np.unique(np.r_[np.arange(0,len(t),3),int(np.argmax(s)),int(np.argmin(s))]);draw.line([(X(float(t[i])),Y(float(s[i]))) for i in ids],fill=color,width=2)
        for event,label in [(setup['takeoff_s'],'TO'),(setup['landing_s'],'LAND')]:draw.line((X(event),y0,X(event),y1),fill='#b5c7d8',width=1);draw.text((X(event)+5,y0),label,font=font(18),fill='#b5c7d8')
        for s in [0,.5,1,1.5,2,2.5]:draw.text((X(s),y1+10),f'{s:.1f}s',font=font(20),fill='#b5c7d8',anchor='mt')
        px=900
        for label,_,color in series:draw.text((px,y0-42),label,font=font(23),fill=color);px+=160
    draw.text((75,1068),'Internal muscle tension (N) is not net joint output (W) or a share of ground force.',font=font(24),fill='#b5c7d8');im.save(OUT/'interpretation_audit.png')
    print('INTERPRETATION_AUDIT',json.dumps({k:report[k] for k in ['peak_net_region_power_W','peak_COM_mechanical_energy_rate_W','phase_work_J']}),flush=True)
if __name__=='__main__':main()
