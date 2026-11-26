"""Analytical fixtures and independent segment-wrench verification.

Passing numerical tests is explicitly separate from human movement validity.
"""
import json,csv,hashlib,numpy as np,mujoco
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from newton_euler_reference import body_wrenches,external_wrenches,accumulate,at_point,energy
from mechanics_engine_adapter import sample,path_speeds
from jump_rig import JumpRig,ROOT
from rigid_baseline_profile import RigidBaselineRig,HELPERS
from verify_jump_reference import normal_weights
OUT=ROOT/'results/mechanics_foundation';G=9.80665

def analytical_checks():
    tests=[]
    def check(name,actual,expected,tol=1e-9):
        actual=np.asarray(actual);expected=np.asarray(expected);err=float(abs(actual-expected).max());tests.append({'name':name,'actual':actual.tolist(),'expected':expected.tolist(),'max_absolute_error_SI':err,'tolerance':tol,'passed':err<=tol})
    m=np.array([0.,5.]);com=np.array([[0,0,0],[.3,0,0]]);zero=np.zeros((2,3));I=np.array([np.zeros((3,3)),np.eye(3)*.01]);own=body_wrenches(m,com,I,zero,zero,zero,[0,0,-G]);w=accumulate([0,0],own)
    check('static_5kg_arm_0.3m_lever',w[1],[0,0,5*G,0,-5*G*.3,0])
    ext=external_wrenches(2,[(1,np.array([.6,0,0]),np.array([0,0,-100.]),np.zeros(3))]);w=accumulate([0,0],own-ext);check('static_arm_plus_100N_tip_load',w[1],[0,0,5*G+100,0,-5*G*.3-100*.6,0])
    masses=np.array([0.,3.,2.]);cs=np.array([[0,0,0],[.2,0,0],[.55,0,0]]);zs=np.zeros((3,3));Is=np.zeros((3,3,3));own=body_wrenches(masses,cs,Is,zs,zs,zs,[0,0,-G]);ext=external_wrenches(3,[(2,np.array([.7,0,0]),np.array([0,0,-100]),np.zeros(3))]);w=accumulate([0,0,1],own-ext)
    check('two_link_root_cut',w[1],[0,0,5*G+100,0,-3*G*.2-2*G*.55-100*.7,0]);check('two_link_distal_cut',at_point(w[2],[.4,0,0]),[0,0,2*G+100,0,-2*G*.15-100*.3,0])
    q=.6;om=1.3;aa=-.8;mass=2.;L=.7;r=np.array([-L/2*np.sin(q),0,-L/2*np.cos(q)]);omega=np.array([0,om,0]);alpha=np.array([0,aa,0]);a=np.cross(alpha,r)+np.cross(omega,np.cross(omega,r));I=np.diag([mass*L*L/12,mass*L*L/12,.001]);own=body_wrenches([mass],[r],[I],[a],[omega],[alpha],[0,0,-G]);check('dynamic_pendulum_tau',own[0,4],mass*L*L/3*aa+mass*G*L/2*np.sin(q))
    own=body_wrenches([7.],[np.array([.8,.2,1.])],[np.eye(3)],[np.array([0,0,-G])],[np.zeros(3)],[np.zeros(3)],[0,0,-G]);check('freefall_zero_required_wrench',own,np.zeros((1,6)))
    torque=100*G*.25;check('single_agonist_100kg_25cm_load_4cm_arm',torque/.04,6129.15625)
    return tests

def launch_fixture():
    mass=90.;T=.2;vto=1.5;z0=.8;A=np.pi/(2*T)*(vto+G*T/2);zto=z0+A*T*T/np.pi-G*T*T/6;end=T+2*vto/G;rows=[];trace=None
    def exact(t):
        if t<=T:
            a=A*np.sin(np.pi*t/T)-G*t/T;v=A*T/np.pi*(1-np.cos(np.pi*t/T))-G*t*t/(2*T);z=z0+A*T/np.pi*(t-T/np.pi*np.sin(np.pi*t/T))-G*t**3/(6*T);return z,v,mass*(a+G)
        s=t-T;return zto+vto*s-.5*G*s*s,vto-G*s,0.
    for hz in [300,600,1200,2400]:
        t=np.unique(np.r_[np.arange(0,end,1/hz),T,T+vto/G,end]);analytic=np.array([exact(float(x)) for x in t]);state=np.array([z0,0.]);numeric=[state.copy()]
        def fun(time,y):return np.array([y[1],exact(time)[2]/mass-G])
        for a,b in zip(t[:-1],t[1:]):
            h=b-a;k1=fun(a,state);k2=fun(a+h/2,state+h*k1/2);k3=fun(a+h/2,state+h*k2/2);k4=fun(b,state+h*k3);state=state+h*(k1+2*k2+2*k3+k4)/6;numeric.append(state.copy())
        numeric=np.array(numeric);F=analytic[:,2];impulse=float(np.trapezoid(F-mass*G,t));work=float(np.trapezoid(F*analytic[:,1],t));dE=.5*mass*analytic[-1,1]**2+mass*G*(analytic[-1,0]-z0);rows.append({'hz':hz,'dt_ms':1000/hz,'maximum_position_error_m':float(abs(numeric[:,0]-analytic[:,0]).max()),'maximum_speed_error_m_s':float(abs(numeric[:,1]-analytic[:,1]).max()),'impulse_momentum_error_Ns':abs(impulse-mass*(analytic[-1,1]-analytic[0,1])),'work_energy_error_J':abs(work-dE)})
        if hz==2400:trace=np.c_[t,analytic,numeric]
    return {'label':'Analytical point-mass launch and flight, no landing; calibration fixture, not a human jump prediction','mass_kg':mass,'takeoff_speed_m_s':vto,'rise_above_takeoff_m':vto*vto/(2*G),'takeoff_s':T,'resolution':rows,'finest_passed':bool(rows[-1]['maximum_position_error_m']<1e-7 and rows[-1]['impulse_momentum_error_Ns']<.01 and rows[-1]['work_energy_error_J']<.01)},trace

def inertia_audit(rig):
    m=rig.m;bad=[]
    for i in range(1,m.nbody):
        if m.body_mass[i]<=0:continue
        radius=float(np.sqrt(max(m.body_inertia[i])/m.body_mass[i]))
        if radius>rig.b.subject.height_m:bad.append({'body':m.body(i).name,'mass_kg':float(m.body_mass[i]),'principal_I_kg_m2':m.body_inertia[i].tolist(),'radius_of_gyration_m':radius})
    return bad

def main():
    OUT.mkdir(parents=True,exist_ok=True);tests=analytical_checks();launch,trace=launch_fixture();np.savetxt(OUT/'analytic_launch.csv',trace,delimiter=',',header='time_s,z_exact_m,v_exact_m_s,Fz_N,z_RK4_m,v_RK4_m_s',comments='')
    source=ROOT/'results/jump_validation/vertical/acceleration_2400hz';z=np.load(source/'motion.npz');loads=np.load(source/'loads.npz');defs=json.loads((source/'contact_definition.json').read_text('utf-8'));old=JumpRig();new=RigidBaselineRig();bad=inertia_audit(old);fixedbad=inertia_audit(new);snapshots=[];j=int(np.argmax(abs(loads['net_joint_moment_Nm'][:,old.b.ind_names.index('shoulder_elv_r')])));frames=sorted(set([0,int(np.argmin(abs(z['times']-1.3333333333))),j]+np.linspace(0,len(z['times'])-1,25).round().astype(int).tolist()))
    max_engine=0.;max_cut=0.;source_root=0.;corrected_root=0.;speed_bug=0.;max_correction=0.;snapshots=[]
    for index in frames:
        q=z['qpos'][index];v=z['qvel'][index];a=z['qacc'][index];contacts=[(p['body_id'],loads['contact_position_world_m'][index,k],loads['contact_force_world_N'][index,k],np.zeros(3)) for k,p in enumerate(defs['points'])];one={'time_s':float(z['times'][index])}
        for name,rig in [('legacy',old),('rigid_profile_fixed_historical_motion',new)]:
            state=sample(rig,q,v,a,contacts);B,_=rig.coordinates(q,z['u'][index]);reduced=B.T@state['generalized'];max_engine=max(max_engine,state['crosscheck_max']);max_cut=max(max_cut,state['cut_vs_virtual_work_max']);res=float(np.linalg.norm(state['subtree_wrenches'][0,3:]));source_root=max(source_root,res) if name=='legacy' else source_root;corrected_root=max(corrected_root,res) if name!='legacy' else corrected_root
            rec={'root_force_residual_world_N':state['subtree_wrenches'][0,:3].tolist(),'root_moment_residual_about_world_origin_Nm':state['subtree_wrenches'][0,3:].tolist(),'shoulder_elv_r_net_generalized_Nm':float(reduced[6+rig.b.ind_names.index('shoulder_elv_r')]),'engine_RNE_crosscheck_max_SI':state['crosscheck_max'],'recursive_cut_vs_virtual_work_max_SI':state['cut_vs_virtual_work_max'],'cuts':{n:state['cuts'][n] for n in ['ankle_angle_r','knee_angle_r','hip_flexion_r','shoulder_elv_r']}}
            if name=='legacy':
                k=6+rig.b.ind_names.index('shoulder_elv_r');helper_gen=np.zeros(rig.m.nv)
                for body in HELPERS:
                    bid=rig.m.body(body).id;J=np.zeros((3,rig.m.nv));Jr=J.copy();mujoco.mj_jacBodyCom(rig.m,rig.d,J,Jr,bid);w=state['body_wrenches'][bid];helper_gen+=J.T@w[:3]+Jr.T@(w[3:]-np.cross(rig.d.xipos[bid],w[:3]))
                rec['coordinate_helper_contribution_to_shoulder_elv_r_Nm']=float((B.T@helper_gen)[k]);speed_bug=max(speed_bug,float(abs(path_speeds(rig)-rig.d.actuator_velocity).max()))
            one[name]=rec
        max_correction=max(max_correction,abs(one['legacy']['shoulder_elv_r_net_generalized_Nm']-one['rigid_profile_fixed_historical_motion']['shoulder_elv_r_net_generalized_Nm']));snapshots.append(one)
    # Fixed-pose epsilon study isolates numerical helper parameters from anatomy.
    sensitivity=[];idx=j;q=z['qpos'][idx];v=z['qvel'][idx];a=z['qacc'][idx]
    for epsilon in [1e-4,1e-5,1e-6]:
        rig=RigidBaselineRig(helper_mass_kg=epsilon);state=sample(rig,q,v,a);B,_=rig.coordinates(q,z['u'][idx]);sensitivity.append({'helper_mass_kg':epsilon,'shoulder_elv_r_Nm':float((B.T@state['generalized'])[6+rig.b.ind_names.index('shoulder_elv_r')])})
    # Static whole-body contact is derived from mg and COM, not from jump input.
    new.b.pose(z['qpos'][0]);ps=np.array([new.b.jacobian(p['body_id'],np.array(p['local_m']))[1] for p in defs['points']]);total=float(new.m.body_mass.sum());COM=np.sum(new.m.body_mass[:,None]*new.d.xipos,axis=0)/total;weights=normal_weights(ps,COM[0],COM[1]);contacts=[(p['body_id'],ps[j],np.array([0,0,weights[j]*total*G]),np.zeros(3)) for j,p in enumerate(defs['points'])];static=sample(new,z['qpos'][0],np.zeros(new.m.nv),np.zeros(new.m.nv),contacts)
    standing={'mass_kg':total,'expected_vertical_support_N':total*G,'calculated_vertical_support_N':sum(float(c[2][2]) for c in contacts),'COP_xy_m':(weights@ps[:,:2]).tolist(),'COM_xy_m':COM[:2].tolist(),'minimum_contact_weight':float(weights.min()),'root_force_residual_N':float(np.linalg.norm(static['subtree_wrenches'][0,:3])),'root_moment_residual_Nm':float(np.linalg.norm(static['subtree_wrenches'][0,3:]))}
    idx=int(np.argmin(abs(z['times']-1.3333333333)));q=z['qpos'][idx];v=z['qvel'][idx];new.b.pose(q,v,z['qacc'][idx]);geometric=path_speeds(new);native=new.d.actuator_velocity.copy();h=1e-6;lengths=[]
    for sign in [-1,1]:
        qq=q.copy();mujoco.mj_integratePos(new.m,qq,v,sign*h);new.b.pose(qq);lengths.append(new.d.actuator_length.copy())
    fd=(lengths[1]-lengths[0])/(2*h);speedcheck={'time_s':float(z['times'][idx]),'central_difference_step_s':h,'Jacobian_vs_length_finite_difference_max_error_m_s':float(abs(fd-geometric).max()),'disabled_native_actuator_velocity_error_m_s':float(abs(fd-native).max()),'passed':bool(abs(fd-geometric).max()<1e-6)}
    with (OUT/'joint_load_snapshots.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['time_s','legacy_shoulder_elv_r_Nm','legacy_helper_contribution_Nm','rigid_profile_same_motion_shoulder_elv_r_Nm','rigid_profile_same_motion_root_moment_residual_norm_Nm'])
        for x in snapshots:writer.writerow([x['time_s'],x['legacy']['shoulder_elv_r_net_generalized_Nm'],x['legacy']['coordinate_helper_contribution_to_shoulder_elv_r_Nm'],x['rigid_profile_fixed_historical_motion']['shoulder_elv_r_net_generalized_Nm'],np.linalg.norm(x['rigid_profile_fixed_historical_motion']['root_moment_residual_about_world_origin_Nm'])])
    summary={'status':'numerical_foundation_checked_human_loads_not_released','units':'kg, m, s, N, Nm, rad','tolerances_are':'declared engineering numerical targets, not measured human prediction accuracy','analytical_tests':tests,'analytic_launch':launch,'legacy_inertial_outliers':bad,'corrected_inertial_outliers':fixedbad,'corrected_profile':new.profile,'engine_crosscheck_max_SI':max_engine,'recursive_cut_crosscheck_max_SI':max_cut,'legacy_root_moment_residual_norm_max_Nm':source_root,'corrected_profile_on_old_motion_root_moment_residual_norm_max_Nm':corrected_root,'maximum_fixed_motion_shoulder_torque_change_Nm':max_correction,'actuation_disabled_native_speed_error_m_s':speed_bug,'muscle_speed_fix':'Use the transmission Jacobian times qvel; do not read the zeroed actuator_velocity after disabling actuation. Muscle force re-release remains blocked.','epsilon_sensitivity_fixed_motion':sensitivity,'snapshots':snapshots,'gates':{'analytic_closed_forms':all(t['passed'] for t in tests),'analytic_launch_impulse_energy_convergence':launch['finest_passed'],'independent_NE_vs_engine':max_engine<1e-6,'recursive_joint_cuts_vs_virtual_work':max_cut<1e-6,'corrected_profile_inertial_extent_sanity':not fixedbad,'legacy_profile_inertial_extent_sanity':not bad,'corrected_profile_old_motion_is_valid':corrected_root<.1,'human_jump_validated':False,'muscle_force_comparison_released':False},'important':'The corrected-profile/old-motion comparison is parameter sensitivity only. It changes the angular-momentum requirements; new motion must be generated and independently audited.','input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'motion.npz',source/'loads.npz']}}
    summary['static_standing']=standing;summary['path_velocity_check']=speedcheck;summary['gates']['static_mg_COP_balance']=bool(max(standing['root_force_residual_N'],standing['root_moment_residual_Nm'])<1e-6);summary['gates']['path_Jacobian_velocity_vs_finite_difference']=speedcheck['passed']
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:summary[k] for k in ['gates','engine_crosscheck_max_SI','recursive_cut_crosscheck_max_SI','maximum_fixed_motion_shoulder_torque_change_Nm','path_velocity_check','epsilon_sensitivity_fixed_motion']}),flush=True)
    make_report(summary)

def make_report(s):
    gates=''.join('<tr><td>'+k+'</td><td class="'+('pass' if v else 'fail')+'">'+('PASS' if v else 'NOT ACCEPTED')+'</td></tr>' for k,v in s['gates'].items());cases=''.join(f'<tr><td>{r["name"]}</td><td>{r["max_absolute_error_SI"]:.3g}</td></tr>' for r in s['analytical_tests']);outliers=''.join(f'<tr><td>{r["body"]}</td><td>{r["mass_kg"]:.6g}</td><td>{r["principal_I_kg_m2"][0]:.6g}</td><td>{r["radius_of_gyration_m"]:.3f}</td></tr>' for r in s['legacy_inertial_outliers']);snap=max(s['snapshots'],key=lambda x:abs(x['legacy']['shoulder_elv_r_net_generalized_Nm']));o=snap['legacy'];n=snap['rigid_profile_fixed_historical_motion'];headline=f'Legacy shoulder demand {o["shoulder_elv_r_net_generalized_Nm"]:.1f} Nm; coordinate-helper contribution {o["coordinate_helper_contribution_to_shoulder_elv_r_Nm"]:.1f} Nm.'
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mechanics foundation verification</title><style>body{font:17px system-ui;background:#101c2b;color:#edf3fb;max-width:1120px;margin:40px auto;padding:0 20px;line-height:1.55}h1{font-size:32px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #344b60}.pass{color:#8ce2d5}.fail{color:#f1b486}.card{background:#1c3045;padding:24px;border-radius:12px;margin:24px 0}a{color:#8ce2d5}img{width:100%}</style><h1>Mechanics foundation / verification</h1><p>Numerical verification is separate from validation of human movement and muscle loading.</p><div class="card"><strong>Root cause confirmed in legacy model parameters</strong><p>'''+headline+'''</p><p>Eight coordinate-helper bodies each had about 1 gram of mass but about 1 kg m² of principal rotational inertia: a radius of gyration near 32 m. These are not suitable anatomical inertias.</p></div><h2>Acceptance gates</h2><table>'''+gates+'''</table><h2>Closed-form mechanics fixtures</h2><p>Pure NumPy Newton–Euler wrenches, static levers, a two-link cut, dynamic pendulum, free fall and a declared single-agonist calibration case.</p><table><tr><th>Case</th><th>Max error in SI component</th></tr>'''+cases+'''</table><h2>Analytical launch / impulse / energy</h2><p>90 kg point-mass calibration, 1.5 m/s takeoff; no landing or human joint/muscle inference.</p><img src="foundation_summary.png"><h2>Legacy inertial outliers</h2><table><tr><th>Body</th><th>Mass / kg</th><th>Ixx / kg m²</th><th>Gyration radius / m</th></tr>'''+outliers+'''</table><h2>Correction policy and limitations</h2><p>Coordinate helpers become point-like numerical bodies. Anatomical mass fractions are preserved and total mass remains 90 kg. Uncalibrated joint damping and spring terms are omitted. Numerical armature is explicitly declared. Epsilon sensitivity is checked.</p><p>Old motion is not automatically valid for the corrected inertia model. Corrected-model historical-motion loads are sensitivity results only. New trajectories need force/momentum, energy, contact, time-resolution and segment-cut checks before muscle-force allocation.</p><p>Disabling actuation also zeroed native actuator_velocity. Future muscle calculations must use path-Jacobian velocity instead. Existing muscle results remain rejected.</p><p><a href="verification.json">Full verification JSON</a> · <a href="joint_load_snapshots.csv">Joint-cut CSV</a> · <a href="analytic_launch.csv">Analytical fixture CSV</a></p></html>'''
    (OUT/'index.html').write_text(html,encoding='utf-8')
    # Standalone summary figure for the reproducible analytic fixture.
    data=np.loadtxt(OUT/'analytic_launch.csv',delimiter=',',skiprows=1);im=Image.new('RGB',(1400,1040),'#101c2b');draw=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n);draw.text((65,30),'MECHANICS FOUNDATION / ANALYTICAL FIXTURE',font=font(32),fill='#edf3fb');draw.text((65,82),'90 kg point mass / exact reference / no human-muscle inference',font=font(24),fill='#b5c7d8')
    for k,(title,y) in enumerate([('Vertical force / N',data[:,3]),('COM height / m',data[:,1]),('Speed / m/s',data[:,2])]):
        x0,x1=140,1330;y0=180+k*260;y1=y0+160;lo=min(0,float(y.min()));hi=max(.01,float(y.max()));span=hi-lo;X=lambda t:x0+(x1-x0)*t/data[-1,0];Y=lambda v:y1-(y1-y0)*(v-lo)/(span*1.05);draw.text((65,y0-42),title,font=font(26),fill='#edf3fb')
        for v in np.linspace(lo,hi,3):draw.line((x0,Y(v),x1,Y(v)),fill='#344b60');draw.text((x0-15,Y(v)),f'{v:.2f}',font=font(21),fill='#b5c7d8',anchor='rm')
        draw.line([(X(t),Y(v)) for t,v in zip(data[::3,0],y[::3])],fill='#8ce2d5',width=3);draw.line((X(.2),y0,X(.2),y1),fill='#f1b486');draw.text((X(.2)+7,y0),'TAKEOFF',font=font(18),fill='#f1b486')
        for t in [0,.1,.2,.3,.4,.5]:draw.text((X(t),y1+12),f'{t:.1f}s',font=font(21),fill='#b5c7d8',anchor='mt')
    finest=s['analytic_launch']['resolution'][-1];draw.text((65,985),f'2400 Hz errors: impulse {finest["impulse_momentum_error_Ns"]:.5g} Ns / energy {finest["work_energy_error_J"]:.5g} J',font=font(24),fill='#b5c7d8');im.save(OUT/'foundation_summary.png')
if __name__=='__main__':main()
