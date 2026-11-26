"""Run analytical force/moment/energy checks before measured replay analysis."""
import json,numpy as np
from pathlib import Path
from verify_mechanics_foundation import analytical_checks,launch_fixture
from newton_euler_reference import body_wrenches,external_wrenches,at_point

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_cmj';G=9.80665

def main():
    OUT.mkdir(parents=True,exist_ok=True);tests=analytical_checks();launch,trace=launch_fixture();extra=[]
    def check(name,actual,expected,tol=1e-8):
        a=np.asarray(actual);e=np.asarray(expected);error=float(abs(a-e).max());extra.append(dict(name=name,actual=a.tolist(),expected=e.tolist(),max_absolute_error_SI=error,tolerance=tol,passed=error<tol))
    mass=1.3;support=1000.;arm=.05;COP=.2;com=.07;tendon=(support*COP-mass*G*com)/arm;expected_reaction=mass*G-support-tendon
    own=body_wrenches([mass],[[com,0,.06]],[np.eye(3)*.001],[[0,0,0]],[[0,0,0]],[[0,0,0]],[0,0,-G]);ext=external_wrenches(1,[(0,np.array([COP,0,0]),np.array([0,0,support]),np.zeros(3)),(0,np.array([-arm,0,.05]),np.array([0,0,tendon]),np.zeros(3))]);reaction=at_point(own[0]-ext[0],[0,0,.1]);check('foot_tendon_joint_reaction_force_and_moment',reaction,[0,0,expected_reaction,0,0,0]);check('foot_all_vertical_forces',expected_reaction+tendon+support-mass*G,0)
    # Opposing muscle endpoint forces are collinear: no whole-system force
    # or moment is created, even when their individual magnitudes are large.
    a=np.array([.2,.1,.5]);b=np.array([.3,.15,.9]);F=10000*(b-a)/np.linalg.norm(b-a);pair=external_wrenches(2,[(0,a,F,np.zeros(3)),(1,b,-F,np.zeros(3))]);check('internal_action_reaction_global_wrench',pair.sum(0),np.zeros(6))
    t=np.linspace(0,2,2001);u=t/2;z=.5*(10*u**3-15*u**4+6*u**5);v=.5/2*(30*u**2-60*u**3+30*u**4);acc=.5/4*(60*u-180*u**2+120*u**3);bar_mass=100.;hand_each=bar_mass*(G+acc)/2;P=2*hand_each*v;energy=.5*bar_mass*v*v+bar_mass*G*z
    check('barbell_hand_force_balance',2*hand_each-bar_mass*G,bar_mass*acc);check('constant_speed_100kg_bar_each_hand_N',bar_mass*G/2,490.3325);check('barbell_work_energy_J',np.trapezoid(P,t),energy[-1]-energy[0],.001)
    tests+=extra;assert all(x['passed'] for x in tests) and launch['finest_passed'];np.savez_compressed(OUT/'analytical_barbell.npz',times=t,height_m=z,velocity_m_s=v,acceleration_m_s2=acc,hand_each_N=hand_each,bar_energy_J=energy)
    report=dict(status='analytical_implementation_checks_passed',tests=tests,analytic_launch=launch,static_foot_fixture=dict(external_support_N=support,tendon_N=tendon,proximal_joint_reaction_vertical_N=expected_reaction,external_lever_m=COP,internal_lever_m=arm,foot_mass_kg=mass),scope='Analytical numerical fixtures only. Does not validate human muscle recruitment, anatomy, or joint compression. No physiological force ceiling.',human_muscle_forces_validated=False)
    (OUT/'basic_checks.json').write_text(json.dumps(report,indent=2));print('BASIC_CHECKS_COMPLETE',len(tests),'fixtures',flush=True)

if __name__=='__main__':main()
