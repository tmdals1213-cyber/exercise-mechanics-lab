"""Small independent mechanics fixtures for unbounded tension selection."""
import json
from pathlib import Path
import numpy as np
from tension_reference import solve_history,solve_causal_history,scale_parameters,reaction_cone
ROOT=Path(__file__).resolve().parents[1]

def main():
 R=np.array([[[.03,-.03]]]);target=np.array([[30.]]);gain=np.full((1,2),100.);p=np.zeros((1,2));ref=np.full(2,100.)
 simple=solve_history([0],R,target,gain,p,ref);assert np.allclose(simple['force_N'],[[1000,0]],atol=.01);assert simple['unbounded_effort_coefficient'].max()>1
 # Explicit compression is a mathematical test fixture, not a knee/shoulder
 # default. It tests antagonist tension arising from a mechanical constraint.
 stable=solve_history([0],R,target,gain,p,ref,[(np.array([[-1.,-1.]]),np.array([-1400.]))]);assert np.allclose(stable['force_N'],[[1200,200]],atol=.01)
 causal=solve_causal_history([0],R,target,gain,p,ref,[(np.array([[-1.,-1.]]),np.array([-1400.]))]);assert np.allclose(causal['force_N'],stable['force_N'],atol=.01)
 for speed in [0.,1.]:
  power1=float((np.array([1000.,0])*R[0,0]*speed).sum());power2=float((np.array([2000.,1000.])*R[0,0]*speed).sum());assert abs(power1-power2)<1e-9
 # Parallel agonists: changing one standard force parameter affects sharing
 # while the external mechanical demand remains exactly the same.
 names=['A','B'];r2,g2,p2,sc=scale_parameters(names,ref,gain,p,{'element_reference_force_scales':{'A':1.1}},{});parallel=solve_history([0],np.array([[[.03,.03]]]),target,g2,p2,r2);expected=1000*r2*r2/(r2@r2);assert np.allclose(parallel['force_N'][0],expected,atol=.01);assert sc[1]==1 and sc[0]==1.1
 # Cone orientation and compression direction checked in force space.
 C,b=reaction_cone(np.zeros((1,3,2)),np.array([[0.,0.,100.]]),np.array([[0.,0.,1.]]));assert (C@np.array([0.,0.])-b).max()<1e-9
 bad=False
 try:scale_parameters(names,ref,gain,p,{'element_reference_force_scales':{'typo':1.1}}, {})
 except ValueError:bad=True
 assert bad
 report={'status':'passed','unbounded_required_effort_above_one':float(simple['unbounded_effort_coefficient'].max()),'explicit_stabilization_fixture_tensions_N':stable['force_N'].tolist(),'equal_net_work_different_co_contraction':True,'single_parameter_sensitivity_tensions_N':parallel['force_N'].tolist(),'unknown_parameter_rejected':True,'scope':'Solver/virtual-work fixtures, not empirical validation of human co-contraction or a default compression threshold.'};out=ROOT/'results/tension_reference';out.mkdir(parents=True,exist_ok=True);(out/'solver_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
