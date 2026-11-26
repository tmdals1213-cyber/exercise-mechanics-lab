"""Shared 37 pre-contact posture/timing + 15 equivalent effort coordinates."""
import copy
import numpy as np
from jump_parameter_rl_worker import FULL_SPECS,parameter_bounds
from jump_effort_control import EFFORT_STEMS

POSTCONTACT_ONLY={'landing_hip','landing_knee','landing_transition_s','landing_ankle','landing_trunk','landing_shoulder'}
POSTURE_SPECS=[(key,span,unit)for key,span,unit in FULL_SPECS if key not in POSTCONTACT_ONLY]

def definition(base,model,effort_span=.25):
    items=[]
    for key,span,unit in POSTURE_SPECS:
        lo,hi=parameter_bounds(model,key,span,unit,base[key])
        items.append(dict(key=key,kind='posture_timing',center=base[key],minimum=lo,maximum=hi,unit=unit))
    curves=base.get('propulsion_effort_knots_fraction',{})
    for stem in EFFORT_STEMS:
        for k,phase in enumerate([.25,.5,.75]):
            value=float(curves.get(stem,[0.,0.,0.])[k])
            items.append(dict(key=f'effort.{stem}.{k}',kind='equivalent_effort',stem=stem,knot=k,phase=phase,center=value,minimum=max(-.40,value-effort_span),maximum=min(.40,value+effort_span),unit='fraction_of_fixed_joint_torque_cap'))
    assert len(items)==52
    return items

def decode(base,items,action):
    action=np.asarray(action,float);assert action.shape==(len(items),)and np.isfinite(action).all()
    p=copy.deepcopy(base);p['propulsion_effort_knots_fraction']={stem:list(base.get('propulsion_effort_knots_fraction',{}).get(stem,[0.,0.,0.]))for stem in EFFORT_STEMS}
    for x,item in zip(np.clip(action,-1,1),items):
        c=item['center'];value=float(c+x*((item['maximum']-c)if x>=0 else(c-item['minimum'])))
        if item['kind']=='posture_timing':p[item['key']]=value
        else:p['propulsion_effort_knots_fraction'][item['stem']][item['knot']]=value
    return p
