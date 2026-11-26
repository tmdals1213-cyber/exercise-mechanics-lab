"""Smooth internal joint-effort residuals, explicitly not muscle tensions."""
import numpy as np

EFFORT_STEMS=('hip_flexion','knee_angle','ankle_angle','flex_extension','shoulder_elv')
KNOT_PHASES=(0.,.25,.50,.75,1.)

def effort_curve(u,values):
    """C2-continuous quintic interpolation, zero outside push and at its ends."""
    if u<=0. or u>=1.:return 0.
    values=(0.,*values,0.);j=min(3,int(u*4));s=u*4-j
    s=s*s*s*(10+s*(-15+6*s))
    return float(values[j]+s*(values[j+1]-values[j]))

def effort_torques(names,limits,u,curves):
    values={stem:effort_curve(u,curves.get(stem,(0.,0.,0.)))for stem in EFFORT_STEMS}
    return np.array([values.get(name if name=='flex_extension'else name.rsplit('_',1)[0],0.)*limit for name,limit in zip(names,limits)])
