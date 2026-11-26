"""Shared absolute-N colours and a deliberately restricted muscle overview."""
import json
import numpy as np

ROWS=[('Gluteals',['gluteals']),('Quadriceps',['quadriceps']),('Hamstrings',['hamstrings']),
      ('Calf',['gastrocnemius','soleus']),('Anterior shin',['tibialis_anterior']),
      ('Abdominals',['abdominals']),('Back / lumbar',['spinal_extensors','quadratus_lumborum']),
      ('Latissimus',['latissimus']),('Trapezius',['trapezius'])]
STOPS=np.log1p([0,100,1000,20000]);COLORS=np.array([[43,97,142],[48,163,169],[240,183,78],[238,82,69]])
MAX_N=20000.

def rgb(value):
    x=np.log1p(np.maximum(value,0));return np.stack([np.interp(x,STOPS,COLORS[:,i]) for i in range(3)],axis=-1).astype(np.uint8)

def joint_rgb(value):
    u=np.clip(np.asarray(value)/MAX_N,0,1)
    return np.stack([8+237*u,8+22*u,10+20*u],axis=-1).astype(np.uint8)

def overview(data):
    force=np.asarray(data['group_tension_N'])
    return np.array([force[[i for i,g in enumerate(data['groups']) if g['key'] in keys]].sum(0)/2 for _,keys in ROWS])

def load(root,key):
    path=root/'results/power_comparison/cmj_analysis/visualization/data.json' if key=='cmj' else root/f'results/power_comparison/{key}/analysis/visualization/data.json'
    if key=='cmj' and not path.exists():path=root/'results/measured_jump_visualization/data.json'
    return json.loads(path.read_text())
