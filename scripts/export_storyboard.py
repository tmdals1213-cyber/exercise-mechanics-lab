"""Optional storyboard PNG; requires matplotlib. Baseline itself is dependency-free."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/ohp_100kg_2s'
data = json.loads((OUT/'motion.json').read_text(encoding='utf-8'))
config, frames = data['config'], data['frames']
bg, panel, muted, gold, cyan = '#10151d', '#19212c', '#a1aec0', '#efca87', '#62e4d0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#edf2f8', 'axes.labelcolor': muted,
                     'xtick.color': muted, 'ytick.color': muted, 'axes.edgecolor': '#405166'})
fig = plt.figure(figsize=(16,9), facecolor=bg)
grid = fig.add_gridspec(2,4,height_ratios=[2.2,1],left=.06,right=.95,bottom=.12,top=.82,hspace=.4,wspace=.3)
fig.text(.06,.945,'EXERCISE MECHANICS / SCENARIO 001',color=cyan,size=12)
fig.text(.06,.89,'100 kg OHP  |  180 cm / 90 kg  |  2-second ascent',size=24,weight='bold')
for column,time in enumerate([0,.5,1.,2.]):
    f=min(frames,key=lambda frame:abs(frame['time_s']-time))
    ax=fig.add_subplot(grid[0,column],facecolor=panel)
    ax.set(xlim=(-.45,.6),ylim=(0,2.4),aspect='equal',xticks=[],yticks=[0,1,2])
    ax.grid(axis='y',alpha=.12)
    ax.set_title(f'{time:.1f} s  /  '+('LOCKOUT' if time==2 else 'START' if time==0 else 'CONSTANT SPEED'),fontsize=11,pad=14)
    def stroke(points,color,width):
        ax.plot([p[2] for p in points],[p[1] for p in points],color=color,lw=width,solid_capstyle='round')
    arm=f['arms'][1]
    stroke([[0,.08,0],[0,.5,0],[0,.94,0],arm['shoulder']],'#5e7187',10)
    stroke([[0,.08,0],[0,.04,.14]],'#5e7187',9)
    stroke([arm['shoulder'],[0,1.62,0]],'#5e7187',9)
    ax.add_patch(Circle((0,1.7),.1,color='#5e7187'))
    stroke([arm['shoulder'],arm['elbow'],arm['wrist'],arm['grip']],gold,6)
    for key in ('shoulder','elbow','wrist'):
        ax.scatter(arm[key][2],arm[key][1],s=30,color='#fff1d6',zorder=4)
    ax.add_patch(Circle((f['bar_z_m'],f['bar_y_m']),.19,color='#627d97',zorder=5))
    ax.add_patch(Circle((f['bar_z_m'],f['bar_y_m']),.025,color='#d9e8f2',zorder=6))
    ax.text(.03,.03,f"height {f['bar_y_m']:.2f} m\nvertical speed {f['bar_vy_m_s']:.2f} m/s",transform=ax.transAxes,color=muted,fontsize=10)
for panel_index,(key,label,color) in enumerate([
    ('bar_vy_m_s','Bar vertical speed (m/s)',cyan),
    ('bar_support_y_N','Total vertical support force (N)',gold)]):
    ax=fig.add_subplot(grid[1,panel_index*2:panel_index*2+2],facecolor=panel)
    ax.plot([f['time_s'] for f in frames],[f[key] for f in frames],color=color,lw=2)
    ax.axvspan(.15,1.85,color=cyan,alpha=.06)
    ax.set(xlabel='Time (s)',ylabel=label,xlim=(0,2.5))
    ax.grid(alpha=.12)
fig.text(.06,.045,'Schematic geometry + bar loading only. Muscle forces, tissue stress, fatigue and damage: NOT COMPUTED.',color=muted,size=11)
fig.savefig(OUT/'storyboard.png',dpi=130,facecolor=bg)
print(OUT/'storyboard.png')
