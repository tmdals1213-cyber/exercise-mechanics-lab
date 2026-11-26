"""Portrait learning story from real PPO checkpoints and physical rollouts."""
from pathlib import Path
from functools import lru_cache
import argparse,json,math
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from build_motion_study_reel import font,BG,FG,MUTED,ACC

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables/standing_long_jump_learning';CONTROL=ROOT/'deliverables/controlled_jump_pilot'
BLUE=(96,183,221);GREEN=(124,210,165);DIM=(39,58,77)
def ease(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
def mix(a,b,u):return tuple(round(x+(y-x)*u)for x,y in zip(a,b))
def txt(d,x,y,s,size=24,c=FG,b=True):d.text((x,y),s,font=font(size,b),fill=c)

class Story:
 def __init__(self):
  self.e=json.loads((OUT/'learning_evidence.json').read_text());self.c=self.e['checkpoints'];self.rows=self.e['episodes'];self.labels=['LANDING TIME','HIP / LANDING','KNEE / LANDING','ARMS / LANDING','TRUNK / LANDING','HIP / FLIGHT','KNEE / FLIGHT','ARMS / FLIGHT']
  self.cases=['rl_batch1_example','rl_best'];self.r=[max(u['batch'],key=lambda r:r['reward'])for u in self.e['updates'][:2]]
  self.z=[np.load(CONTROL/p/'motion.npz')for p in self.cases];self.rep=[json.loads((CONTROL/p/'control_summary.json').read_text())for p in self.cases];self.cam=[json.loads((CONTROL/p/'fixed_side_camera_v2.json').read_text())for p in self.cases]
  self.pos=[[(x,250+j*430/(n-1))for j in range(n)]for x,n in zip([110,385,675,960],[7,32,32,8])]

 def graph(self,im,checkpoint,progress=0,backward=False,after=None,blend=0):
  d=ImageDraw.Draw(im);s=self.c[checkpoint];weights=[np.array(w)for w in s['weights']];acts=[np.array(v)for v in s['activations']]
  if after is not None:weights=[w+(np.array(v)-w)*blend for w,v in zip(weights,self.c[after]['weights'])];acts=[v+(np.array(w)-v)*blend for v,w in zip(acts,self.c[after]['activations'])]
  for k,w in enumerate(weights):
   # Show every node, and strongest connections per receiving neuron.
   for j in range(w.shape[0]):
    for h in np.argsort(abs(w[j]))[-2:]:
     a=self.pos[k][h];b=self.pos[k+1][j];mag=min(1,abs(w[j,h])*2);col=mix(DIM,BLUE if w[j,h]>=0 else ACC,mag*.45);d.line((*a,*b),fill=col,width=1)
     p=progress*3-(2-k if backward else k)
     if 0<=p<=1:
      if backward:p=1-p
      x=a[0]+(b[0]-a[0])*p;y=a[1]+(b[1]-a[1])*p;d.ellipse((x-2,y-2,x+2,y+2),fill=ACC if backward else BLUE)
  for layer,values in zip(self.pos,acts):
   scale=max(.05,float(abs(values).max()))
   for (x,y),v in zip(layer,values):
    c=mix(DIM,BLUE if v>=0 else ACC,min(1,abs(float(v))/scale));d.ellipse((x-6,y-6,x+6,y+6),fill=c,outline=MUTED,width=1)
  for x,label in zip([55,330,620,905],['STATE 7','TANH 32','TANH 32','ACTION 8']):txt(d,x,210,label,18,MUTED)
  d.rounded_rectangle((45,716,1035,766),radius=14,fill=(23,39,57));txt(d,65,729,'CRITIC  7 > 32 > 32 > 1',18,MUTED);txt(d,615,729,f"V(s)  {s['critic_value']:+.3f}",19,BLUE)

 @lru_cache(maxsize=12)
 def body(self,case,fi):
  with Image.open(CONTROL/self.cases[case]/'fixed_side_render_v2'/f'frame_{fi:04d}.png')as src:return src.convert('RGBA').crop((0,190,1080,980)).resize((740,541),Image.Resampling.LANCZOS)

 def jump(self,im,case,t):
  d=ImageDraw.Draw(im);z=self.z[case];rep=self.rep[case];cam=self.cam[case];T=float(z['times'][-1]);fi=min(round(t/T*144),144);body=self.body(case,fi);im.paste(body,(170,775),body);i=np.argmin(abs(z['times']-t));ppm=740/cam['ortho_width_m'];centre=cam['camera_center_forward_m'];toe=cam['initial_foremost_sole_y_m'];cz=cam['camera_center_z_m']
  def p(v):return(540+(-v[1]+toe-centre)*ppm,775+(525-(v[2]-cz)*(1080/cam['ortho_width_m'])-190)*740/1080)
  if t>=rep['touchdown_s']:
   ids=np.flatnonzero((z['times']>=rep['takeoff_s'])&(z['times']<=rep['touchdown_s']));n=max(1,round(ease((t-rep['touchdown_s'])/.65)*len(ids)));path=[p(z['com'][j])for j in ids[:n]]
   if len(path)>1:d.line(path,fill=ACC,width=3)
  x,y=p(z['com'][i]);d.ellipse((x-4,y-4,x+4,y+4),fill=ACC);txt(d,45,810,'PHYSICS',21,MUTED);txt(d,45,841,f'TRIAL {self.r[case]["training_trial"]:02}',21,ACC)

 def actions(self,d,values,y=1290,heading='SAMPLED PARAMETERS'):
  txt(d,45,y,heading,20,MUTED)
  for k,(label,value)in enumerate(zip(self.labels,values)):
   x=45+(k//4)*530;yy=y+36+(k%4)*35;txt(d,x,yy,label,16,MUTED);a=x+260;d.line((a,yy+10,a+190,yy+10),fill=DIM,width=5);d.line((a+95,yy+4,a+95,yy+16),fill=MUTED,width=1);end=a+95+float(np.clip(value,-1,1))*95;d.line((a+95,yy+10,end,yy+10),fill=BLUE if value>=0 else ACC,width=5)

 def cycle(self,case,t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);txt(d,45,25,'@biomechanics_ksm',20,MUTED);txt(d,45,66,f'LEARNING CYCLE {case+1}',38);txt(d,770,81,f'BATCH OF 8',22,MUTED)
  if t<1:phase='01  INPUT STATE'
  elif t<2:phase='02  FORWARD PASS'
  elif t<6.8:phase='03  SIMULATE JUMP'
  elif t<7.7:phase='04  REWARD'
  elif t<8.7:phase='05  BACKPROPAGATION'
  elif t<9.4:phase='06  ADAM UPDATE'
  else:phase='07  NEXT OUTPUT'
  txt(d,45,135,phase,31,ACC);back=7.7<=t<8.7;u=ease((t-8.7)/.7);progress=((t-7.7)if back else (t-1))%1;self.graph(im,case,progress,back,case+1,u)
  if t<1:
   txt(d,65,866,'BODY + TAKEOFF',42);txt(d,65,938,'178 cm / 87 kg',29,MUTED);txt(d,65,990,'1.29 m/s forward',29,MUTED);txt(d,65,1042,'1.47 m/s upward',29,MUTED);txt(d,65,1150,'Fixed across every trial',24,BLUE)
  elif t<7.7:self.jump(im,case,float(np.clip(t-2,0,4.8)))
  else:
   update=self.e['updates'][case];txt(d,70,850,'8 PHYSICAL JUMPS',35);txt(d,70,925,'PPO CLIPPED OBJECTIVE',25,MUTED);txt(d,70,991,'REWARD + CRITIC > ADVANTAGE',24,MUTED)
   if back:
    ev=update['backprop_evidence'];txt(d,70,1080,f"GRADIENT L2   {ev['gradient_L2']:.3f}",30,ACC);txt(d,70,1140,'Loss > gradients > parameters',23,MUTED)
   else:txt(d,70,1080,f"WEIGHT CHANGE   {update['weight_delta_L2']:.4f}",30,BLUE);txt(d,70,1140,'Updated Gaussian policy',24,MUTED)
  self.actions(d,self.r[case]['action']if t<9.4 else self.c[case+1]['activations'][-1],1250,'NEXT POLICY MEAN'if t>=9.4 else'SAMPLED PARAMETERS')
  if 6.8<=t<7.7:txt(d,550,1150,f"+{self.r[case]['reward']:.3f}",51,GREEN);txt(d,550,1205,'VALID LANDING',23,GREEN)
  txt(d,45,1478,'PPO / LANDING PARAMETERS',19,MUTED);txt(d,775,1478,'FIXED TAKEOFF',19,MUTED)
  return im

 def montage(self,t):
  im=Image.new('RGB',(1080,1920),BG);d=ImageDraw.Draw(im);trial=min(32,1+int(t/4*32));r=self.rows[trial-1];checkpoint=min(3,(trial-1)//8);txt(d,45,25,'@biomechanics_ksm',20,MUTED);txt(d,45,66,'TRY > LEARN > UPDATE',38);txt(d,45,134,f'TRIAL {trial:02} / 32',32,ACC);self.graph(im,checkpoint,(t*4)%1,t%1>.75)
  self.actions(d,r['action'],810);txt(d,45,1030,'REWARD',21,MUTED);txt(d,45,1068,f"{r['reward']:+.3f}",46,GREEN if r['stable']else ACC);txt(d,620,1088,'VALID LANDING'if r['stable']else'UNSTABLE LANDING',23,GREEN if r['stable']else ACC)
  # Actual rewards and retained best. No synthetic trial body replays.
  xs=np.linspace(55,1025,32);ys=[float(v['reward'])for v in self.rows];lo=min(-.1,min(ys));hi=max(ys)+.05
  def yy(v):return 1410-(v-lo)/(hi-lo)*220
  d.line((45,1410,1035,1410),fill=DIM,width=2);best=self.e['baseline_distance_m'];path=[]
  for j in range(trial):
   v=self.rows[j];y=yy(v['reward']);d.ellipse((xs[j]-4,y-4,xs[j]+4,y+4),fill=BLUE if v['stable']else ACC)
   if v['stable']:best=max(best,v['contact_distance_m'])
   path.append((xs[j],yy(best)))
  if len(path)>1:d.line(path,fill=ACC,width=3)
  txt(d,45,1450,f'BEST FOUND  {best:.3f} m',29,ACC);txt(d,735,1457,'4 PPO BATCHES',19,MUTED);return im

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');a=ap.parse_args();s=Story()
 for name,im in [('forward',s.cycle(0,1.55)),('simulation',s.cycle(0,4)),('backprop',s.cycle(0,8.15)),('updated',s.cycle(1,9.7)),('montage',s.montage(3.9))]:im.save(OUT/f'{name}_preview.png')
 if a.preview:return
 out=OUT/'standing_long_jump_learning.mp4';w=imageio_ffmpeg.write_frames(str(out),(1080,1920),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18','-preset','medium','-threads','4','-movflags','+faststart']);w.send(None);count=0
 def send(im):
  nonlocal count
  ImageDraw.Draw(im).rectangle((0,1536,1079,1919),fill=BG);arr=np.asarray(im);assert np.all(arr[1536:]==BG);w.send(arr);count+=1
 try:
  for case in range(2):
   for k in range(300):send(s.cycle(case,k/30))
  for k in range(120):send(s.montage(k/30))
  # Final actual best jump: fixed side plus both tracked views.
  source=CONTROL/'rl_best/fixed_side_multiview_review.mp4';reader=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24');meta=next(reader);last=None
  for k,raw in enumerate(reader):
   if k>=144:break
   im=Image.frombytes('RGB',meta['size'],raw);d=ImageDraw.Draw(im);d.rectangle((0,0,1080,140),fill=BG);txt(d,45,25,'BEST FOUND / 32 TRIALS',35);txt(d,45,78,f"{s.e['best']['contact_distance_m']:.3f} m   /   +{s.e['best_gain_percent']:.1f}%",30,ACC);d.rectangle((0,1344,1080,1535),fill=BG);txt(d,45,1370,'STARTER',20,MUTED);txt(d,45,1405,f"{s.e['baseline_distance_m']:.3f} m",34);txt(d,430,1370,'BEST SAMPLED',20,MUTED);txt(d,430,1405,f"{s.e['best']['contact_distance_m']:.3f} m",34,ACC);txt(d,805,1370,'POLICY MEAN',20,MUTED);txt(d,805,1405,f"{s.e['policy_evaluations'][-1]['policy_mean_distance_m']:.3f} m",34,BLUE);txt(d,45,1480,'LANDING ONLY / SAME TAKEOFF',20,MUTED);last=im;send(im)
  reader.close()
  for k in range(60):send(last)
 finally:w.close()
 (OUT/'story_manifest.json').write_text(json.dumps(dict(video=out.name,frames=count,duration_s=count/30,slow_real_updates=2,training_episodes=32,checkpoints=[0,8,16,24,32],backward_replay_checkpoint_error=[u['backprop_evidence']['saved_checkpoint_max_weight_error']for u in s.e['updates'][:2]],network=[7,32,32,8],reward='Nearest actual foot contact distance after stable landing',scope=s.e['scope'],flow_pulses='Schematic; node activation and checkpoint weights are recorded values.',bottom_reserved_px=384),indent=2));print('LEARNING_STORY_READY',out,flush=True)
if __name__=='__main__':main()
