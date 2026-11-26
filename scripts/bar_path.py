"""A smooth forward/back arc with analytic horizontal velocity/acceleration."""
import math

def forward_motion(t,progress,progress_velocity,progress_acceleration,config):
 a=config['bar_start_forward_m'];b=config.get('bar_face_clearance_bump_m',0.);T=config['ascent_s']
 u=max(0.,min(float(t)/T,1.));phase=math.pi*u
 value=a*(1-progress)**2+b*math.sin(phase)**2
 velocity=-2*a*(1-progress)*progress_velocity+b*math.pi/T*math.sin(2*phase)
 acceleration=2*a*(progress_velocity**2-(1-progress)*progress_acceleration)+b*2*math.pi**2/T**2*math.cos(2*phase)
 if t<=0 or t>=T:velocity=acceleration=0.
 return value,velocity,acceleration

def forward_offset(progress,config,time_s):
 return forward_motion(time_s,progress,0.,0.,config)[0]
