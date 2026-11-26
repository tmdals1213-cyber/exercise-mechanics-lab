"""Foot approach path: gain box height before crossing its rear face."""
from centroidal_jump import Polynomial


class BoxFlightCurve:
    def __init__(self,start,velocity,end,duration,rear_y=-.18):
        self.duration=duration;self.end=end.copy()
        self.horizontal=Polynomial(start,velocity,0,end,0,0,duration)
        self.tuck_time=.55*duration
        wait_y=rear_y+.245+.035
        self.y_before=Polynomial(start[1],velocity[1],0,wait_y,0,0,self.tuck_time)
        self.y_after=Polynomial(wait_y,0,0,end[1],0,0,duration-self.tuck_time)
        high=end[2]+.065
        self.z_before=Polynomial(start[2],velocity[2],0,high,0,0,self.tuck_time)
        self.z_after=Polynomial(high,0,0,end[2],-.35,0,duration-self.tuck_time)

    def value(self,t,derivative=0):
        value=self.horizontal.value(t,derivative)
        before=t<self.tuck_time;tt=t if before else t-self.tuck_time
        value[1]=(self.y_before if before else self.y_after).value(tt,derivative)[0]
        value[2]=(self.z_before if before else self.z_after).value(tt,derivative)[0]
        return value
