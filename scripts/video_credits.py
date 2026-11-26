"""Compact English attribution cards inside the portrait safe area."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/power_comparison'
BG=(13,23,38);FG=(237,243,251);MUTED=(166,188,209);ACCENT=(140,226,213)
DURATION_S=3.;FPS=30


def make_card(key):
    im=Image.new('RGB',(1080,1920),BG);draw=ImageDraw.Draw(im)
    def line(y,s,size=24,color=FG,bold=False):
        f=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
        assert draw.textbbox((0,0),s,font=f)[2]<=972,('Credit line too wide',s)
        draw.text((54,y),s,font=f,fill=color)
    line(80,'MODEL & REFERENCES',36,FG,True)
    line(140,'Countermovement jump' if key=='cmj' else 'High-bar squat',25,MUTED)
    line(235,'CALCULATION MODEL',22,ACCENT,True)
    line(278,"Adapted MyoSim 'myofullbody' / MuJoCo",26)
    line(322,'MyoHub/myo_sim  |  custom rigid-body and force-sharing analysis',22,MUTED)
    line(374,'Shoulder paths: Seth et al. (2019), adapted via RMR',24)
    line(414,'doi:10.3389/fnbot.2019.00090',22,MUTED)
    line(510,'MOTION / FORCE REFERENCES',22,ACCENT,True)
    if key=='cmj':
        entries=[('Motion data: Calisti, Mohr & Federolf (2025)',
                  'doi:10.6084/m9.figshare.28890545  |  CC BY 4.0'),
                 ('CMJ tendon-force comparison: Wade et al. (2018)',
                  'doi:10.1038/s41598-018-20387-0')]
    else:
        entries=[('Muscle-force comparison: Goodman et al. (2023)',
                  'doi:10.1519/JSC.0000000000004506'),
                 ('Loaded-squat reference: Purzel et al. (2025)',
                  'doi:10.1111/sms.70058')]
    for y,(title,identifier) in zip([556,650],entries):
        line(y,title,24);line(y+38,identifier,22,MUTED)
    line(795,'ANATOMICAL SURFACES / ADAPTED',22,ACCENT,True)
    line(842,'BodyParts3D / DBCLS via MyoHub  |  CC BY 4.0',23)
    line(885,'Z-Anatomy: Kervyn, Zielinski & Vinent  |  CC BY-SA 4.0',23)
    line(928,'Supplementary BP3D / DBCLS  |  CC BY-SA 2.1 JP',23)
    line(971,'Registration, posing, display shape and colour modified.',21,MUTED)
    line(1090,'Model-based estimates; reference studies use different conditions.',23,MUTED)
    line(1140,'Full references, source links and asset notices accompany this video.',23,MUTED)
    assert np.all(np.asarray(im)[1536:]==BG)
    path=OUT/(key+'_credits.png');im.save(path)
    return im,dict(file=path.name,duration_s=DURATION_S,frames=round(DURATION_S*FPS),
        motion_statistics_include_credits=False,bottom_reserved_px=384,
        model="Adapted MyoSim myofullbody / MuJoCo; custom mechanics and conditional force sharing",
        full_notices='CREDITS.md',references='muscle_reference_review.html')


if __name__=='__main__':
    for key in ['cmj','squat_60','squat_100','squat_140']:
        _,metadata=make_card(key);print(json.dumps(metadata))
