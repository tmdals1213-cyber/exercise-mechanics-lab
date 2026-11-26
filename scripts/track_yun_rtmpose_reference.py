"""Independent 2D pose estimate with manually guided athlete bounding boxes."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rtmlib import RTMPose
from rtmlib.tools.file import download_checkpoint
ROOT=Path(__file__).resolve().parents[1]
MODEL='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip'
LINKS=[(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--start',type=float,default=157.5);parser.add_argument('--end',type=float,default=159.35);parser.add_argument('--output-name',default='rtmpose_full_body');args=parser.parse_args()
 cv2.setNumThreads(2);out=ROOT/'results/reference_box_jump/yun_user_video'/args.output_name;out.mkdir(exist_ok=True)
 model=download_checkpoint(MODEL,dst_dir=str(ROOT/'.tools/pose_reference/models'),progress=False)
 pose=RTMPose(model,model_input_size=(288,384),backend='onnxruntime',device='cpu');options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;pose.session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
 anchors=np.array([[157.5,535,240,645,525],[158.,545,245,690,525],[158.35,570,335,770,530],[158.5,580,330,785,530],[158.65,610,230,820,530],[158.85,625,180,835,415],[158.99,645,200,845,405],[159.13,675,200,900,405],[159.35,685,160,920,405],[159.6,735,100,930,445],[159.8,745,65,940,450]])
 mapping={11:5,12:6,13:7,14:8,15:9,16:10,23:11,24:12,25:13,26:14,27:15,28:16,29:24,30:25,31:20,32:21}
 cap=cv2.VideoCapture(str(ROOT/'datasets/box_jump_reference/yun_user_video/source.webm'));fps=cap.get(cv2.CAP_PROP_FPS);first=int(np.ceil(args.start*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,first);times=[];points=[];raw_points=[];raw_scores=[];bboxes=[];qc=[];k=first
 while True:
  ok,img=cap.read();t=k/fps
  if not ok or t>args.end:break
  bbox=np.array([np.interp(t,anchors[:,0],anchors[:,j])for j in range(1,5)]);kp,score=pose(img,bboxes=[bbox]);kp=kp[0];score=score[0];p=np.zeros((33,5));p[:,2]=np.nan
  for mp,halpe in mapping.items():p[mp,:2]=kp[halpe];p[mp,3:]=score[halpe]
  times.append(t);points.append(p);raw_points.append(kp);raw_scores.append(score);bboxes.append(bbox)
  if len(times)%4==1:qc.append((t,cv2.cvtColor(img,cv2.COLOR_BGR2RGB),p.copy()))
  if len(times)%15==0:print('RTMPOSE_REFERENCE',len(times),round(t,3),flush=True)
  k+=1
 cap.release();times=np.array(times);points=np.array(points)
 np.savez_compressed(out/'tracked_landmarks.npz',source_times_s=times,image_landmarks_xy_z_visibility_presence=points,raw_halpe26_keypoints_px=np.array(raw_points),raw_halpe26_scores=np.array(raw_scores),manual_bounding_boxes_px=np.array(bboxes),mapped_mediapipe_indices=np.array(list(mapping)))
 sheet=Image.new('RGB',(1600,((len(qc)+3)//4)*330),(13,23,38));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
 for cell,(t,pixels,p)in enumerate(qc):
  im=Image.fromarray(pixels);d=ImageDraw.Draw(im)
  for a,b in LINKS:d.line([tuple(p[a,:2]),tuple(p[b,:2])],fill=(0,245,220),width=3)
  im=im.crop((440,130,1060,700)).resize((400,300));x=(cell%4)*400;y=(cell//4)*330;sheet.paste(im,(x,y+30));draw.text((x+8,y+4),f'{t:.3f} s',font=font,fill='white')
 sheet.save(out/'athlete_tracking_review.png')
 report=dict(status='independent_pose_estimator_review_pending',source='https://www.youtube.com/watch?v=ngyj9rSEjLc&t=155s',estimator='RTMPose-X Halpe26 via RTMLib',code='https://github.com/Tau-J/rtmlib',model_url=MODEL,model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),frames=len(times),interval_s=[float(times[0]),float(times[-1])],manual_bounding_boxes=True,compatibility_array='Only mapped_mediapipe_indices are populated; other indices are zero placeholders with zero score. Z is unavailable.',metric_3D=False,force_analysis_allowed=False,physical_time_verified=False)
 (out/'tracking_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
