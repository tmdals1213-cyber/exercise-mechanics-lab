from pathlib import Path
import json,yt_dlp,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/standing_long_jump_reference';OUT.mkdir(parents=True,exist_ok=True)
REFS=[('ymca_analysis','https://www.youtube.com/watch?v=0kw9AYvD73Q'),('pass_knsu_technique','https://www.youtube.com/watch?v=H5g1dtgRL2k'),('koong_basic','https://www.youtube.com/watch?v=dO5QMg9Pb-M')]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');a=ap.parse_args();reports=[]
 for name,url in REFS:
  folder=OUT/name;folder.mkdir(exist_ok=True)
  try:
   options=dict(quiet=True,no_warnings=True,skip_download=not a.download,format='bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]',outtmpl=str(folder/'source.%(ext)s'),noplaylist=True,retries=1,socket_timeout=15,writeinfojson=True,js_runtimes={'node':{}})
   with yt_dlp.YoutubeDL(options)as y:info=y.extract_info(url,download=a.download)
   report=dict(key=name,url=url,title=info.get('title'),channel=info.get('channel'),duration_s=info.get('duration'),description=info.get('description'),chapters=info.get('chapters'),local_file=str(y.prepare_filename(info))if a.download else None,status='downloaded_for_internal_motion_reference'if a.download else'metadata_collected',publication_reuse=False);(folder/'reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');reports.append(report);print('REFERENCE',name,report['title'],report['duration_s'],report['status'],flush=True)
  except Exception as e:
   reports.append(dict(key=name,url=url,status='access_failed',error=str(e)));print('REFERENCE_ACCESS_FAILED',name,str(e)[:240],flush=True)
 (OUT/'collection.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
