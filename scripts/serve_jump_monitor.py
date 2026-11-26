"""Loopback-only monitoring. Read-only HTTP; no remote actions or shell execution."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import argparse, datetime, json, mimetypes, threading, time,re
from jump_monitor_data import ROOT, MONITOR, RUN, read, atomic, lines

_cache={};_cache_lock=threading.Lock()
def appended_rows(path, fields=None):
    """Read only appended bytes; long overnight logs must not be reparsed every 3 s."""
    path=Path(path)
    with _cache_lock:
        try:size=path.stat().st_size
        except OSError:return []
        key=str(path);offset,pending,rows=_cache.get(key,(0,b'',[]))
        if size<offset:offset,pending,rows=0,b'',[]
        with path.open('rb')as file:
            file.seek(offset);chunk=file.read();offset=file.tell()
        parts=(pending+chunk).split(b'\n');pending=parts.pop()
        for line in parts:
            try:
                row=json.loads(line)
                if fields is not None:row={k:row.get(k)for k in fields}
                rows.append(row)
            except ValueError:pass
        _cache[key]=offset,pending,rows
        return list(rows)

def snapshot():
    state = read(RUN / 'status.json', {})
    label = state.get('search_label', 'rigid_landing_initial_v29')
    folder = ROOT / 'results/controlled_broad_jump' / label
    trial_fields=['trial','cost','stable','robust_stable','feasible','valid_distance_m','first_heel_m','end_z','late_max_speed','refinement']
    trials = appended_rows(folder / 'trials.jsonl',trial_fields)
    # Select an evenly spaced history, always including the newest record.
    stride = max(1, len(trials) // 1600)
    indices=set(range(0,len(trials),stride));record=-1
    for i,row in enumerate(trials):
        distance=row.get('valid_distance_m')if row.get('robust_stable')else None
        if distance is not None and distance>record:record=distance;indices.add(i)
    selected=[trials[i]for i in sorted(indices)]
    if trials and (not selected or selected[-1] != trials[-1]): selected.append(trials[-1])
    fields = ['trial', 'cost', 'stable', 'robust_stable', 'feasible', 'valid_distance_m', 'first_heel_m', 'end_z', 'late_max_speed']
    history = [{k: row.get(k) for k in fields} | {'fine_distance_m': (row.get('refinement') or {}).get('valid_distance_m')} for row in selected]
    train = ROOT / 'results/controlled_broad_jump' / state.get('training_label', 'ppo_full_overnight_v30')
    episodes = []
    for f in sorted(train.glob('worker_*/episodes.jsonl')):
        episodes.extend(dict(row, worker=f.parent.name) for row in appended_rows(f,['episode','kind','reward','stable','contact_distance_m','finished_at']))
    episodes.sort(key=lambda r: r.get('finished_at', 0))
    best = read(folder / 'best_candidate.json', {})
    now = time.time()
    return dict(now=now, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), state=state,
                heartbeat_age_s=now - state.get('heartbeat', now), search_label=label,
                search=read(folder / 'progress.json', {}), definition=read(folder / 'search_definition.json', {}),
                candidate={k: best.get(k) for k in fields}, history=history,recent_initial_trials=[{k:r.get(k)for k in fields}for r in trials[-100:]],
                training=read(train / 'live_training.json', {}),
                updates=appended_rows(train / 'updates.jsonl')[-600:],
                episodes=[{k: r.get(k) for k in ['episode', 'worker', 'kind', 'reward', 'stable', 'contact_distance_m', 'finished_at']} for r in episodes[-2000:]],
                verification=read(ROOT / 'results/controlled_broad_jump/two_metre_initial_solution_verified/verification.json'),review=read(RUN/'review.json'),
                events=lines(RUN / 'events.jsonl')[-30:],
                motion=read(MONITOR / 'motion.json'),
                model=dict(height_cm=178, mass_kg=87, policy_outputs=43, actor=[7,32,32,43], critic=[7,32,32,1],
                           integration='RK4', dt_ms=[.5,.25,.125], muscle_forces=False),
                video_ready=(ROOT / 'deliverables/controlled_jump_pilot/two_metre_initial_solution/fixed_side_multiview_review.mp4').exists())

def publish():
    while True:
        try:
            obj = snapshot()
            atomic(MONITOR / 'snapshot.json', obj)
            data = 'window.JUMP_SNAPSHOT=' + json.dumps(obj, ensure_ascii=False, allow_nan=False) + ';\nwindow.dispatchEvent(new Event("jump-snapshot"));'
            (MONITOR / 'snapshot.js.tmp').write_text(data, encoding='utf-8')
            (MONITOR / 'snapshot.js.tmp').replace(MONITOR / 'snapshot.js')
        except Exception as exc:
            print('MONITOR_SNAPSHOT_ERROR', repr(exc), flush=True)
        time.sleep(3)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/status':
            try:
                data = json.dumps(snapshot(), ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(500, str(exc))
            return
        assets = {'/': MONITOR / 'index.html', '/index.html': MONITOR / 'index.html',
                  '/snapshot.js': MONITOR / 'snapshot.js', '/snapshot.json': MONITOR / 'snapshot.json',
                  '/video.mp4': ROOT / 'deliverables/controlled_jump_pilot/two_metre_initial_solution/fixed_side_multiview_review.mp4',
                  '/review.html': ROOT/'deliverables/jump_optimization_review/2026-11-25/index.html',
                  '/effort.html': ROOT/'deliverables/standing_long_jump_effort_seed/index.html',
                  '/policy_initialization.json': ROOT/'deliverables/standing_long_jump_effort_seed/policy_initialization.json',
                  '/effort_launch_policy_initial.zip': ROOT/'deliverables/standing_long_jump_effort_seed/effort_launch_policy_initial.zip',
                  '/force_seed_summary.json': ROOT/'deliverables/standing_long_jump_effort_seed/force_seed_summary.json',
                  '/effort_anatomy.mp4': ROOT/'deliverables/controlled_jump_pilot/effort_seed_anatomy_v1/fixed_side_multiview_review.mp4',
                  '/effort_anatomy_cover.png': ROOT/'deliverables/controlled_jump_pilot/effort_seed_anatomy_v1/fixed_side_cover.png',
                  '/ai_concept.html': ROOT/'deliverables/jump_ai_single_view_concept/index.html',
                  '/ai_single_view.mp4': ROOT/'deliverables/jump_ai_single_view_concept/single_view_layout_preview.mp4',
                  '/ai_single_view.png': ROOT/'deliverables/jump_ai_single_view_concept/layout_preview.png',
                  '/storyboard.json': ROOT/'deliverables/jump_ai_single_view_concept/storyboard.json',
                  '/inference_evidence.json': ROOT/'deliverables/jump_ai_single_view_concept/inference_evidence.json'}
        review=ROOT/'deliverables/jump_distance_stability_review'
        assets.update({'/distance_stability.html':review/'index.html','/distance_stability.mp4':review/'distance_stability_learning.mp4','/distance_stability.png':review/'fall.png'})
        assets.update({'/distance_stability/'+name:review/name for name in ['evidence.json','pilot_summary.json','updates.json','pareto.json','audit.json','delivery_check.json']})
        live=ROOT/'deliverables/jump_landing_live'
        assets['/landing_live.html']=live/'index.html'
        controller=ROOT/'deliverables/landing_controller_audit'
        assets.update({'/controller_audit.html':controller/'index.html','/controller_audit/controller_comparison.png':controller/'controller_comparison.png','/controller_audit/summary.json':controller/'summary.json'})
        search_audit=ROOT/'deliverables/jump_search_audit'
        assets.update({'/search_audit.html':search_audit/'index.html','/search_audit/search_comparison.png':search_audit/'search_comparison.png','/search_audit/summary.json':search_audit/'summary.json'})
        draft=ROOT/'deliverables/jump_rl_study_draft_v1'
        assets.update({'/rl_draft.html':draft/'index.html','/rl_draft.mp4':draft/'standing_long_jump_rl_draft.mp4'})
        assets.update({'/rl_draft/'+name:draft/name for name in ['preview_hook.png','preview_checked.png','preview_comparison.png','timeline.json','metrics.json','source_manifest.json','draft_delivery.json','delivery_check.json','caption_en.txt','ANATOMY_CREDITS.md']})
        draft_v2=ROOT/'deliverables/jump_rl_study_draft_v2'
        assets.update({'/rl_draft_v2.html':draft_v2/'index.html','/rl_draft_v2.mp4':draft_v2/'standing_long_jump_rl_draft_v2.mp4'})
        assets.update({'/rl_draft_v2/'+name:draft_v2/name for name in ['preview_hook.png','preview_control.png','preview_brain.png','preview_update.png','preview_ghosts.png','preview_final_motion.png','preview_four_stages.png','timeline.json','metrics.json','source_manifest.json','draft_delivery.json','delivery_check.json','caption_en.txt','ANATOMY_CREDITS.md']})
        draft_v3=ROOT/'deliverables/jump_rl_study_draft_v3'
        assets.update({'/rl_draft_v3.html':draft_v3/'index.html','/rl_draft_v3.mp4':draft_v3/'standing_long_jump_rl_draft_v3.mp4'})
        assets.update({'/rl_draft_v3/'+name:draft_v3/name for name in ['preview_control.png','preview_parameters_air.png','preview_learning.png','preview_persistent_ghosts.png','preview_multiview.png','preview_summary.png','parameter_catalogue.json','muscle_analysis.json','metrics.json','source_manifest.json','training_compatibility.json','source_check.json','delivery_check.json','draft_delivery.json','ANATOMY_CREDITS.md']})
        draft_v4=ROOT/'deliverables/jump_rl_study_draft_v4'
        assets.update({'/rl_draft_v4.html':draft_v4/'index.html','/rl_draft_v4.mp4':draft_v4/'standing_long_jump_rl_draft_v4.mp4'})
        assets.update({'/rl_draft_v4/'+name:draft_v4/name for name in ['preview_three_views.png','preview_trajectory.png','preview_fullbody.png','preview_four_simultaneous.png','parameter_catalogue.json','muscle_analysis.json','metrics.json','source_manifest.json','training_compatibility.json','source_check.json','prefix_check.json','torque_usage.json','delivery_check.json','draft_delivery.json','ANATOMY_CREDITS.md']})
        draft_v5=ROOT/'deliverables/jump_rl_study_draft_v5'
        assets.update({'/rl_draft_v5.html':draft_v5/'index.html','/rl_draft_v5.mp4':draft_v5/'standing_long_jump_rl_final.mp4'})
        assets.update({'/rl_draft_v5/'+name:draft_v5/name for name in ['preview_four_side.png','preview_four_oblique.png','finalization_status.json','trial_provenance.json','delivery_check.json','caption_en.txt']})
        assets.update({'/landing_live/'+name:live/name for name in ['status.json','latest_motion.json','best_stable_motion.json','best_distance_motion.json','verified_motion.json','range_expansion_status.json','finish_rule_status.json','robust_search_status.json','robust_continuation_status.json','robust_launch_status.json']})
        match=re.fullmatch(r'/landing_live/episode/(?:(landing_stability_v[23])_)?worker_(\d{2})_episode_(\d{6})',path)
        if match and int(match.group(2))<3:
            assets[path]=ROOT/'results/controlled_broad_jump'/(match.group(1)or'landing_stability_v2')/('worker_'+match.group(2))/('episode_'+match.group(3))/'motion.json'
        file = assets.get(path)
        if file is None or not file.is_file(): self.send_error(404); return
        # Video byte ranges let a browser seek without loading the entire render.
        size=file.stat().st_size; start=0; end=size-1; code=200
        if self.headers.get('Range','').startswith('bytes='):
            try:
                a,b=self.headers['Range'][6:].split('-'); start=int(a or 0); end=min(int(b)if b else end,end)
                if not 0<=start<=end<size: raise ValueError()
                code=206
            except ValueError: self.send_error(416); return
        self.send_response(code); self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Cache-Control','no-store'); self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with file.open('rb')as f:
                f.seek(start); remaining=end-start+1
                while remaining:
                    chunk=f.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError): pass

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--port',type=int,default=8765); args=ap.parse_args()
    MONITOR.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    threading.Thread(target=publish,daemon=True).start()
    print(f'JUMP_MONITOR http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
