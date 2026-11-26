"""Check provenance, source processing and independent replay calculations."""
import json,hashlib,numpy as np
from pathlib import Path
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/measured_cmj'
def main():
    prov=json.loads((ROOT/'datasets/cmj_calisti2025/provenance.json').read_text());basic=json.loads((OUT/'basic_checks.json').read_text());summary=json.loads((OUT/'summary.json').read_text());inputs=json.loads((OUT/'input_checks.json').read_text());source=ROOT/'datasets/cmj_calisti2025/source'
    for f in prov['source_files']:assert hashlib.sha256((source/f['file']).read_bytes()).hexdigest()==f['sha256'],f['file']
    assert len(basic['tests'])==13 and all(t['passed'] for t in basic['tests']) and basic['analytic_launch']['finest_passed']
    R=np.array(inputs['rotation_raw_to_opensim']);assert np.max(abs(R@R.T-np.eye(3)))<1e-10 and abs(np.linalg.det(R)-1)<1e-10
    for r in summary['trials']:
        trial=r['trial'];z=np.load(OUT/trial/'mechanics_replay.npz');raw=np.load(OUT/f'{trial}_measured.npz');assert len(z['times'])==len(raw['times']);assert np.max(abs(z['times']-raw['times']))<1e-9
        assert r['independent_NE_vs_OpenSim_max_SI']<1e-5 and r['velocity_adapter_max_error']<1e-8
        for key in ['COM_m','COM_acceleration_m_s2','net_generalized_load','calf_moment_arm_m']:assert np.isfinite(z[key]).all(),(trial,key)
        valid=raw['contact'];assert np.isfinite(raw['COP_m'][valid]).all();assert np.max(abs(raw['applied_ground_force_N'][valid]-raw['ground_force_filtered_N'][valid]))<1e-10;assert np.max(abs(raw['applied_ground_force_N'][~valid]))==0
        p=next(p for p in inputs['reports'] if p['trial']==trial);assert p['force_moment_reconstruction_error_Nm']<1e-8
    assert not summary['muscle_force_validation_passed'] and not summary['physiological_prediction_ready'] and not inputs['measured_force_magnitudes_rescaled']
    class Links(HTMLParser):
        links=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if tag in ['a','img','script']:
                x=d.get('href') or d.get('src')
                if x and not x.startswith(('http','#')):self.links.append(x)
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:assert (OUT/link).exists(),link
    result=dict(status='pipeline_checks_passed_empirical_validation_incomplete',analytical_fixtures=13,measured_trials=3,provenance_files=len(prov['source_files']),source_hashes_match=True,coordinate_transform_proper=True,force_amplitudes_unchanged=True,independent_NE_max_error_SI=max(r['independent_NE_vs_OpenSim_max_SI'] for r in summary['trials']),local_links=len(parser.links),human_muscle_forces_validated=False)
    (OUT/'pipeline_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
