"""Launch-only source selection; leaves earlier comparison studies untouched."""
import json

def selection(root):
    p=root/'deliverables/instagram_launch/launch_case_selection.json'
    return json.loads(p.read_text()) if p.exists() else {}

def analysis_path(root,key):
    defaults={'cmj':'results/power_comparison/cmj_analysis','squat_100':'results/power_comparison/squat_100/analysis'}
    return root/selection(root).get('analysis',defaults).get(key,defaults[key])

def render_root(root):
    return root/'deliverables/instagram_launch'/selection(root).get('render_cases','render_cases')

def load(root,key):
    return json.loads((analysis_path(root,key)/'visualization/data.json').read_text())
