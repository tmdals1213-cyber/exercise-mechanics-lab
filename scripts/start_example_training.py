"""Start a small fresh PPO experiment using the supplied registered model."""
from pathlib import Path
import argparse,sys,subprocess,shutil
from jump_monitor_data import ROOT,read,atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=48);ap.add_argument('--hours',type=float,default=1.);ap.add_argument('--resume',action='store_true');args=ap.parse_args();assert args.episodes>=48
    run=ROOT/'results/controlled_broad_jump/landing_stability_v3'
    if(run/'definition.json').exists()and not args.resume:raise RuntimeError('Existing run found: use --resume, or choose a fresh checkout. No history overwritten.')
    folder=ROOT/'examples/long_jump';assert(folder/'registered_model.mjb.gz').exists(),'Run scripts/download_example.py first.';config=read(folder/'learning_definition.json');config.update(registered_example='examples/long_jump',initialization_complete=True,inherited_training_episodes=0,focus_context_one=False,adaptive_exploration=False);config.pop('original_budget_deadline',None);config.pop('previous_parameter_bounds_preserved_in',None)
    web=ROOT/'deliverables/jump_landing_live';web.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'scripts/jump_landing_live.html',web/'index.html');atomic(ROOT/'datasets/local_retention/protected_cases.json',dict(protected_case_directories=[],future_failed_raw_per_worker=24,keep_scores_and_history=True));definition=web/'example_definition.json';atomic(definition,config)
    command=[sys.executable,str(ROOT/'scripts/train_landing_stability.py'),'--label','landing_stability_v3','--episodes',str(args.episodes),'--hours',str(args.hours),'--workers','3'];command+=['--resume']if args.resume else['--definition-file',str(definition)];subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
