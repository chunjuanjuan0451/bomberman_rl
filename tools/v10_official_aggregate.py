"""Create the immutable s103010 official paired aggregate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--directory',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite aggregate')
    rows=[]
    for seed in range(104401,104411):
        item={'seed':seed}
        for arm in ('baseline','candidate'):
            p=args.directory/f'v10-s103010-{arm}-{seed}.json'; d=json.loads(p.read_text())
            item[arm]={'run_id':d['run_id'],'status':d['status'],'exit_code':d['exit_code'],'metrics':d['metrics_by_agent']['model_a_v10'],'manifest':str(p)}
        rows.append(item)
    def agg(arm):
        vals=[x[arm]['metrics'] for x in rows]; scores=[x['score'] for x in vals]; steps=[x['steps'] for x in vals]
        return {'mean_score':float(np.mean(scores)),'median_score':float(np.median(scores)),'min_score':int(min(scores)),'max_score':int(max(scores)),'full_score_count':int(sum(x==50 for x in scores)),'step_limit_count':int(sum(x==400 for x in steps)),'mean_steps':float(np.mean(steps)),'survival_rate':float(np.mean([x['rounds']>0 and x['suicides']==0 for x in vals])),'invalid_actions':int(sum(x['invalid_actions'] for x in vals)),'mean_decision_time_ms':float(np.mean([x['mean_decision_time_ms'] for x in vals])),'max_mean_decision_time_ms':float(max(x['mean_decision_time_ms'] for x in vals))}
    base,cand=agg('baseline'),agg('candidate'); wins=sum(x['candidate']['metrics']['score']>x['baseline']['metrics']['score'] for x in rows); ties=sum(x['candidate']['metrics']['score']==x['baseline']['metrics']['score'] for x in rows); losses=10-wins-ties
    per_seed_drop=max(x['baseline']['metrics']['score']-x['candidate']['metrics']['score'] for x in rows)
    gate={'passed':bool(cand['mean_score']>=base['mean_score'] and cand['step_limit_count']<=base['step_limit_count'] and cand['mean_steps']<=base['mean_steps'] and per_seed_drop<=5 and cand['survival_rate']>=base['survival_rate'] and cand['invalid_actions']<=base['invalid_actions'] and losses<=1 and any(x['candidate']['metrics']['score']!=x['baseline']['metrics']['score'] or x['candidate']['metrics']['steps']!=x['baseline']['metrics']['steps'] for x in rows) and cand['max_mean_decision_time_ms']<450 and all(x['exit_code']==0 for x in [x['baseline'] for x in rows]+[x['candidate'] for x in rows])),'paired_wins':wins,'paired_ties':ties,'paired_losses':losses,'max_per_seed_score_drop':per_seed_drop,'failure_reasons':[]}
    checks=[('candidate_mean_score_lt_baseline',cand['mean_score']<base['mean_score']),('candidate_step_limit_count_gt_baseline',cand['step_limit_count']>base['step_limit_count']),('candidate_mean_steps_gt_baseline',cand['mean_steps']>base['mean_steps']),('max_per_seed_score_drop_gt_5',per_seed_drop>5),('candidate_survival_lt_baseline',cand['survival_rate']<base['survival_rate']),('candidate_invalid_actions_gt_baseline',cand['invalid_actions']>base['invalid_actions']),('paired_losses_gt_1',losses>1),('no_action_outcome_difference',not any(x['candidate']['metrics']['score']!=x['baseline']['metrics']['score'] or x['candidate']['metrics']['steps']!=x['baseline']['metrics']['steps'] for x in rows)),('candidate_latency_ge_450ms',cand['max_mean_decision_time_ms']>=450),('nonzero_exit_code',not all(x['exit_code']==0 for x in [x['baseline'] for x in rows]+[x['candidate'] for x in rows]))]
    gate['failure_reasons']=[name for name,bad in checks if bad]
    report={'kind':'v10-phase3.3-policy-official','status':'completed','run_id':'model-a-v10-nonlinear-policy-official-s103010','aggregates':{'baseline':base,'candidate':cand},'paired':rows,'gate':gate,'checkpoint_sha256':{'value':digest('experiments/checkpoints/model-a-v10-centered-blend-coin-heaven-s103004.npz'),'policy':digest('experiments/checkpoints/model-a-v10-nonlinear-policy-distill-s103009.pt')},'config_sha256':{'baseline':digest('experiments/configs/v10-phase3.3-policy-s103009-baseline.json'),'candidate':digest('experiments/configs/v10-phase3.3-policy-s103009-candidate.json')}}
    args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); tmp.replace(args.output); print(json.dumps({'output':str(args.output),'gate':gate,'aggregates':report['aggregates']},indent=2))
if __name__=='__main__': main()
