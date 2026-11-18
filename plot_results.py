"""Create a real-metrics comparison figure from an experiments.py results JSON."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
 data=json.loads(a.results.read_text()); models=data['models']; names=list(models)
 fig,axs=plt.subplots(2,3,figsize=(14,8));x=np.arange(len(names))
 series=[('cv','observation_macro_f1','Grouped CV observation Macro F1'),
         ('student_test','top1_accuracy','Student-test Top-1 accuracy'),
         ('student_test','macro_f1','Student-test Macro F1'),
         ('student_test','candidate_log_loss','Student-test candidate Log Loss'),
         ('student_test','candidate_miss_rate','Student-test candidate miss rate'),
         ('student_test','route_continuity','Student-test route continuity')]
 for ax,(section,key,title) in zip(axs.flat,series):
  values=[models[n][section][key] for n in names]
  ax.bar(x,values,color=['#4c78a8','#f58518','#54a24b','#e45756','#72b7b2'][:len(names)])
  ax.set_xticks(x,names,rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
  if key!='candidate_log_loss': ax.set_ylim(0,1)
 fig.suptitle('PathSnap model comparison from recorded experiments');fig.tight_layout()
 out=a.results.parent/'comparison_metrics.png';fig.savefig(out,dpi=170);plt.close(fig);print(out)
 csv_path=a.results.parent/'comparison_detailed.csv'
 with csv_path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['model','algorithm_family','cv_top1','cv_observation_macro_f1',
    'cv_candidate_macro_f1','cv_candidate_log_loss','cv_candidate_miss_rate','student_top1',
    'student_macro_f1','student_candidate_log_loss','student_candidate_miss_rate',
    'student_route_continuity','training_seconds_full_80_trips','inference_seconds_model_only',
    'evaluation_data_preparation_seconds','cv_seconds'])
  for name,m in models.items():
   c,s=m['cv'],m['student_test'];w.writerow([name,m['algorithm_family'],c['observation_top1_accuracy'],
    c['observation_macro_f1'],c['candidate_macro_f1'],c['candidate_log_loss'],c['candidate_miss_rate'],
    s['top1_accuracy'],s['macro_f1'],s['candidate_log_loss'],s['candidate_miss_rate'],
    s['route_continuity'],s['training_seconds_full_80_trips'],s['inference_seconds_total'],
    s.get('evaluation_data_preparation_seconds_total',s.get('candidate_generation_seconds_total',0)),
    c['cross_validation_seconds']])
 md=['# Stage 4 model comparison','',f"Run: `{data['run_id']}`",'',
     'Models share nine features, identical route-derived candidate labels, five GroupKFold folds by trip ID and the same ten student-test trips.',
     'Candidate Macro F1 and candidate Log Loss are computed only where the reference road was present in that observation candidate set; Candidate Miss Rate is reported separately. Observation-level metrics include misses.',
     'Primary selection criterion: grouped-CV observation-level Macro F1. This selects the model from validation folds, not from the student-test scores.','',
     '| Model | CV Top-1 | CV Macro F1 | Student Top-1 | Student Macro F1 | Student candidate Log Loss | Candidate miss | Continuity | Train seconds | Classifier inference seconds |',
     '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,m in models.items():
  c,s=m['cv'],m['student_test'];md.append(f"| {name} | {c['observation_top1_accuracy']:.4f} | {c['observation_macro_f1']:.4f} | {s['top1_accuracy']:.4f} | {s['macro_f1']:.4f} | {s['candidate_log_loss']:.4f} | {s['candidate_miss_rate']:.4f} | {s['route_continuity']:.4f} | {s['training_seconds_full_80_trips']:.2f} | {s['inference_seconds_total']:.2f} |")
 winner=max(models,key=lambda n:models[n]['cv']['observation_macro_f1'])
 md += ['',f"Selected on grouped-CV observation Macro F1: **{winner}** (Macro F1 {models[winner]['cv']['observation_macro_f1']:.4f}).",'',
        'Classifier inference time measures model probability scoring on prebuilt candidate features; candidate and reference preparation is reported separately in the CSV. See `results.json` for fold trip IDs, fold fit times and full metric records. `comparison_detailed.csv` retains the consolidated numeric table.']
 (a.results.parent/'comparison_summary.md').write_text('\n'.join(md)+'\n')
 print(csv_path);print(a.results.parent/'comparison_summary.md')
if __name__=='__main__':main()
