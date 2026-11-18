"""Validate feature and candidate-label generation across train and student-test trips."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from pathsnap import Trip,candidate_examples

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('Student folder'))
 p.add_argument('--output',type=Path,default=Path('artifacts/data_pipeline'))
 p.add_argument('--candidates',type=int,default=24);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[]
 for split in ('train','student_test'):
  folders=sorted(q for q in (a.data/split).iterdir() if q.is_dir())
  for i,folder in enumerate(folders,1):
   trip=Trip.from_directory(folder);x,y,arc_ids,obs_ids,truth=candidate_examples(trip,a.candidates)
   counts=np.bincount(obs_ids,weights=y.astype(np.int8),minlength=len(truth))
   cand_count=np.bincount(obs_ids,minlength=len(truth))
   row={'split':split,'trip_id':trip.trip_id,'gps_observations':len(truth),
     'road_nodes':len(trip.nodes),'road_arcs':len(trip.arcs),'reference_route_arcs':len(trip.route),
     'feature_count':x.shape[1],'candidate_rows':len(y),'mean_candidates_per_observation':float(cand_count.mean()),
     'candidate_miss_count':int((counts==0).sum()),'candidate_miss_rate':float(np.mean(counts==0)),
     'positive_labels_per_observation_max':int(counts.max()),'nonfinite_feature_values':int((~np.isfinite(x)).sum()),
     'invalid_candidate_arc_ids':int(((arc_ids<0)|(arc_ids>=len(trip.arcs))).sum())}
   rows.append(row);print(f"{split} {i}/{len(folders)} {trip.trip_id}: miss={row['candidate_miss_rate']:.3%}, features={x.shape[1]}",flush=True)
 fields=list(rows[0])
 suffix=f'_k{a.candidates}'
 with (a.output/f'candidate_label_audit{suffix}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 summary={'candidate_count_requested':a.candidates,'feature_count':rows[0]['feature_count'],
   'trips':{},'feature_names':['point_to_arc_distance_m','heading_error_rad','arc_length_m',
   'along_arc_fraction','gps_speed_mps','elapsed_time_progress','step_displacement_m',
   'timestamp_gap_s','heading_alignment_cosine']}
 for split in ('train','student_test'):
  r=[x for x in rows if x['split']==split]
  summary['trips'][split]={'trip_count':len(r),'total_observations':sum(x['gps_observations'] for x in r),
    'candidate_miss_rate_weighted':sum(x['candidate_miss_count'] for x in r)/sum(x['gps_observations'] for x in r),
    'candidate_miss_rate_trip_min_median_max':[float(np.min([x['candidate_miss_rate'] for x in r])),
      float(np.median([x['candidate_miss_rate'] for x in r])),float(np.max([x['candidate_miss_rate'] for x in r]))],
    'nonfinite_feature_values':sum(x['nonfinite_feature_values'] for x in r),
    'invalid_candidate_arc_ids':sum(x['invalid_candidate_arc_ids'] for x in r),
    'max_positive_labels_per_observation':max(x['positive_labels_per_observation_max'] for x in r)}
 (a.output/f'candidate_label_audit{suffix}.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
