"""Generate real-data PathSnap PNGs from saved experiments without model retraining."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,log_loss,average_precision_score
from sklearn.inspection import permutation_importance

from pathsnap import Trip,ProjectedGraph,candidate_inputs,candidate_examples,labeled_candidate_mask

COLORS=['#32688e','#f28e2b','#59a14f','#e15759','#af7aa1']
FEATURES=['point-edge distance (m)','heading error (rad)','edge length (m)','edge fraction',
 'GPS speed (m/s)','elapsed progress','step displacement (m)','time gap (s)','heading alignment']
def fold_value(fold,key):
 aliases={'observation_macro_f1':'macro_f1','observation_top1_accuracy':'top1_accuracy'}
 return fold['metrics'].get(key,fold['metrics'][aliases[key]])

def save(fig,path):
 fig.tight_layout();fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig)

def positive_probability(model,x):
 p=model.predict_proba(x);classes=list(model.classes_) if hasattr(model,'classes_') else list(model[-1].classes_)
 return p[:,classes.index(1)]

def top1(scores,arcs,obs,nobs):
 out=np.full(nobs,-1,dtype=int);starts=np.r_[0,np.flatnonzero(obs[1:]!=obs[:-1])+1];ends=np.r_[starts[1:],len(obs)]
 for s,e in zip(starts,ends):out[obs[s]]=int(arcs[s+np.argmax(scores[s:e])])
 return out

def dataset_plots(data_root,out_root,trip_id):
 out=out_root/'dataset';out.mkdir(parents=True,exist_ok=True)
 folder=data_root/'train'/trip_id
 trip=Trip.from_directory(folder,include_route=False);graph=ProjectedGraph(trip)

 fig,ax=plt.subplots(figsize=(9,7));sc=ax.scatter(graph.track_xy[:,0],graph.track_xy[:,1],c=np.arange(len(graph.track_xy)),s=5,cmap='viridis',label='GPS observations')
 ax.plot(graph.track_xy[:,0],graph.track_xy[:,1],color='#555555',lw=.7,alpha=.45,label='GPS path')
 fig.colorbar(sc,ax=ax,label='Observation order');ax.set(title=f'GPS trajectory - training trip {trip_id}',xlabel='Local east (m)',ylabel='Local north (m)');ax.legend();ax.axis('equal')
 save(fig,out/'gps_trajectory.png')

 lo=graph.track_xy.min(axis=0)-300;hi=graph.track_xy.max(axis=0)+300
 mids=graph.mid;mask=np.all((mids>=lo)&(mids<=hi),axis=1)
 edge_ids=np.flatnonzero(mask)
 if len(edge_ids)>50000: edge_ids=edge_ids[np.linspace(0,len(edge_ids)-1,50000).astype(int)]
 segments=np.stack((graph.start[edge_ids],graph.end[edge_ids]),axis=1)
 fig,ax=plt.subplots(figsize=(9,7));ax.add_collection(LineCollection(segments,colors='#8c9aa5',linewidths=.45,alpha=.65,label=f'Road arcs ({len(edge_ids):,} shown)'))
 nodes=graph.nodes_xy;nm=np.all((nodes>=lo)&(nodes<=hi),axis=1)
 ax.scatter(nodes[nm,0],nodes[nm,1],s=2,color='#32688e',alpha=.35,label=f'Road nodes ({int(nm.sum()):,})')
 ax.scatter(graph.track_xy[:,0],graph.track_xy[:,1],s=4,color='#e15759',alpha=.55,label='GPS')
 ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.axis('equal');ax.set(title=f'Road network and GPS - trip {trip_id}',xlabel='Local east (m)',ylabel='Local north (m)');ax.legend(loc='best')
 save(fig,out/'road_network_nodes_arcs.png')

 indexes=np.unique(np.linspace(0,len(graph.track_xy)-1,min(14,len(graph.track_xy))).astype(int))
 cand=[]
 for i in indexes:
  eids,_,_=graph.candidates(graph.track_xy[i],24);cand.extend((int(i),int(e)) for e in eids)
 fig,ax=plt.subplots(figsize=(10,7));
 for i,e in cand:
  ax.plot(graph.start[e,:1].tolist()+graph.end[e,:1].tolist(),graph.start[e,1:].tolist()+graph.end[e,1:].tolist(),color='#f28e2b',lw=.8,alpha=.32)
 ax.scatter(graph.track_xy[indexes,0],graph.track_xy[indexes,1],s=35,color='#e15759',marker='x',label='Selected GPS observations',zorder=3)
 ax.plot(graph.track_xy[:,0],graph.track_xy[:,1],color='#777777',lw=.5,alpha=.25,label='GPS path')
 ax.axis('equal');ax.set(title=f'Generated road candidates at 14 GPS points - trip {trip_id}',xlabel='Local east (m)',ylabel='Local north (m)');ax.legend()
 save(fig,out/'gps_points_generated_candidates.png')

 gaps=[]
 for split in ('train','student_test'):
  for d in sorted((data_root/split).iterdir()):
   if d.is_dir():
    a=np.loadtxt(d/f'{d.name}.track',ndmin=2);gaps.extend(np.diff(a[:,2]).tolist())
 gaps=np.asarray(gaps);fig,ax=plt.subplots(figsize=(9,5));shown=gaps[(gaps>0)&(gaps<=30)]
 ax.hist(shown,bins=np.arange(0,31,.5),color='#4e79a7',edgecolor='white');ax.axvline(1,color='#e15759',ls='--',label='1-second interval')
 ax.set(title=f'Timestamp-step distribution (all 90 trips; {int((gaps>30).sum()):,} gaps >30 s omitted from axis)',xlabel='Gap between consecutive observations (s)',ylabel='Step count');ax.legend()
 save(fig,out/'timestamp_gap_distribution.png')

 inventory=list(csv.DictReader(open('artifacts/dataset_exploration/trip_inventory.csv')))
 weak=[int(r['weak_component_count']) for r in inventory if r.get('weak_component_count')]
 conn=[float(r['route_connected_step_fraction']) for r in inventory if r.get('route_connected_step_fraction')]
 fig,axs=plt.subplots(1,2,figsize=(11,4.8));axs[0].hist(weak,bins=25,color='#59a14f',edgecolor='white');axs[0].set(title='Weak components per trip graph',xlabel='Connected component count',ylabel='Trip count')
 axs[1].hist(conn,bins=np.linspace(0,1,21),color='#af7aa1',edgecolor='white');axs[1].set(title='Consecutive reference-route arc connectivity',xlabel='Connected-step fraction',ylabel='Trip count')
 fig.suptitle('Road-network connectivity statistics (all supplied trips)');save(fig,out/'road_network_connectivity.png')
 return [str(p) for p in sorted(out.glob('*.png'))]

def evaluate_saved_models(results_path,data_root,plots_root,force=False):
 results=json.loads(results_path.read_text());run_id=results['run_id'];out=plots_root/run_id
 out.mkdir(parents=True,exist_ok=True);cache=out/'evaluation_details.json'
 if cache.exists() and not force:return json.loads(cache.read_text()),out
 details={'run_id':run_id,'candidate_count':results['candidate_count'],'models':{}}
 trips=[];models_loaded={}
 import pickle
 for name,record in results['models'].items():
  model_path=Path(record['model_path'])
  if not model_path.is_absolute():model_path=Path.cwd()/model_path
  with model_path.open('rb') as f:models_loaded[name]=pickle.load(f)
 for folder in sorted((data_root/'student_test').iterdir()):
  if not folder.is_dir():continue
  inp=Trip.from_directory(folder,include_route=False)
  x,arcs,obs=candidate_inputs(inp,results['candidate_count'])

  x=x.astype(np.float32)

  model_scores={}
  for name,model in models_loaded.items():
   scores=positive_probability(model,x);model_scores[name]=(scores,top1(scores,arcs,obs,len(inp.track)))
  ref=Trip.from_directory(folder,include_route=True)
  truth=ProjectedGraph(ref).labels_from_route();y=(arcs==truth[obs])
  mask=labeled_candidate_mask(y,obs,len(truth))
  trips.append((inp,ref,x,y,arcs,obs,truth,mask,model_scores))
 for name,record in results['models'].items():
  all_y=[];all_p=[];conf=np.zeros((2,2),dtype=int);truth_codes=[];pred_codes=[]
  point_correct=point_count=0;continuity_num=continuity_den=0
  feature_score_sum=np.zeros(len(FEATURES));feature_score_n=0
  for inp,ref,x,y,arcs,obs,truth,mask,model_scores in trips:
   model=models_loaded[name]
   scores,pred=model_scores[name];all_y.append(y[mask]);all_p.append(scores[mask])
   pred_binary=scores[mask]>=.5;cm=confusion_matrix(y[mask],pred_binary,labels=[0,1]);conf+=cm
   point_correct+=int(np.sum(pred==truth));point_count+=len(truth)
   truth_codes.extend(f'{inp.trip_id}:{int(q)}' for q in truth);pred_codes.extend(f'{inp.trip_id}:{int(q)}' for q in pred)
   graph=ProjectedGraph(inp)
   for a,b in zip(pred,pred[1:]):
    continuity_den+=1;continuity_num+=int(a==b or int(b) in graph.edge_adjacency[int(a)])
   if name=='logistic_regression':
    est=model.named_steps['logisticregression'];imp=np.abs(est.coef_[0])
   elif name=='gaussian_naive_bayes':
    imp=np.abs(model.theta_[1]-model.theta_[0])/np.sqrt(model.var_[1]+model.var_[0]+1e-12)
   elif hasattr(model,'feature_importances_'):
    imp=np.asarray(model.feature_importances_)
   else:

    valid=np.flatnonzero(mask);rng=np.random.default_rng(42)
    sample=valid if len(valid)<=6000 else rng.choice(valid,6000,replace=False)
    perm=permutation_importance(model,x[sample],y[sample],scoring='average_precision',
                                n_repeats=3,random_state=42,n_jobs=1)
    imp=perm.importances_mean
   feature_score_sum+=imp;feature_score_n+=1
  yy,pp=np.concatenate(all_y),np.concatenate(all_p);pred=pp>=.5
  precision,recall,f1,_=precision_recall_fscore_support(yy,pred,average='binary',zero_division=0)
  details['models'][name]={'candidate_confusion_matrix':conf.tolist(),
    'candidate_precision':float(precision),'candidate_recall':float(recall),'candidate_f1':float(f1),
    'candidate_macro_f1':float(__import__('sklearn.metrics',fromlist=['f1_score']).f1_score(yy,pred,average='macro',zero_division=0)),
    'candidate_log_loss':float(log_loss(yy,np.column_stack((1-pp,pp)),labels=[0,1])),
    'observation_top1_accuracy':point_correct/max(1,point_count),
    'route_continuity':continuity_num/max(1,continuity_den),
    'feature_importance':(feature_score_sum/max(1,feature_score_n)).tolist(),
    'feature_importance_method':('absolute standardized logistic coefficients' if name=='logistic_regression' else
      'Gaussian NB standardized mean difference' if name=='gaussian_naive_bayes' else
      'native impurity importance' if name in ('decision_tree','extra_trees') else
      'student-test permutation average-precision importance (diagnostic only)')}

 for name,metrics in details['models'].items():
  stored=results['models'][name]['student_test']['top1_accuracy']
  if abs(metrics['observation_top1_accuracy']-stored)>1e-10:
   raise RuntimeError(f'Saved-model evaluation differs from experiment result for {name}')
 cache.write_text(json.dumps(details,indent=2));return details,out

def model_plots(results,details,out):
 colors=dict(zip(results['models'],COLORS));written=[]
 for name,rec in results['models'].items():
  folder=out/'models'/name;folder.mkdir(parents=True,exist_ok=True);d=details['models'][name];color=colors[name]
  cm=np.asarray(d['candidate_confusion_matrix']);fig,ax=plt.subplots(figsize=(6.2,5.2));im=ax.imshow(cm,cmap='Blues')
  for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>cm.max()*.55 else '#222222',fontsize=12)
  ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],
    xlabel='Predicted candidate label',ylabel='Reference candidate label',title=f'{name}: student-test candidate confusion matrix')
  fig.colorbar(im,ax=ax,label='Candidate count');p=folder/'confusion_matrix.png';save(fig,p);written.append(str(p))
  fig,ax=plt.subplots(figsize=(6.5,4.8));vals=[d['candidate_precision'],d['candidate_recall'],d['candidate_f1']]
  bars=ax.bar(['Precision','Recall','F1'],vals,color=[color]*3)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.015,f'{v:.3f}',ha='center',fontsize=10)
  ax.set(ylim=(0,1.08),ylabel='Score',title=f'{name}: student-test candidate metrics (threshold 0.5)');ax.grid(axis='y',alpha=.2)
  p=folder/'precision_recall_f1.png';save(fig,p);written.append(str(p))
  folds=rec['cv']['folds'];macro=[fold_value(f,'observation_macro_f1') for f in folds];top=[fold_value(f,'observation_top1_accuracy') for f in folds]
  fig,ax=plt.subplots(figsize=(7,4.8));x=np.arange(1,len(folds)+1);ax.plot(x,macro,'o-',label='Observation Macro F1',color=color)
  ax.plot(x,top,'s--',label='Observation Top-1 accuracy',color='#4e79a7');ax.set(xticks=x,xlabel='GroupKFold fold',ylabel='Score',ylim=(0,1),title=f'{name}: held-out performance by trip fold')
  ax.grid(alpha=.25);ax.legend();
  ax.text(.02,.03,f'Macro F1 mean ± SD: {np.mean(macro):.3f} ± {np.std(macro,ddof=1):.3f}',transform=ax.transAxes,fontsize=9)
  p=folder/'cross_validation_folds.png';save(fig,p);written.append(str(p))
  imp=np.asarray(d['feature_importance']);order=np.argsort(imp)
  fig,ax=plt.subplots(figsize=(8,5.3));ax.barh(np.array(FEATURES)[order],imp[order],color=color);ax.set(xlabel='Importance (model-specific scale)',title=f'{name}: feature analysis\n{d["feature_importance_method"]}');ax.grid(axis='x',alpha=.2)
  p=folder/'feature_analysis.png';save(fig,p);written.append(str(p))
 return written

def comparison_plots(results,out):
 folder=out/'comparison';folder.mkdir(parents=True,exist_ok=True);models=results['models'];names=list(models);x=np.arange(len(names));written=[]
 configs=[('student_test','top1_accuracy','Top-1 accuracy','Score','student_top1_accuracy.png',False),
 ('student_test','macro_f1','Macro F1','Score','student_macro_f1.png',False),
 ('student_test','candidate_log_loss','Candidate Log Loss','Log loss','student_candidate_log_loss.png',False),
 ('student_test','training_seconds_full_80_trips','Training time','Seconds','training_time.png',False),
 ('student_test','inference_seconds_total','Classifier inference time','Seconds','inference_time.png',False)]
 for section,key,title,ylabel,filename,_ in configs:
  vals=[models[n][section][key] for n in names];fig,ax=plt.subplots(figsize=(9,5));bars=ax.bar(x,vals,color=COLORS[:len(names)])
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+(max(vals)*.012 if max(vals) else .01),f'{v:.3f}' if v<2 else f'{v:.2f}',ha='center',fontsize=9)
  ax.set(xticks=x,xticklabels=names,ylabel=ylabel,title=title);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
  if key in ('top1_accuracy','macro_f1'):ax.set_ylim(0,1.08)
  p=folder/filename;save(fig,p);written.append(str(p))

 fig,ax=plt.subplots(figsize=(10,5.5))
 means=[];stds=[]
 for name in names:
  scores=[fold_value(f,'observation_macro_f1') for f in models[name]['cv']['folds']]
  means.append(float(np.mean(scores)));stds.append(float(np.std(scores,ddof=1)))
 bars=ax.bar(x,means,yerr=stds,capsize=5,color=COLORS[:len(names)])
 for b,m,s in zip(bars,means,stds):ax.text(b.get_x()+b.get_width()/2,m+s+.01,f'{m:.3f} ± {s:.3f}',ha='center',fontsize=9)
 ax.set(xticks=x,xticklabels=names,ylabel='Observation-level Macro F1',ylim=(0,1),title='Five-fold trip-grouped CV mean ± standard deviation');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
 p=folder/'cross_validation_mean_std.png';save(fig,p);written.append(str(p))

 fig,axs=plt.subplots(2,3,figsize=(15,9));items=[('student_test','top1_accuracy','Top-1 accuracy'),('student_test','macro_f1','Macro F1'),
 ('student_test','candidate_log_loss','Candidate Log Loss'),('student_test','training_seconds_full_80_trips','Training seconds'),
 ('student_test','inference_seconds_total','Classifier inference seconds'),('cv','observation_macro_f1','Grouped-CV Macro F1')]
 for ax,(sec,key,title) in zip(axs.flat,items):
  vals=[models[n][sec][key] if sec=='student_test' else np.mean([fold_value(f,key) for f in models[n]['cv']['folds']]) for n in names]
  err=[0 if sec=='student_test' or key!='observation_macro_f1' else np.std([fold_value(f,key) for f in models[n]['cv']['folds']],ddof=1) for n in names]
  b=ax.bar(x,vals,yerr=err,capsize=3,color=COLORS[:len(names)])
  ax.set(xticks=x,xticklabels=names,title=title);ax.tick_params(axis='x',rotation=35,labelsize=8);ax.grid(axis='y',alpha=.2)
  if key in ('top1_accuracy','macro_f1','observation_macro_f1'):ax.set_ylim(0,1)
  for bar,v in zip(b,vals):ax.text(bar.get_x()+bar.get_width()/2,v+(max(vals)*.02 if max(vals) else .01),f'{v:.3f}' if v<2 else f'{v:.2f}',ha='center',fontsize=7)
 fig.suptitle('PathSnap five-model comparison dashboard (actual recorded results)');save(fig,folder/'model_comparison_dashboard.png');written.append(str(folder/'model_comparison_dashboard.png'))
 return written

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--data',type=Path,default=Path('Student folder'))
 p.add_argument('--plots-root',type=Path,default=Path('artifacts/plots'));p.add_argument('--trip-id',default='00000034')
 p.add_argument('--force-evaluation',action='store_true');a=p.parse_args()
 if a.results is None:
  choices=sorted(Path('artifacts/experiments').glob('comparison-*/results.json'))
  if not choices:raise SystemExit('No saved combined experiment results found. Run experiments.py first.')
  a.results=choices[-1]
 result=json.loads(a.results.read_text())
 if any(not m['cv']['folds'] or 'metrics' not in m['cv']['folds'][0] for m in result['models'].values()):
  raise SystemExit(f'{a.results} has no saved per-fold metrics. Re-run experiments.py once; plotting itself never retrains.')
 out=a.plots_root/result['run_id'];out.mkdir(parents=True,exist_ok=True)
 files=dataset_plots(a.data,a.plots_root,a.trip_id)
 details,_=evaluate_saved_models(a.results,a.data,a.plots_root,a.force_evaluation)
 files+=model_plots(result,details,out);files+=comparison_plots(result,out)
 manifest={'results':str(a.results),'run_id':result['run_id'],'plots':[str(Path(f)) for f in files],
           'plot_count':len(files),'dpi':300,'evaluation_cache':str(out/'evaluation_details.json')}
 (out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
