"""Structured inventory and data-quality report for every provided train/dev trip."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import Trip


def read(path: Path, kind: str) -> np.ndarray:
    try:
        a=np.loadtxt(path, dtype=np.float64)
        if kind=='route': return np.asarray(a,dtype=np.float64).reshape(-1,1)
        return np.atleast_2d(a)
    except ValueError:
        return np.empty((0, 0))


def inspect_trip(folder: Path, split: str) -> dict:
    tid=folder.name
    files={ext:folder/f'{tid}.{ext}' for ext in ('track','nodes','arcs','route')}
    missing=[ext for ext,p in files.items() if not p.is_file()]
    row={'split':split,'trip_id':tid,'missing_files':','.join(missing)}
    for ext,p in files.items():
        row[f'{ext}_bytes']=p.stat().st_size if p.exists() else 0
    if any(not p.exists() for p in files.values()):
        return row
    track,nodes,arcs,route=(read(files[e],e) for e in ('track','nodes','arcs','route'))
    row.update({'track_rows':len(track),'node_rows':len(nodes),'arc_rows':len(arcs),'route_rows':len(route)})
    row['track_columns']=track.shape[1]; row['node_columns']=nodes.shape[1]
    row['arc_columns']=arcs.shape[1]; row['route_columns']=route.shape[1]
    for name,a in [('track',track),('nodes',nodes),('arcs',arcs),('route',route)]:
        row[f'{name}_nonfinite_values']=int((~np.isfinite(a)).sum()) if a.size else 0
    coords=track[:,:2] if track.shape[1]>=2 else np.empty((0,2))
    row['invalid_coordinate_rows']=int((~np.isfinite(coords).all(axis=1)).sum()) if len(coords) else 0
    row['out_of_range_coordinate_rows']=int(((np.abs(coords[:,0])>180)|(np.abs(coords[:,1])>90)).sum()) if len(coords) else 0
    if len(coords):
        row.update({'lon_min':float(coords[:,0].min()),'lon_max':float(coords[:,0].max()),
                    'lat_min':float(coords[:,1].min()),'lat_max':float(coords[:,1].max())})
    else:
        row.update({'lon_min':None,'lon_max':None,'lat_min':None,'lat_max':None})
    if track.shape[1]>=3 and len(track)>1:
        dt=np.diff(track[:,2]); row['timestamp_nonincreasing_steps']=int((dt<=0).sum())
        row['timestamp_non_1s_steps']=int((np.abs(dt-1.0)>0.001).sum())
        row['timestamp_exact_1s_fraction']=float(np.mean(np.abs(dt-1.0)<=0.001))
        row['timestamp_gap_gt_1s_steps']=int((dt>1.001).sum())
        row['timestamp_gap_gt_5s_steps']=int((dt>5).sum())
        row['timestamp_step_median_s']=float(np.median(dt))
        row['timestamp_step_p95_s']=float(np.quantile(dt,.95))
    else:
        row.update({'timestamp_nonincreasing_steps':None,'timestamp_gap_gt_1s_steps':None,
                    'timestamp_gap_gt_5s_steps':None,'timestamp_step_median_s':None,
                    'timestamp_step_p95_s':None,'timestamp_non_1s_steps':None,
                    'timestamp_exact_1s_fraction':None})
    if len(arcs) and len(nodes) and np.isfinite(arcs[:,:2]).all():
        ai=arcs[:,:2].astype(np.int64)
        row['invalid_arc_node_references']=int(((ai<0)|(ai>=len(nodes))).any(axis=1).sum())
        valid=ai[(ai>=0).all(axis=1)&(ai<len(nodes)).all(axis=1)]
        degree=np.bincount(valid.ravel(),minlength=len(nodes))
        row['isolated_nodes']=int((degree==0).sum())
        if len(valid):
            rr=np.r_[valid[:,0],valid[:,1]]; cc=np.r_[valid[:,1],valid[:,0]]
            graph=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(nodes),len(nodes))).tocsr()
            row['weak_component_count']=int(connected_components(graph,directed=False,return_labels=False))
        else: row['weak_component_count']=len(nodes)
    else:
        row.update({'invalid_arc_node_references':None,'isolated_nodes':None,'weak_component_count':None})
    if len(route):
        rid=route.reshape(-1)
        valid=np.isfinite(rid)&(rid>=0)&(rid<len(arcs))
        row['invalid_route_arc_references']=int((~valid).sum())
        route_ids=rid[valid].astype(np.int64)
        if len(route_ids)>1 and len(arcs):
            ai=arcs[:,:2].astype(np.int64)
            row['route_connected_step_fraction']=float(np.mean(ai[route_ids[:-1],1]==ai[route_ids[1:],0]))
        else: row['route_connected_step_fraction']=None
    else:
        row.update({'invalid_route_arc_references':None,'route_connected_step_fraction':None})
    return row


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('Student folder'))
    p.add_argument('--output',type=Path,default=Path('artifacts/dataset_exploration'))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ('train','student_test'):
        folders=sorted(q for q in (a.data/split).iterdir() if q.is_dir())
        print(f'Inspecting {split}: {len(folders)} trips',flush=True)
        for i,folder in enumerate(folders,1):
            rows.append(inspect_trip(folder,split))
            if i%10==0 or i==len(folders): print(f'  {i}/{len(folders)}',flush=True)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (a.output/'trip_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'source':str(a.data),'trip_count':len(rows),'partitions':{},'file_formats':{
        '.track':'longitude, latitude, timestamp in seconds (per supplied data guide)',
        '.nodes':'longitude, latitude; 0-based row index is node id',
        '.arcs':'from_node_id, to_node_id; 0-based row index is arc id',
        '.route':'chronological arc_id sequence, where provided'},
        'checks':'Missing files, row/column counts, finite values, coordinate range, timestamp ordering/gaps, node references, isolated nodes, weak components, route references and route adjacency.'}
    for split in ('train','student_test'):
        sub=[r for r in rows if r['split']==split]
        def stat(key):
            v=[r[key] for r in sub if isinstance(r.get(key),(int,float))]
            return {'min':min(v),'median':float(np.median(v)),'max':max(v)} if v else None
        summary['partitions'][split]={'trips':len(sub),'track_rows':stat('track_rows'),
            'nodes':stat('node_rows'),'arcs':stat('arc_rows'),'route_rows':stat('route_rows'),
            'timestamp_step_median_s':stat('timestamp_step_median_s'),
            'trips_missing_files':sum(bool(r['missing_files']) for r in sub),
            'invalid_coordinate_rows':sum(r.get('invalid_coordinate_rows',0) for r in sub),
            'timestamp_nonincreasing_steps':sum(r.get('timestamp_nonincreasing_steps') or 0 for r in sub),
            'timestamp_non_1s_steps':sum(r.get('timestamp_non_1s_steps') or 0 for r in sub),
            'timestamp_gaps_gt_1s_steps':sum(r.get('timestamp_gap_gt_1s_steps') or 0 for r in sub),
            'timestamp_gaps_gt_5s_steps':sum(r.get('timestamp_gap_gt_5s_steps') or 0 for r in sub),
            'invalid_arc_node_references':sum(r.get('invalid_arc_node_references') or 0 for r in sub),
            'invalid_route_arc_references':sum(r.get('invalid_route_arc_references') or 0 for r in sub),
            'route_connected_step_fraction':stat('route_connected_step_fraction'),
            'timestamp_exact_1s_fraction':stat('timestamp_exact_1s_fraction'),
            'weak_component_count':stat('weak_component_count')}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    report=['# PathSnap dataset exploration','',f"Source: `{a.data}`",'']
    for split,d in summary['partitions'].items():
        report += [f'## {split}', '',f"- Trips: {d['trips']}",f"- GPS rows (min/median/max): {d['track_rows']}",
          f"- Nodes (min/median/max): {d['nodes']}",f"- Arcs (min/median/max): {d['arcs']}",
          f"- Route rows (min/median/max): {d['route_rows']}",f"- Missing-file trips: {d['trips_missing_files']}",
          f"- Invalid coordinate rows: {d['invalid_coordinate_rows']}",
          f"- Nonincreasing timestamp steps: {d['timestamp_nonincreasing_steps']}",
          f"- Non-1 s timestamp steps: {d['timestamp_non_1s_steps']}; exact-1 s fraction (min/median/max): {d['timestamp_exact_1s_fraction']}",
          f"- Timestamp gaps >1 s / >5 s: {d['timestamp_gaps_gt_1s_steps']} / {d['timestamp_gaps_gt_5s_steps']}",
          f"- Invalid graph references: {d['invalid_arc_node_references']}",
          f"- Invalid route references: {d['invalid_route_arc_references']}",
          f"- Route connected-step fraction (min/median/max): {d['route_connected_step_fraction']}",
          f"- Weak components per graph (min/median/max): {d['weak_component_count']}",'']
    (a.output/'summary.md').write_text('\n'.join(report))
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for split,color in [('train','#3a78a1'),('student_test','#ed9d43')]:
        sub=[r for r in rows if r['split']==split]
        axs[0,0].hist([r['track_rows'] for r in sub],bins=18,alpha=.65,label=split,color=color)
        axs[0,1].hist([r['timestamp_step_median_s'] for r in sub if r['timestamp_step_median_s'] is not None],
                      bins=15,alpha=.65,label=split,color=color)
        axs[1,0].hist([r['weak_component_count'] for r in sub if r['weak_component_count'] is not None],
                      bins=20,alpha=.65,label=split,color=color)
    axs[0,0].set(title='GPS observations per trip',xlabel='Rows')
    axs[0,1].set(title='Median sampling interval',xlabel='Seconds')
    axs[1,0].set(title='Road graph weak components',xlabel='Components')
    axs[1,1].bar(['train','student_test'],[len([r for r in rows if r['split']==s]) for s in ('train','student_test')],color=['#3a78a1','#ed9d43'])
    axs[1,1].set(title='Trip counts',ylabel='Trips')
    for ax in axs.flat: ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(); fig.tight_layout(); fig.savefig(a.output/'dataset_exploration.png',dpi=160); plt.close(fig)
    print(f"Wrote inventory and exploration report to {a.output}",flush=True)


if __name__=='__main__': main()
