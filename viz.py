"""Folium map generation for PathSnap (used by the API and usable as a standalone script)."""
from __future__ import annotations

import numpy as np
import folium

from pathsnap import Trip

LEGEND = """
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 14px;
 border:1px solid #888;border-radius:8px;font:13px/1.6 sans-serif;box-shadow:0 2px 8px #0004">
 <b>PathSnap legend</b><br>
 <span style="color:#1f77b4">&#9679;</span> Original GPS observations<br>
 <span style="color:#ff7f0e">&#9679;</span> Matched positions (on predicted road)<br>
 <span style="color:#d62728"><b>&#9473;</b></span> Predicted route<br>
 {reference_row}
 <span style="color:#9a9a9a"><b>&#9473;</b></span> Road network (nearby)<br>
 <span style="color:#2ca02c">&#9650;</span> Start &nbsp; <span style="color:#000">&#9632;</span> End
</div>"""
REFERENCE_ROW = '<span style="color:#2ca02c"><b>&#9473;</b></span> Reference route<br>'


def _arc_line(trip: Trip, arc: int) -> list[list[float]]:
    a, b = trip.arcs[arc, :2].astype(int)
    return [[trip.nodes[a, 1], trip.nodes[a, 0]], [trip.nodes[b, 1], trip.nodes[b, 0]]]  # lat, lon


def build_map(trip: Trip, result: dict, reference: np.ndarray | None = None,
              pad_deg: float = 0.004, max_network_arcs: int = 6000) -> str:
    """Return the HTML of an interactive map comparing GPS, roads, predicted and reference routes."""
    track = trip.track[:, :2]
    lat_lon = [[float(lat), float(lon)] for lon, lat in track]
    m = folium.Map(location=np.mean(lat_lon, axis=0).tolist(), zoom_start=15,
                   tiles='OpenStreetMap', control_scale=True)

    # Road network near the trip (bounding box of the track, padded)
    lo, hi = track.min(axis=0) - pad_deg, track.max(axis=0) + pad_deg
    ends = trip.nodes[trip.arcs[:, :2].astype(int), :2]  # (n_arcs, 2, 2)
    near = np.flatnonzero(((ends >= lo) & (ends <= hi)).all(axis=2).any(axis=1))[:max_network_arcs]
    roads = folium.FeatureGroup(name='Road network (nearby)', show=True)
    for arc in near:
        folium.PolyLine(_arc_line(trip, int(arc)), color='#9a9a9a', weight=2, opacity=.7,
                        tooltip=f'arc {int(arc)}').add_to(roads)
    roads.add_to(m)

    if reference is not None and len(reference):
        ref = folium.FeatureGroup(name='Reference route', show=True)
        for arc in dict.fromkeys(int(a) for a in reference):
            folium.PolyLine(_arc_line(trip, arc), color='#2ca02c', weight=7, opacity=.55,
                            tooltip=f'reference arc {arc}').add_to(ref)
        ref.add_to(m)

    pred = folium.FeatureGroup(name='Predicted route', show=True)
    for arc in dict.fromkeys(int(a) for a in result['route']):
        folium.PolyLine(_arc_line(trip, arc), color='#d62728', weight=4, opacity=.95,
                        tooltip=f'predicted arc {arc}').add_to(pred)
    pred.add_to(m)

    gps = folium.FeatureGroup(name='Original GPS observations', show=True)
    folium.PolyLine(lat_lon, color='#1f77b4', weight=1, opacity=.5).add_to(gps)
    for i, (lat, lon) in enumerate(lat_lon):
        label = f'GPS #{i}' + (f' (t={trip.track[i, 2]:g})' if trip.track.shape[1] > 2 else '')
        folium.CircleMarker([lat, lon], radius=3, color='#1f77b4', fill=True, fill_opacity=.9,
                            tooltip=label).add_to(gps)
    gps.add_to(m)

    matched = folium.FeatureGroup(name='Matched positions', show=False)
    for i, (lon, lat) in enumerate(result['matched_coordinates']):
        folium.CircleMarker([lat, lon], radius=3, color='#ff7f0e', fill=True, fill_opacity=.9,
                            tooltip=f'GPS #{i} -> arc {result["matched_arc_indices"][i]}').add_to(matched)
    matched.add_to(m)

    folium.Marker(lat_lon[0], tooltip='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(lat_lon[-1], tooltip='End', icon=folium.Icon(color='black', icon='stop')).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        LEGEND.format(reference_row=REFERENCE_ROW if reference is not None else '')))
    m.fit_bounds([[float(lo[1]), float(lo[0])], [float(hi[1]), float(hi[0])]])
    return m.get_root().render()


if __name__ == '__main__':
    # Standalone: python viz.py <trip_folder> <model.pkl> <out.html>
    import pickle
    import sys
    from pathsnap import decode
    folder, model_path, out = sys.argv[1:4]
    with open(model_path, 'rb') as f:
        mdl = pickle.load(f)
    t = Trip.from_directory(folder, include_route=False)
    ref = Trip.from_directory(folder, include_route=True).route
    open(out, 'w', encoding='utf-8').write(build_map(t, decode(t, mdl, 24, 0.0), ref))
    print('wrote', out)
