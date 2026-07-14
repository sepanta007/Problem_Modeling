from __future__ import annotations

import html

import folium

from .itinerary import build_itinerary
from .network import RoadNetwork

_LEG_COLORS = ["#2563eb", "#7c3aed", "#059669", "#db2777", "#d97706"]

# turn phrase -> glyph shown in the little coloured badge
_GLYPH = {
    "départ": "▶",
    "tout droit": "↑",
    "à droite": "↱",
    "légèrement à droite": "↗",
    "à gauche": "↰",
    "légèrement à gauche": "↖",
    "demi-tour": "↻",
}


def _fmt(d: float) -> str:
    return f"{d:,.0f} m" if d < 1000 else f"{d / 1000:.2f} km"


def _glyph(instr: str) -> str:
    return _GLYPH.get(instr, "•")


_CSS = """
<style>
#route-panel *{box-sizing:border-box}
#route-panel{
  position:fixed; top:16px; right:16px; z-index:9999; width:360px;
  max-height:88vh; display:flex; flex-direction:column;
  background:#ffffff; border-radius:16px; overflow:hidden;
  box-shadow:0 10px 40px rgba(15,23,42,.28);
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif; color:#0f172a;
}
#route-panel .rp-head{
  padding:16px 18px 14px; color:#fff;
  background:linear-gradient(135deg,#1e3a8a 0%,#4f46e5 55%,#7c3aed 100%);
}
#route-panel .rp-title{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  opacity:.85;margin-bottom:4px}
#route-panel .rp-route{font-size:16px;font-weight:700;line-height:1.35}
#route-panel .rp-stats{margin-top:10px;display:flex;gap:8px}
#route-panel .rp-chip{background:rgba(255,255,255,.18);border-radius:999px;
  padding:4px 12px;font-size:13px;font-weight:600;backdrop-filter:blur(4px)}
#route-panel .rp-body{overflow-y:auto;padding:6px 0 10px}
#route-panel .rp-body::-webkit-scrollbar{width:8px}
#route-panel .rp-body::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:8px}
#route-panel .rp-leg{padding:12px 16px 4px;font-weight:700;font-size:14px;
  display:flex;align-items:center;gap:8px}
#route-panel .rp-dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
#route-panel .rp-legdist{margin-left:auto;font-size:12px;font-weight:600;color:#64748b}
#route-panel .rp-step{display:flex;align-items:center;gap:11px;padding:7px 16px;
  border-radius:10px;margin:1px 8px;transition:background .12s}
#route-panel .rp-step:hover{background:#f1f5f9}
#route-panel .rp-badge{flex:0 0 auto;width:26px;height:26px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;color:#fff;
  font-size:15px;font-weight:700;box-shadow:0 1px 3px rgba(0,0,0,.2)}
#route-panel .rp-street{font-size:13.5px;font-weight:600;line-height:1.25}
#route-panel .rp-sub{font-size:11.5px;color:#94a3b8;margin-top:1px}
</style>
"""


def _panel_html(legs: list[dict], total_m: float, title: str) -> str:
    n_steps = sum(len(leg["steps"]) for leg in legs)
    body = []
    for k, leg in enumerate(legs):
        color = _LEG_COLORS[k % len(_LEG_COLORS)]
        leg_m = sum(s.distance_m for s in leg["steps"])
        body.append(
            f"<div class='rp-leg'><span class='rp-dot' style='background:{color}'></span>"
            f"{html.escape(leg['from'])} → {html.escape(leg['to'])}"
            f"<span class='rp-legdist'>{_fmt(leg_m)}</span></div>"
        )
        for s in leg["steps"]:
            sub = _fmt(s.distance_m)
            if s.instruction != "départ":
                sub += f" · {html.escape(s.instruction)}"
            body.append(
                f"<div class='rp-step'>"
                f"<span class='rp-badge' style='background:{color}'>{_glyph(s.instruction)}</span>"
                f"<span><div class='rp-street'>{html.escape(s.street)}</div>"
                f"<div class='rp-sub'>{sub}</div></span></div>"
            )
    return (
        _CSS +
        "<div id='route-panel'>"
        "<div class='rp-head'>"
        "<div class='rp-title'>Itinéraire optimal</div>"
        f"<div class='rp-route'>{html.escape(title)}</div>"
        "<div class='rp-stats'>"
        f"<span class='rp-chip'>📍 {total_m/1000:.2f} km</span>"
        f"<span class='rp-chip'>🧭 {n_steps} étapes</span>"
        f"<span class='rp-chip'>🚗 {len(legs)} trajets</span>"
        "</div></div>"
        f"<div class='rp-body'>{''.join(body)}</div>"
        "</div>"
    )


def _pin(color: str, glyph: str) -> folium.DivIcon:
    """A circular map pin with a glyph inside."""
    html_ = (
        f"<div style='width:30px;height:30px;border-radius:50% 50% 50% 0;"
        f"transform:rotate(-45deg);background:{color};border:2.5px solid #fff;"
        f"box-shadow:0 2px 6px rgba(0,0,0,.4);display:flex;align-items:center;"
        f"justify-content:center'>"
        f"<span style='transform:rotate(45deg);color:#fff;font-size:15px;"
        f"font-weight:700'>{glyph}</span></div>"
    )
    return folium.DivIcon(html=html_, icon_size=(30, 30), icon_anchor=(15, 30))


def build_route_map(
    net: RoadNetwork,
    legs: list[dict],
    out_html: str,
    title: str = "Itinéraire",
) -> str:
    """Render an interactive route map. ``legs``: list of
    ``{"path": [node ids], "from": label, "to": label}``."""
    all_pts: list[tuple[float, float]] = []
    for leg in legs:
        leg["steps"] = build_itinerary(net, leg["path"])
        all_pts += [(float(net.lat[n]), float(net.lon[n])) for n in leg["path"]]

    lat0 = sum(p[0] for p in all_pts) / len(all_pts)
    lon0 = sum(p[1] for p in all_pts) / len(all_pts)
    m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="cartodbpositron",
                   control_scale=True, zoom_control=True)

    total_m = 0.0
    for k, leg in enumerate(legs):
        color = _LEG_COLORS[k % len(_LEG_COLORS)]
        coords = [(float(net.lat[n]), float(net.lon[n])) for n in leg["path"]]
        # "Casing": a wider translucent dark line under the coloured route.
        folium.PolyLine(coords, color="#0f172a", weight=11, opacity=0.25).add_to(m)
        folium.PolyLine(coords, color=color, weight=6, opacity=0.95,
                        tooltip=f"{leg['from']} → {leg['to']}").add_to(m)
        for s in leg["steps"]:
            if s.instruction == "départ" or s.start_node < 0:
                continue
            folium.CircleMarker(
                [float(net.lat[s.start_node]), float(net.lon[s.start_node])],
                radius=4, color="#ffffff", weight=1.5, fill=True,
                fill_color=color, fill_opacity=1,
                popup=folium.Popup(
                    f"<b>{_glyph(s.instruction)} {html.escape(s.instruction)}</b>"
                    f"<br>{html.escape(s.street)}"
                    f"<br><span style='color:#888'>{_fmt(s.distance_m)}</span>",
                    max_width=220),
            ).add_to(m)
        total_m += sum(s.distance_m for s in leg["steps"])

    # Start / intermediate / end pins.
    stops = [legs[0]["path"][0]] + [leg["path"][-1] for leg in legs]
    labels = [legs[0]["from"]] + [leg["to"] for leg in legs]
    for i, (node, lab) in enumerate(zip(stops, labels)):
        if i == 0:
            pin = _pin("#16a34a", "▶")
        elif i == len(stops) - 1:
            pin = _pin("#dc2626", "★")
        else:
            pin = _pin("#ea580c", str(i))
        folium.Marker([float(net.lat[node]), float(net.lon[node])],
                      tooltip=f"{lab}", icon=pin).add_to(m)

    m.get_root().html.add_child(folium.Element(_panel_html(legs, total_m, title)))
    m.fit_bounds([[min(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                  [max(p[0] for p in all_pts), max(p[1] for p in all_pts)]])
    m.save(out_html)
    return out_html
