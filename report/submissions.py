import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cb import login, BASE

COMP = 16180
TASKS = {35766: "Main v7", 35767: "Small v7", 36237: "Main", 36241: "Small"}
BOARD = {36237: "Main", 36241: "Small"}
ME = "obarskyi"

s = login()
comp = s.get(f"{BASE}/api/competitions/{COMP}/", timeout=60).json()
phases = {ph["id"]: ph.get("name") for ph in comp.get("phases") or []}
raw = s.get(f"{BASE}/api/submissions/?competition={COMP}&page_size=500", timeout=60).json()
if isinstance(raw, dict):
    raw = raw.get("results", [])
raw = [x for x in raw if x.get("phase") in phases]


def scores(entry):
    out = {}
    for sc in entry.get("scores") or []:
        key = (sc.get("column_key") or sc.get("index") or "").lower()
        try:
            out[key] = float(sc["score"])
        except (TypeError, ValueError):
            pass
        if sc.get("is_primary"):
            out["primary"] = out.get(key)
    return out


rows = []
for x in sorted(raw, key=lambda x: x["created_when"]):
    task = (x.get("task") or {}).get("id") if isinstance(x.get("task"), dict) else x.get("task")
    if task not in TASKS:
        continue
    sc = scores(x)
    rows.append({
        "id": x["id"],
        "date": x["created_when"][:16].replace("T", " "),
        "track": TASKS[task],
        "file": x.get("filename") or "",
        "status": x.get("status"),
        "accuracy": sc.get("accuracy", sc.get("primary")),
        "coverage": sc.get("coverage"),
        "invalid": sc.get("invalid_predictions"),
    })

board = {}
for lb in comp.get("leaderboards", []):
    board = s.get(f"{BASE}/api/leaderboards/{lb['id']}/?secret_key=", timeout=60).json()
    break

ranks = {}
for task, label in BOARD.items():
    best = {}
    for x in board.get("submissions", []):
        if x.get("task") != task:
            continue
        v = scores(x).get("primary")
        if v is None:
            continue
        if x["owner"] not in best or v > best[x["owner"]]:
            best[x["owner"]] = v
    order = sorted(best.items(), key=lambda kv: -kv[1])
    mine = best.get(ME)
    ranks[label] = {
        "teams": len(order),
        "rank": next((i for i, (o, _) in enumerate(order, 1) if o == ME), None),
        "rank_tied_best": (1 + sum(1 for _, v in order if v > mine)) if mine is not None else None,
        "ours": mine,
        "top": order[:5],
    }

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
print(f"snapshot {now}; phases {phases}")
for label, r in ranks.items():
    print(f"{label}: ours {r['ours']}, rank {r['rank']} (first among ties: {r['rank_tied_best']}) of {r['teams']}; top {r['top']}")
print()
print("| id | date (UTC) | track | bundle | accuracy | coverage | invalid |")
print("| --- | --- | --- | --- | --- | --- | --- |")
fmt = lambda v: "---" if v is None else (f"{v:.4f}" if isinstance(v, float) and v % 1 else f"{v:g}")
for r in rows:
    print(f"| {r['id']} | {r['date']} | {r['track']} | `{r['file']}` | {fmt(r['accuracy'])} | {fmt(r['coverage'])} | {fmt(r['invalid'])} |"
          + ("" if r["status"] == "Finished" else f" {r['status']}"))
json.dump({"snapshot": now, "ranks": ranks, "rows": rows},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions.json"), "w"), indent=1)
