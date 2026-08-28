"""Per-wave yield for task_search. Usage: python -m reasoning_core.task_search.report [glob]"""
import json,glob,sys,collections,os
pat=sys.argv[1] if len(sys.argv)>1 else "runs/ts_*/WAVE0/*/*/run.json"
waves=collections.defaultdict(list)
for p in sorted(glob.glob(pat)):
    d=os.path.dirname(p); j=json.load(open(p))
    ev=os.path.join(d,"events.jsonl"); steps=den=0
    for line in open(ev) if os.path.exists(ev) else []:
        try:e=json.loads(line)
        except:continue
        pt=e.get("part",{})
        if e.get("type")=="tool_use":
            steps+=1
            if pt.get("state",{}).get("status")=="error": den+=1
    arm=d.split("runs/")[1].split("/")[0]; wave=d.split("/WAVE0/")[1].split("/")[0]
    waves[(arm,wave)].append((j.get("status"),steps,den))
for (arm,wave),rows in sorted(waves.items(),key=lambda k:k[0][1]):
    n=len(rows); ok=sum(r[0]=="success" for r in rows)
    st=[r[1] for r in rows]; dn=sum(r[2] for r in rows); tot=sum(st)
    print(f"{arm:9s} {wave}  n={n:2d} success={ok}  steps med={sorted(st)[n//2]:3d} max={max(st):3d}"
          f"  denied={dn}/{tot} ({100*dn//max(tot,1)}%)")
    for s,c in collections.Counter(r[0] for r in rows).most_common(): print(f"{'':11s}  {c:2d} {s}")
