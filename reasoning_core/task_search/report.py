"""Per-wave yield for task_search. Usage: python -m reasoning_core.task_search.report [glob]

Cross-wave roll-up: one line per wave, the status counter under it. For what a single
worker actually did -- self-checks, denials, where the steps went -- use trajectory.py.
"""
import json,glob,sys,collections,os
# WAVE0 was hardcoded here until WAVE1 existed, at which point this quietly reported
# on a subset. The wave is the directory above the timestamp, whatever it is called.
pat=sys.argv[1] if len(sys.argv)>1 else "runs/*/*/*/*/run.json"
DENIED="prevents you from using"
waves=collections.defaultdict(list)
for p in sorted(glob.glob(pat)):
    d=os.path.dirname(p); j=json.load(open(p))
    ev=os.path.join(d,"events.jsonl"); steps=den=0
    for line in open(ev) if os.path.exists(ev) else []:
        try:e=json.loads(line)
        except:continue
        st=e.get("part",{}).get("state",{})
        if e.get("type")=="tool_use":
            steps+=1
            # An errored call is not a denied one: a failing pytest is the job.
            if st.get("status")!="completed" and DENIED in str(st.get("error","")): den+=1
    arm,wave,stamp=d.split(os.sep)[-4:-1]
    waves[(arm,wave,stamp)].append((j.get("status"),steps,den))
for (arm,wave,stamp),rows in sorted(waves.items(),key=lambda k:k[0][2]):
    n=len(rows); ok=sum(r[0]=="success" for r in rows)
    st=[r[1] for r in rows]; dn=sum(r[2] for r in rows); tot=sum(st)
    print(f"{arm:9s} {wave:6s} {stamp}  n={n:2d} success={ok}  steps med={sorted(st)[n//2]:3d} max={max(st):3d}"
          f"  denied={dn}/{tot} ({100*dn//max(tot,1)}%)")
    for s,c in collections.Counter(r[0] for r in rows).most_common(): print(f"{'':11s}  {c:2d} {s}")
