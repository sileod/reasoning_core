from reasoning_core import list_tasks, get_task
import time
for t in list_tasks():
    if t in done:
        continue
    t0=time.time()
    print(t.ljust(30, '.'), end="")
    try:
        get_task(t).validate()
        t=time.time()-t0
        done[t]=t
        print(t)
    except Exception as e:
        print(e)
print('Done')