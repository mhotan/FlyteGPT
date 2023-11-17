from flytekit import task, workflow, Resources
import time

@task(
    requests=Resources(cpu="14", mem="60Gi", gpu="1"),
    limits=Resources(cpu="14", mem="60Gi", gpu="1"),
)
def sleep():
    time.sleep(60 * 60 * 24) # 1 day

@workflow
def workflow():
    sleep()