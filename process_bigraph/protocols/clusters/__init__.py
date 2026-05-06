"""Cluster context managers for distributed process_bigraph runs.

Each module here defines one context-manager-shaped cluster type:

    with EC2SSMRayCluster(...) as cluster:
        pool = get_or_create_pool(MyActor, {...}, size=72, cluster=cluster)
        ...

Cluster modules import their cloud SDKs lazily so this package itself
imports cleanly without any cloud extras installed. Install only what
you need:

    pip install process-bigraph[ec2-ssm]   # boto3 for EC2SSMRayCluster

See ``doc/distributed_lifecycles.md`` for how clusters fit into the
``cluster ⊃ pool ⊃ session ⊃ tick`` layering.
"""
