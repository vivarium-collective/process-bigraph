"""EC2SSMRayCluster — Ray cluster on EC2, controlled via SSM rather than
Ray's autoscaler. Use as a context manager.

Why this exists in process_bigraph (not just one downstream repo)
-----------------------------------------------------------------
Ray's built-in autoscaler hardcodes assumptions that don't hold on
restricted-egress / minimum-AMI cloud environments:

  - rsync over ssh for ``file_mounts`` (rsync 3.4 ↔ 3.1 protocol drift
    breaks bring-up with no diagnosable error)
  - cached config under /tmp/ray-config-* goes stale across teardown,
    surfacing as "SecurityGroup not found" mid-run
  - assumes a stock-Linux interactive shell (sources ~/.bashrc, runs
    ``bash --login -c -i``); breaks on minimal AMIs

So instead we go around it:

  - ``aws ec2 run-instances`` for head + workers (parallel)
  - SSM agent (already on ECS-AL2 etc.) for control plane — no ssh
  - ``docker run --network host`` so Ray's many ports work without
    SG fiddling, ``--shm-size=10g`` so plasma doesn't fall back to /tmp
  - ``ray start --head`` / ``ray start --address=<head>:6379`` inside
    the same image as everything else
  - All experiment dispatch is done via ``cluster.exec(...)``, which is
    a thin wrapper over ``docker exec`` on the head node — every
    use-case can dispatch its own thing without needing this module
    to know about it

Typical use::

    from process_bigraph.protocols.clusters.ec2_ssm import EC2SSMRayCluster

    with EC2SSMRayCluster(
        image_uri="123.dkr.ecr.us-gov-west-1.amazonaws.com/spatio-flux:abc",
        n_workers=4,
        cluster_id="my-experiment",
        keep_cluster=False,
    ) as cluster:
        # cluster.head_id / .worker_ids / .head_ip / .ssm / .ec2 / .region
        # are all available now.
        cluster.exec_blocking_on_head(
            "python -m my_pkg.run_experiment > /tmp/x.log 2>&1",
            log_path="/tmp/x.log",
            timeout_s=3600,
        )

Install with::

    pip install process-bigraph[ec2-ssm]

The class makes its boto3 import lazy so this module is safe to import
even without that extra installed (the constructor will then error
clearly rather than silently importing nothing).
"""
from __future__ import annotations

import os
import re
import shlex
import time
import urllib.request
from typing import Any, Optional


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError as e:
        raise ImportError(
            "EC2SSMRayCluster requires boto3. Install with: "
            "pip install process-bigraph[ec2-ssm]"
        ) from e


def _instance_arch(instance_type: str) -> str:
    """Returns 'arm64' for AWS Graviton instance types, 'x86_64' otherwise.

    Graviton families end with 'g' before the dot (e.g. c7g.xlarge,
    m6g.large, r7gd.metal, t4g.medium). Older 'a1' family is also ARM
    but that's pre-Graviton and rarely used now.
    """
    family = instance_type.split(".", 1)[0]
    # Strip optional storage suffix like 'd' (e.g. r7gd → r7g) and any
    # trailing 'n'/'en' network qualifier (e.g. c6gn → c6g).
    base = family.rstrip("dn") if family.endswith(("dn", "n", "d")) else family
    # ARM families: <letter><digit>g[<letter>...]. Test the post-digit char.
    # Skip leading letters, then digits, then check next char.
    i = 0
    while i < len(base) and base[i].isalpha():
        i += 1
    while i < len(base) and base[i].isdigit():
        i += 1
    if i < len(base) and base[i] == "g":
        return "arm64"
    if family == "a1":
        return "arm64"
    return "x86_64"


# ---------------------------------------------------------------------------
# IMDSv2 — discover network setup from the submit node we're running on.
# When this module runs *outside* an EC2 instance, callers should pass
# region / subnet_id / sg_id explicitly.
# ---------------------------------------------------------------------------

def _imds_get(path: str) -> str:
    token = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )).read().decode()
    return urllib.request.urlopen(urllib.request.Request(
        f"http://169.254.169.254/latest/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )).read().decode()


def discover_network() -> dict:
    """Discover region/vpc/subnet/sg from the local IMDSv2 endpoint.
    Only works when called from inside an EC2 instance — for off-EC2
    use, pass the values explicitly to ``EC2SSMRayCluster``."""
    region = _imds_get("meta-data/placement/region")
    mac = _imds_get("meta-data/mac")
    return {
        "region": region,
        "vpc_id": _imds_get(f"meta-data/network/interfaces/macs/{mac}/vpc-id"),
        "subnet_id": _imds_get(f"meta-data/network/interfaces/macs/{mac}/subnet-id"),
        "sg_id": _imds_get(
            f"meta-data/network/interfaces/macs/{mac}/security-group-ids"
        ).split()[0],
        "submit_ip": _imds_get("meta-data/local-ipv4"),
    }


# ---------------------------------------------------------------------------
# SSM helpers (low-level building blocks; ``EC2SSMRayCluster`` uses them
# internally but they're intentionally exported so callers can dispatch
# their own one-off commands without going through the class).
# ---------------------------------------------------------------------------

def ssm_run(
    ssm,
    instance_ids: list[str],
    commands: list[str] | str,
    *,
    name: str,
    timeout_s: int = 300,
    poll_interval: float = 3.0,
) -> dict[str, dict]:
    """Send shell commands to instances, wait for all to finish, return invocations.
    Raises RuntimeError on any non-Success status, dumping per-instance
    stdout/stderr first so callers can see what broke without an extra trip."""
    if isinstance(commands, str):
        commands = [commands]
    print(f"  [ssm:{name}] sending to {len(instance_ids)} instance(s)")
    cmd_id = ssm.send_command(
        InstanceIds=instance_ids,
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands, "executionTimeout": [str(timeout_s)]},
        Comment=name[:100],
    )["Command"]["CommandId"]

    deadline = time.time() + timeout_s + 30
    pending = set(instance_ids)
    results: dict[str, dict] = {}
    while pending and time.time() < deadline:
        time.sleep(poll_interval)
        for iid in list(pending):
            try:
                inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=iid)
            except ssm.exceptions.InvocationDoesNotExist:
                continue
            if inv["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                results[iid] = inv
                pending.discard(iid)
                marker = "✓" if inv["Status"] == "Success" else "✗"
                print(f"  [ssm:{name}] {marker} {iid}: {inv['Status']}")
    for iid in pending:
        results[iid] = {"Status": "TimedOut",
                        "StandardErrorContent": "ssm poll timeout"}
        print(f"  [ssm:{name}] ✗ {iid}: TimedOut")

    failed = {iid: r for iid, r in results.items() if r.get("Status") != "Success"}
    if failed:
        for iid, r in failed.items():
            print(f"  [ssm:{name}]   stderr from {iid}:")
            print(f"    {r.get('StandardErrorContent', '')[:1500]}")
            print(f"  [ssm:{name}]   stdout from {iid}:")
            print(f"    {r.get('StandardOutputContent', '')[:1500]}")
        raise RuntimeError(
            f"SSM '{name}' failed on {len(failed)}/{len(instance_ids)} instance(s)")
    return results


def wait_ssm_agents(ssm, instance_ids: list[str], *, timeout_s: int = 240) -> None:
    """Block until SSM agent on every instance has reported Online."""
    deadline = time.time() + timeout_s
    pending = set(instance_ids)
    while pending and time.time() < deadline:
        resp = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": list(pending)}],
        )
        ready = {i["InstanceId"]
                 for i in resp.get("InstanceInformationList", [])
                 if i.get("PingStatus") == "Online"}
        pending -= ready
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"SSM agent not online after {timeout_s}s: {pending}")


def probe_remote_log(
    ssm, instance_id: str, log_path: str, line_cursor: int,
    *, container: str = None,
) -> tuple[str, int]:
    """Fetch new log lines since ``line_cursor`` on a remote instance.
    Returns ``(new_text, updated_cursor)``. Quiet on failure (returns
    empty text and unchanged cursor) so the caller can keep polling.

    If ``container`` is set, the log file is read via ``docker exec``
    inside that container — useful when dispatching long-running work
    inside a container whose logs aren't on the host filesystem.

    The inner script is single-quoted via shlex.quote so HOST bash
    doesn't expand $L / $(...) / $p before docker exec sees them.
    """
    inner = (
        f"L=$(wc -l < {log_path} 2>/dev/null || echo 0); "
        f'if [ "$L" -gt {line_cursor} ]; then '
        f"  sed -n '{line_cursor + 1},$p' {log_path} 2>/dev/null | head -300; "
        f"fi; "
        f'echo "__CURSOR__=$L"'
    )
    if container:
        probe_cmd = f"docker exec {container} bash -c {shlex.quote(inner)}"
    else:
        probe_cmd = f"bash -c {shlex.quote(inner)}"

    try:
        cmd_id = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [probe_cmd]},
            Comment="log-probe",
        )["Command"]["CommandId"]
    except Exception:
        return "", line_cursor

    for _ in range(10):
        time.sleep(0.7)
        try:
            inv = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        except Exception:
            return "", line_cursor
        if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
            content = inv.get("StandardOutputContent", "")
            new_cursor = line_cursor
            text_lines: list[str] = []
            for ln in content.splitlines():
                if ln.startswith("__CURSOR__="):
                    try:
                        new_cursor = int(ln.split("=", 1)[1])
                    except ValueError:
                        pass
                else:
                    text_lines.append(ln)
            return "\n".join(text_lines), new_cursor
    return "", line_cursor


# ---------------------------------------------------------------------------
# EC2 lifecycle helpers (also exported)
# ---------------------------------------------------------------------------

CLUSTER_TAG = "process-bigraph-cluster"


def find_existing_cluster(
    ec2, cluster_id: str, *, cluster_tag: str = CLUSTER_TAG,
) -> tuple[Optional[str], list[str]]:
    """Return (head_id, worker_ids) for a previously-launched cluster
    matching ``cluster_id``, or (None, []) if nothing's running."""
    resp = ec2.describe_instances(Filters=[
        {"Name": f"tag:{cluster_tag}", "Values": [cluster_id]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    head, workers = None, []
    for r in resp["Reservations"]:
        for inst in r["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            if tags.get("process-bigraph-role") == "head":
                head = inst["InstanceId"]
            elif tags.get("process-bigraph-role") == "worker":
                workers.append(inst["InstanceId"])
    return head, workers


def get_private_ips(ec2, instance_ids: list[str]) -> dict[str, str]:
    resp = ec2.describe_instances(InstanceIds=instance_ids)
    ips = {}
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            ips[i["InstanceId"]] = i["PrivateIpAddress"]
    return ips


def ensure_intra_cluster_traffic(ec2, sg_id: str) -> None:
    """Add a self-referencing all-traffic ingress rule to the cluster SG
    so nodes can reach each other on Ray's many ports (6379, plus
    dynamic raylet/object-manager/dashboard ports). Idempotent — if the
    rule already exists, AWS returns InvalidPermission.Duplicate which
    we swallow."""
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "-1",
                "UserIdGroupPairs": [{"GroupId": sg_id}],
            }],
        )
        print(f"  ✓ added intra-cluster ingress rule to {sg_id}")
    except Exception as e:
        msg = str(e)
        if "InvalidPermission.Duplicate" in msg:
            print(f"  ✓ intra-cluster ingress rule already on {sg_id}")
        else:
            print(f"  ⚠ could not add intra-cluster rule to {sg_id}: {msg}")


# ---------------------------------------------------------------------------
# Cluster context manager
# ---------------------------------------------------------------------------

class EC2SSMRayCluster:
    """Context-manager-shaped Ray cluster on EC2, controlled via SSM.

    Usage::

        with EC2SSMRayCluster(image_uri="...", n_workers=4) as cluster:
            cluster.exec_blocking_on_head(
                "python -u -m my_pkg.run > /tmp/log 2>&1",
                log_path="/tmp/log",
                timeout_s=1800,
            )

    Lifetime semantics:

    - ``__enter__``: launches (or reuses by ``cluster_id`` tag) head +
      ``n_workers`` workers, waits SSM agents online, applies an
      intra-cluster SG ingress rule (idempotent), pulls the image to
      every node, starts ``ray --head`` on head and ``ray start
      --address=...`` on workers, blocks until all nodes register.
    - ``__exit__``: terminates every instance unless ``keep_cluster``
      was set. Same flag that lets you run multiple sweeps over one
      brought-up cluster — useful when bring-up dominates wall time.
    """

    def __init__(
        self,
        *,
        image_uri: str,
        n_workers: int,
        cluster_id: Optional[str] = None,
        keep_cluster: bool = False,
        head_instance_type: str = "m5.2xlarge",
        worker_instance_type: str = "m5.4xlarge",
        iam_profile: str = "ray-process-bigraph-node",
        baked_ami_id: Optional[str] = None,
        # Network: when None, discover from IMDSv2 (must be running on EC2).
        region: Optional[str] = None,
        subnet_id: Optional[str] = None,
        sg_id: Optional[str] = None,
        # Defaults to ``CLUSTER_TAG`` constant; override only if you want
        # to share infra naming with another tooling layer.
        cluster_tag: str = CLUSTER_TAG,
        container_name: str = "process_bigraph_ray",
    ):
        boto3 = _require_boto3()
        self._boto3 = boto3
        self._image_uri = image_uri
        self._n_workers = int(n_workers)
        self._cluster_id = cluster_id or f"pb-cluster-{int(time.time())}"
        self._keep_cluster = bool(keep_cluster)
        self._head_type = head_instance_type
        self._worker_type = worker_instance_type
        self._iam_profile = iam_profile
        self._baked_ami_id = baked_ami_id
        self._cluster_tag = cluster_tag
        self._container = container_name

        # Resolve network — explicit args win, else IMDSv2.
        if region and subnet_id and sg_id:
            self._region = region
            self._subnet_id = subnet_id
            self._sg_id = sg_id
        else:
            net = discover_network()
            self._region = region or net["region"]
            self._subnet_id = subnet_id or net["subnet_id"]
            self._sg_id = sg_id or net["sg_id"]

        self._ec2 = boto3.client("ec2", region_name=self._region)
        self._ssm = boto3.client("ssm", region_name=self._region)

        # Populated by __enter__.
        self._head_id: Optional[str] = None
        self._worker_ids: list[str] = []
        self._head_ip: Optional[str] = None
        self._launched_now = False

    # ------------------------------------------------------------------ #
    # public read-only properties
    # ------------------------------------------------------------------ #

    @property
    def region(self) -> str: return self._region
    @property
    def head_id(self) -> str:
        if not self._head_id:
            raise RuntimeError("Cluster not entered — use 'with EC2SSMRayCluster(...) as c:'")
        return self._head_id
    @property
    def worker_ids(self) -> list[str]: return list(self._worker_ids)
    @property
    def head_ip(self) -> str:
        if not self._head_ip:
            raise RuntimeError("Cluster not entered yet")
        return self._head_ip
    @property
    def all_ids(self) -> list[str]:
        return ([self._head_id] if self._head_id else []) + list(self._worker_ids)
    @property
    def ssm(self): return self._ssm
    @property
    def ec2(self): return self._ec2
    @property
    def container(self) -> str: return self._container
    @property
    def image_uri(self) -> str: return self._image_uri

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "EC2SSMRayCluster":
        ensure_intra_cluster_traffic(self._ec2, self._sg_id)

        ami_id = self._baked_ami_id
        if not ami_id:
            # Auto-detect architecture from instance type so the SSM
            # parameter we pick matches what we're about to launch.
            # AWS EC2 ARM-family prefixes (Graviton 1/2/3) all end in 'g'
            # followed by '.' (e.g. c7g.xlarge, m6g.large, t4g.medium,
            # r7g.metal). Everything else is x86_64.
            #
            # Without this, launching c7g/m7g/etc. against the default
            # x86 AMI fails immediately with:
            #   InvalidParameterValue: architecture 'arm64' of the
            #   specified instance type does not match the architecture
            #   'x86_64' of the specified AMI.
            head_arch = _instance_arch(self._head_type)
            worker_arch = _instance_arch(self._worker_type)
            if head_arch != worker_arch:
                raise ValueError(
                    f"Mixed-arch cluster not supported: head={self._head_type} "
                    f"({head_arch}) vs worker={self._worker_type} ({worker_arch}). "
                    f"Pass baked_ami_id explicitly if you really want this.")
            if head_arch == "arm64":
                ssm_param = "/aws/service/ecs/optimized-ami/amazon-linux-2/arm64/recommended/image_id"
            else:
                ssm_param = "/aws/service/ecs/optimized-ami/amazon-linux-2/recommended/image_id"
            ami_id = self._ssm.get_parameter(Name=ssm_param)["Parameter"]["Value"]
            print(f"→ arch={head_arch}  ami_ssm_param={ssm_param}")
        print(f"→ ami={ami_id}")

        head_id, worker_ids = find_existing_cluster(
            self._ec2, self._cluster_id, cluster_tag=self._cluster_tag)
        launched_now = False
        if head_id and len(worker_ids) >= self._n_workers:
            print(f"→ reusing existing cluster: head={head_id} workers={worker_ids}")
        else:
            if head_id or worker_ids:
                print(f"→ partial existing cluster, terminating: "
                      f"head={head_id} workers={worker_ids}")
                self._terminate(([head_id] if head_id else []) + worker_ids)
                time.sleep(5)
            print(f"→ launching {self._n_workers + 1} instances")
            head_id, worker_ids = self._launch_instances(ami_id)
            launched_now = True
            print(f"→ launched: head={head_id} workers={worker_ids}")

        self._head_id = head_id
        self._worker_ids = worker_ids
        self._launched_now = launched_now

        try:
            if launched_now:
                print("→ waiting for instances running")
                self._ec2.get_waiter("instance_running").wait(
                    InstanceIds=self.all_ids,
                    WaiterConfig={"Delay": 5, "MaxAttempts": 60})
                print("→ waiting for SSM agents online")
                wait_ssm_agents(self._ssm, self.all_ids)

            ips = get_private_ips(self._ec2, self.all_ids)
            self._head_ip = ips[head_id]
            print(f"→ head_ip={self._head_ip}")

            print("→ docker pull on all nodes (idempotent)")
            self._docker_pull_all()
            print("→ starting ray head")
            self._start_ray_head()
            print("→ starting ray workers")
            self._start_ray_workers()
            print("→ waiting for workers to register")
            self._wait_workers_registered()
        except Exception:
            # Bring-up failed mid-stream: dump diagnostics and tear down
            # so the caller doesn't have to.
            try:
                self.collect_node_diag("bringup-failure")
            except Exception:
                pass
            if not self._keep_cluster:
                self._terminate(self.all_ids)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            print("→ collecting diagnostics from cluster nodes (exception in body)")
            try:
                self.collect_node_diag("body-failure")
            except Exception as e:
                print(f"  diag collection failed: {e}")

        if self._keep_cluster:
            print(f"→ keeping cluster (keep_cluster=True). "
                  f"Tear down later: aws ec2 terminate-instances --region "
                  f"{self._region} --instance-ids {' '.join(self.all_ids)}")
        else:
            self._terminate(self.all_ids)
        # Don't suppress exceptions
        return None

    # ------------------------------------------------------------------ #
    # public dispatch helpers — for callers who want to run experiments
    # on the cluster
    # ------------------------------------------------------------------ #

    def exec_blocking_on_head(
        self, command: str, *,
        log_path: Optional[str] = None,
        timeout_s: int = 3600,
        poll_interval: float = 30.0,
    ) -> dict:
        """Run ``command`` (a bash string) inside the head node's
        container synchronously. Returns the final SSM invocation dict.

        If ``log_path`` is given, every ``poll_interval`` seconds we
        tail that file via SSM probes and stream new lines to local
        stdout. Whether your command actually writes to that file is
        up to you — the wrapper does NOT auto-redirect — because
        multi-stage commands (run, capture rc, upload log, sync
        results) typically need control over which stage's output
        goes where. A simple pattern::

            command = (
                f"set +e; "
                f": > {LOG}; "
                f"my_program > {LOG} 2>&1; rc=$?; "
                f"echo '=== exit: '$rc >> {LOG}; "
                f"aws s3 cp {LOG} s3://...; "
                f"exit $rc"
            )
            cluster.exec_blocking_on_head(command, log_path=LOG)

        Raises RuntimeError on non-Success terminal status, dumping the
        full log file (if provided) plus SSM stdout/stderr first.
        """
        wrapped = f"docker exec {self._container} bash -c {shlex.quote(command)}"

        print("  → submitting command (streaming log every "
              f"{poll_interval:.0f}s)" if log_path else "  → submitting command")
        cmd_id = self._ssm.send_command(
            InstanceIds=[self.head_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": ["set +e", wrapped],
                "executionTimeout": [str(timeout_s)],
            },
            Comment="exec-on-head",
        )["Command"]["CommandId"]
        print(f"  → cmd_id={cmd_id}")

        deadline = time.time() + timeout_s + 120
        cursor = 0
        last_status = None
        inv: dict = {}
        while time.time() < deadline:
            time.sleep(poll_interval)
            if log_path:
                new_text, cursor = probe_remote_log(
                    self._ssm, self.head_id, log_path, cursor,
                    container=self._container)
                if new_text.strip():
                    for line in new_text.splitlines():
                        print(f"  exp│ {line}")
            try:
                inv = self._ssm.get_command_invocation(
                    CommandId=cmd_id, InstanceId=self.head_id)
                status = inv.get("Status", "InProgress")
            except self._ssm.exceptions.InvocationDoesNotExist:
                status = "Pending"
            except Exception:
                status = "Unknown"
            if status != last_status:
                print(f"  → status: {status}")
                last_status = status
            if status in ("Success", "Failed", "Cancelled", "TimedOut"):
                if log_path:
                    new_text, _ = probe_remote_log(
                        self._ssm, self.head_id, log_path, cursor,
                        container=self._container)
                    for line in new_text.splitlines():
                        if line.strip():
                            print(f"  exp│ {line}")
                if status != "Success":
                    self._dump_failure(inv, log_path)
                    raise RuntimeError(f"exec-on-head {status}")
                return inv
        raise RuntimeError(f"command didn't terminate within {timeout_s}s")

    def collect_node_diag(self, label: str) -> None:
        """Best-effort dump of container state on every node. Stays in
        the calling process's stdout — useful for failure post-mortems."""
        try:
            results = ssm_run(self._ssm, self.all_ids, [
                "echo '=== docker ps -a ===' && docker ps -a",
                f"echo '=== {self._container} logs (tail 100) ===' "
                f"&& docker logs --tail 100 {self._container} 2>&1 || true",
                "echo '=== systemd docker ===' && systemctl is-active docker",
                "echo '=== disk ===' && df -h / /var/lib/docker 2>&1 | head -5",
                "echo '=== ports listening ===' && "
                "(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | head -20",
            ], name=f"diag-{label}", timeout_s=60)
            for iid, inv in results.items():
                out = inv.get("StandardOutputContent", "").strip()
                print(f"\n  --- diag {label} on {iid} ---")
                for line in out.splitlines()[:80]:
                    print(f"    {line}")
        except Exception as e:
            print(f"  diag collection failed: {e}")

    # ------------------------------------------------------------------ #
    # private bring-up steps
    # ------------------------------------------------------------------ #

    def _launch_instances(self, ami_id: str) -> tuple[str, list[str]]:
        base_kwargs = dict(
            ImageId=ami_id,
            SubnetId=self._subnet_id,
            SecurityGroupIds=[self._sg_id],
            IamInstanceProfile={"Name": self._iam_profile},
        )
        head_resp = self._ec2.run_instances(
            **base_kwargs,
            InstanceType=self._head_type,
            MinCount=1, MaxCount=1,
            BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                                  "Ebs": {"VolumeSize": 80, "VolumeType": "gp3"}}],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"{self._cluster_id}-head"},
                    {"Key": self._cluster_tag, "Value": self._cluster_id},
                    {"Key": "process-bigraph-role", "Value": "head"},
                ],
            }],
        )
        head_id = head_resp["Instances"][0]["InstanceId"]
        worker_resp = self._ec2.run_instances(
            **base_kwargs,
            InstanceType=self._worker_type,
            MinCount=self._n_workers, MaxCount=self._n_workers,
            BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                                  "Ebs": {"VolumeSize": 60, "VolumeType": "gp3"}}],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"{self._cluster_id}-worker"},
                    {"Key": self._cluster_tag, "Value": self._cluster_id},
                    {"Key": "process-bigraph-role", "Value": "worker"},
                ],
            }],
        )
        worker_ids = [i["InstanceId"] for i in worker_resp["Instances"]]
        return head_id, worker_ids

    def _terminate(self, instance_ids: list[str]) -> None:
        if not instance_ids:
            return
        print(f"→ terminating {len(instance_ids)} instance(s): {instance_ids}")
        self._ec2.terminate_instances(InstanceIds=instance_ids)

    def _docker_pull_all(self) -> None:
        """Pull the worker image from ECR on every instance.

        Quirks (learned the hard way on AL2 + SSM):

        - SSM has no TTY → ``docker login`` fails. Use
          amazon-ecr-credential-helper as docker's credsStore instead.
        - SSM's PATH on AL2 doesn't have ``aws``. Use the helper directly.
        - Cloud-init's UserData yum can hold the rpm DB lock when our
          SSM commands first arrive. Wait for it to release.
        - Some baked AMIs end up with a 0-byte helper binary (yum says
          installed but the file is empty) — detect and force reinstall.
        - Need ``HOME=/root`` so docker reads /root/.docker/config.json.
        """
        ecr_host = self._image_uri.split("/", 1)[0]
        config_json = '{"credHelpers": {"' + ecr_host + '": "ecr-login"}}'
        ssm_run(self._ssm, self.all_ids, [
            "set +e",
            "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "export HOME=/root",
            "for _ in $(seq 1 30); do "
            "    pgrep -x yum >/dev/null 2>&1 || break; sleep 2; "
            "done",
            "yum install -y amazon-ecr-credential-helper >/dev/null",
            "echo \"=== yum install exit: $? ===\"",
            "HELPER=/usr/bin/docker-credential-ecr-login",
            "if [[ ! -s \"$HELPER\" ]]; then "
            "    echo '  helper is 0 bytes, force-reinstalling'; "
            "    rm -f \"$HELPER\"; "
            "    yum reinstall -y amazon-ecr-credential-helper >/dev/null 2>&1 "
            "        || yum install -y amazon-ecr-credential-helper >/dev/null; "
            "fi",
            "echo '--- helper binary ---'",
            "ls -la /usr/bin/docker-credential-ecr-login 2>&1",
            "[[ -s /usr/bin/docker-credential-ecr-login ]] "
            "    || { echo '  STILL 0 BYTES — bailing'; exit 1; }",
            "docker-credential-ecr-login version 2>&1; echo \"  version exit: $?\"",
            "echo '--- IMDS instance role ---'",
            "TOKEN=$(curl -fsS -m 5 -X PUT "
            "    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' "
            "    http://169.254.169.254/latest/api/token); "
            "curl -fsS -m 5 -H \"X-aws-ec2-metadata-token: $TOKEN\" "
            "    http://169.254.169.254/latest/meta-data/iam/info 2>&1; "
            "echo \"  imds exit: $?\"",
            "echo '--- helper probe (separated streams) ---'",
            f"echo '{ecr_host}' > /tmp/probe-in",
            "docker-credential-ecr-login get < /tmp/probe-in "
            "    > /tmp/probe-out 2> /tmp/probe-err",
            "echo \"  helper exit: $?\"",
            "echo '  stdout:'; sed 's/^/    /' /tmp/probe-out | head -5",
            "echo '  stderr:'; sed 's/^/    /' /tmp/probe-err | head -10",
            "echo '--- docker config ---'",
            "mkdir -p /root/.docker",
            f"echo '{config_json}' > /root/.docker/config.json",
            "cat /root/.docker/config.json",
            "echo \"  HOME=$HOME  USER=$USER  "
            "DOCKER_CONFIG=${DOCKER_CONFIG:-unset}\"",
            "echo '--- docker pull ---'",
            f"docker pull {self._image_uri} 2>&1; rc=$?; "
            "echo \"  pull exit: $rc\"",
            f"docker image inspect {self._image_uri} >/dev/null 2>&1 "
            "    && echo 'pulled-ok' || echo 'NOT-PULLED'",
            f"docker image inspect {self._image_uri} >/dev/null 2>&1",
        ], name="docker-pull", timeout_s=600)

    def _start_ray_head(self) -> None:
        ssm_run(self._ssm, [self.head_id], [
            "set -e",
            f"docker rm -f {self._container} 2>/dev/null || true",
            # --network host: Ray uses many random ports (gcs, raylet,
            #   object manager, dashboard agent). Host network avoids
            #   port-mapping each one.
            # --shm-size=10g: plasma object store; default 64MB falls
            #   back to /tmp/ray which breaks at scale.
            # --restart unless-stopped: survives daemon restart.
            # ray start --block keeps PID 1 alive so docker doesn't exit.
            f"docker run -d --name {self._container} --network host "
            f"  --shm-size=10g --restart unless-stopped "
            f"  --entrypoint ray {self._image_uri} "
            f"  start --head --port=6379 --dashboard-host=0.0.0.0 --block",
            "sleep 5",
            f"docker logs --tail 30 {self._container}",
        ], name="start-head", timeout_s=180)

    def _start_ray_workers(self) -> None:
        ssm_run(self._ssm, self._worker_ids, [
            "set -e",
            f"docker rm -f {self._container} 2>/dev/null || true",
            f"docker run -d --name {self._container} --network host "
            f"  --shm-size=10g --restart unless-stopped "
            f"  --entrypoint ray {self._image_uri} "
            f"  start --address={self._head_ip}:6379 --block",
            "sleep 5",
            f"docker logs --tail 30 {self._container}",
        ], name="start-workers", timeout_s=180)

    def _wait_workers_registered(self, *, timeout_s: int = 180) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                res = ssm_run(self._ssm, [self.head_id], [
                    f"docker exec {self._container} ray status 2>&1 | head -50",
                ], name="ray-status", timeout_s=30)
                out = res[self.head_id]["StandardOutputContent"]
                # Count distinct hex node IDs.
                ids = set(re.findall(r"node_[a-f0-9]{20,}", out))
                print(f"  ray status nodes seen: {len(ids)} "
                      f"(need {self._n_workers + 1})")
                if len(ids) >= self._n_workers + 1:
                    print(f"  ✓ all nodes registered")
                    return
            except Exception as e:
                print(f"  ray status query failed: {e}")
            time.sleep(5)
        raise RuntimeError(
            f"workers didn't register within {timeout_s}s")

    def _dump_failure(self, inv: dict, log_path: Optional[str]) -> None:
        """On non-Success terminal status, dump the full remote log
        (if a log path was given) plus SSM's own stdout/stderr."""
        if log_path:
            print()
            print(f"  ═══ FULL {log_path} ═══")
            try:
                full_text, _ = probe_remote_log(
                    self._ssm, self.head_id, log_path,
                    line_cursor=0, container=self._container)
                if full_text.strip():
                    for line in full_text.splitlines()[-200:]:
                        print(f"  exp│ {line}")
                else:
                    print(f"  ({log_path} empty — wrapper bash died "
                          f"before the program ran. See SSM stderr below.)")
            except Exception as e:
                print(f"  (couldn't fetch {log_path}: {e})")

        print()
        print("  ═══ SSM stderr (wrapper bash) ═══")
        stderr_content = inv.get("StandardErrorContent", "")
        if stderr_content.strip():
            for line in stderr_content.splitlines()[-50:]:
                print(f"  ssm│ {line}")
        else:
            print("  (empty)")
        print("  ═══ SSM stdout (wrapper bash) ═══")
        stdout_content = inv.get("StandardOutputContent", "")
        if stdout_content.strip():
            for line in stdout_content.splitlines()[-50:]:
                print(f"  ssm│ {line}")
        else:
            print("  (empty)")
