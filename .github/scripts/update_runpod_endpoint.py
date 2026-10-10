"""Point the RunPod serverless endpoint at a freshly built image.

Required env vars:
    RUNPOD_API_KEY      - RunPod API key (needs account read/write)
    RUNPOD_ENDPOINT_ID  - Serverless endpoint ID to update
    NEW_IMAGE           - Full image reference (name@sha256:...)

Previously this edited the template returned by GraphQL `myself.endpoints`,
reported success, yet the endpoint kept its old pinned digest. Now it:
  1. reads the endpoint config via the REST API,
  2. sets the endpoint's template image via GraphQL (endpoint-owned templates
     are invisible to REST /templates),
  3. re-reads the template and FAILS if the image did not change,
  4. recycles workers (workersMax -> 0 -> original) so none keep the old image,
  5. prints Network Volume / data center / idle timeout config for debugging.
"""
import json
import os
import sys
import time

import requests

REST = "https://rest.runpod.io/v1"

api_key = os.environ.get("RUNPOD_API_KEY")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
image = os.environ.get("NEW_IMAGE")

if not all([api_key, endpoint_id, image]):
    print("ERROR: Missing required environment variables", file=sys.stderr)
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def rest(method, path, body=None):
    resp = requests.request(method, f"{REST}{path}", headers=headers, json=body, timeout=30)
    if not resp.ok:
        print(f"ERROR: {method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        sys.exit(1)
    return resp.json() if resp.text else {}


ep = rest("GET", f"/endpoints/{endpoint_id}")
template_id = ep.get("templateId")
if not template_id:
    print(f"ERROR: endpoint has no templateId: {json.dumps(ep)[:500]}", file=sys.stderr)
    sys.exit(1)

print("===== Endpoint config =====")
# Must be the endpoint the backend calls (EC2 RUNPOD_ENDPOINT_ID). workersMax 0 /
# no volume here usually means the GitHub secret points at a stale endpoint.
for key in ("name", "templateId", "gpuTypeIds", "dataCenterIds", "networkVolumeId",
            "networkVolumeIds", "idleTimeout", "workersMin", "workersMax", "flashboot",
            "scalerType", "scalerValue"):
    if key in ep:
        print(f"  {key}: {ep[key]}")

volume_ids = ep.get("networkVolumeIds") or ([ep["networkVolumeId"]] if ep.get("networkVolumeId") else [])
if not volume_ids:
    print("  ⚠️  No Network Volume attached → models are downloaded on every cold start.")
for vid in volume_ids:
    vol = rest("GET", f"/networkvolumes/{vid}")
    dc = vol.get("dataCenterId")
    print(f"  Network Volume {vid}: name={vol.get('name')} size={vol.get('size')}GB dataCenter={dc}")
    if ep.get("dataCenterIds") and dc not in ep["dataCenterIds"]:
        print(f"  ⚠️  Volume data center {dc} is not in endpoint dataCenterIds → workers can't mount it.")
if (ep.get("idleTimeout") or 0) < 20:
    print("  ⚠️  idleTimeout < 20 s: worker may stop between 15 s realtime windows.")

# Endpoint-owned templates (endpoint created from an image in the UI) are not
# visible to REST /templates (404), but GraphQL can read and save them.
TEMPLATE_FIELDS = "id name imageName dockerArgs volumeInGb containerDiskInGb isServerless env { key value }"


def gql(query):
    resp = requests.post("https://api.runpod.io/graphql", headers=headers,
                         json={"query": query}, timeout=30)
    data = resp.json() if resp.text else {}
    if not resp.ok or data.get("errors"):
        print(f"ERROR: GraphQL -> HTTP {resp.status_code}: {json.dumps(data)[:500]}", file=sys.stderr)
        sys.exit(1)
    return data["data"]


def endpoint_template():
    for e in gql(f"{{ myself {{ endpoints {{ id template {{ {TEMPLATE_FIELDS} }} }} }} }}")["myself"]["endpoints"]:
        if e["id"] == endpoint_id:
            return e["template"]
    print("ERROR: endpoint not visible via GraphQL", file=sys.stderr)
    sys.exit(1)


tmpl = endpoint_template()
if tmpl["id"] != template_id:
    print(f"WARNING: GraphQL template {tmpl['id']} != REST templateId {template_id}")
print(f"\nTemplate {tmpl['id']}: {tmpl['imageName']} -> {image}")

q = json.dumps  # JSON string literals are valid GraphQL string literals
env_gql = ", ".join(f"{{key: {q(e['key'])}, value: {q(e['value'])}}}" for e in tmpl.get("env") or [])
gql(f"""mutation {{ saveTemplate(input: {{
    id: {q(tmpl['id'])}, name: {q(tmpl['name'] or f"meetasr-{tmpl['id']}")},
    imageName: {q(image)}, dockerArgs: {q(tmpl.get('dockerArgs') or '')},
    volumeInGb: {tmpl.get('volumeInGb') or 0}, containerDiskInGb: {tmpl.get('containerDiskInGb') or 10},
    isServerless: {str(tmpl.get('isServerless', True)).lower()}, env: [{env_gql}]
}}) {{ id imageName }} }}""")

current = endpoint_template()["imageName"]
if current != image:
    print(f"ERROR: endpoint still uses {current!r} after update", file=sys.stderr)
    sys.exit(1)
print(f"Verified: endpoint now uses {current}")

# Workers do not roll over to a new image on their own: recycle them.
workers_max = ep.get("workersMax")
if workers_max:
    print(f"Recycling workers (workersMax {workers_max} -> 0 -> {workers_max})...")
    rest("PATCH", f"/endpoints/{endpoint_id}", {"workersMax": 0})
    time.sleep(10)
    rest("PATCH", f"/endpoints/{endpoint_id}", {"workersMax": workers_max})
    print("Workers recycled; new workers will pull the new image.")
