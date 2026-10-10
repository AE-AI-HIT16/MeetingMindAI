"""Point the RunPod serverless endpoint at a freshly built image.

Required env vars:
    RUNPOD_API_KEY      - RunPod API key (needs account read/write)
    RUNPOD_ENDPOINT_ID  - Serverless endpoint ID to update
    NEW_IMAGE           - Full image reference (name@sha256:...)

Previously this edited the template returned by GraphQL `myself.endpoints`,
reported success, yet the endpoint kept its old pinned digest. Now it:
  1. reads the endpoint's own templateId via the REST API,
  2. sets that template's image,
  3. re-reads the endpoint and FAILS if the image did not change,
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


def get_endpoint():
    return rest("GET", f"/endpoints/{endpoint_id}?includeTemplate=true")


ep = get_endpoint()
template_id = ep.get("templateId") or (ep.get("template") or {}).get("id")
old_image = (ep.get("template") or {}).get("imageName")
if not template_id:
    print(f"ERROR: endpoint has no templateId: {json.dumps(ep)[:500]}", file=sys.stderr)
    sys.exit(1)

print("===== Endpoint config =====")
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

print(f"\nTemplate {template_id}: {old_image} -> {image}")
rest("PATCH", f"/templates/{template_id}", {"imageName": image})

ep = get_endpoint()
current = (ep.get("template") or {}).get("imageName")
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
