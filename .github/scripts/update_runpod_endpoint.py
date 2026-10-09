"""Update RunPod serverless endpoint image by updating its bound template.

Required env vars:
    RUNPOD_API_KEY      - RunPod API key
    RUNPOD_ENDPOINT_ID  - Serverless endpoint ID to update
    NEW_IMAGE           - Full image reference (name@sha256:...)
"""
import os
import sys
import json
import requests

GRAPHQL_URL = "https://api.runpod.io/graphql"

api_key = os.environ.get("RUNPOD_API_KEY")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
image = os.environ.get("NEW_IMAGE")

if not all([api_key, endpoint_id, image]):
    print("ERROR: Missing required environment variables", file=sys.stderr)
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def gql(query):
    resp = requests.post(GRAPHQL_URL, headers=headers,
                         json={"query": query}, timeout=30)
    if not resp.ok:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    if data.get("errors"):
        print(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}", file=sys.stderr)
        sys.exit(1)
    return data.get("data", {})


# Step 1: get endpoint + its bound template's full details
data = gql("""
{
    myself {
        endpoints {
            id
            template {
                id
                name
                imageName
                dockerArgs
                volumeInGb
                containerDiskInGb
                isServerless
                env {
                    key
                    value
                }
            }
        }
    }
}
""")

endpoints = data.get("myself", {}).get("endpoints", [])
tmpl = None
for ep in endpoints:
    if ep.get("id") == endpoint_id:
        tmpl = ep.get("template")
        break

if not tmpl:
    print(f"ERROR: endpoint {endpoint_id} not found or returned no template", file=sys.stderr)
    sys.exit(1)

template_id   = tmpl["id"]
name          = tmpl.get("name") or f"meetasr-{template_id}"
docker_args   = tmpl.get("dockerArgs") or ""
volume_gb     = tmpl.get("volumeInGb") or 0
disk_gb       = tmpl.get("containerDiskInGb") or 10
is_serverless = tmpl.get("isServerless", True)
env_list      = tmpl.get("env") or []

print(f"Bound template: {template_id!r} name={name!r}")
print(f"  dockerArgs={docker_args!r}  volumeInGb={volume_gb}  containerDiskInGb={disk_gb}")
print(f"  env vars: {[e['key'] for e in env_list]}")
print(f"New image: {image}")

# Build env array string for GraphQL
env_gql = ", ".join(
    f'{{key: "{e["key"]}", value: "{e["value"]}"}}'
    for e in env_list
)

# Step 2: update the template image in-place, keeping all other fields
data = gql(f"""
mutation {{
    saveTemplate(input: {{
        id: "{template_id}",
        name: "{name}",
        imageName: "{image}",
        dockerArgs: "{docker_args}",
        volumeInGb: {volume_gb},
        containerDiskInGb: {disk_gb},
        isServerless: {str(is_serverless).lower()},
        env: [{env_gql}]
    }}) {{
        id
        imageName
    }}
}}
""")

result = data.get("saveTemplate", {})
print(f"Template {result.get('id')} updated -> {result.get('imageName')}")
print("New workers will pull the updated image automatically.")
