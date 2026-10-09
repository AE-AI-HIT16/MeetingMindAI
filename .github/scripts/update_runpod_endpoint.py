"""Update RunPod serverless endpoint image by updating its bound template.

Strategy:
1. Query endpoint to get its bound templateId
2. Query that template's full details (name, containerDiskInGb, etc.)
3. Re-save the template with the new imageName, keeping all other fields

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


# Step 1: get endpoint's bound templateId
data = gql("""
{
    myself {
        endpoints {
            id
            templateId
        }
    }
}
""")
endpoints = data.get("myself", {}).get("endpoints", [])
template_id = None
for ep in endpoints:
    if ep.get("id") == endpoint_id:
        template_id = ep.get("templateId")
        break

if not template_id:
    print(f"ERROR: endpoint {endpoint_id} not found or has no templateId", file=sys.stderr)
    sys.exit(1)

print(f"Bound template: {template_id}")

# Step 2: get the template's current fields (name is required by saveTemplate)
data = gql(f"""
{{
    myself {{
        podTemplates {{
            id
            name
            containerDiskInGb
            imageName
            isServerless
        }}
    }}
}}
""")
templates = data.get("myself", {}).get("podTemplates", [])
tmpl = next((t for t in templates if t.get("id") == template_id), None)

if tmpl:
    name = tmpl["name"]
    disk = tmpl.get("containerDiskInGb") or 10
    print(f"Template name: {name!r}, disk: {disk}GB")
else:
    # podTemplates doesn't expose serverless templates — use sensible defaults
    name = f"meetasr-{template_id}"
    disk = 10
    print(f"Template not in podTemplates; using defaults name={name!r} disk={disk}GB")

# Step 3: update the template image in-place
data = gql(f"""
mutation {{
    saveTemplate(input: {{
        id: "{template_id}",
        name: "{name}",
        imageName: "{image}",
        isServerless: true,
        containerDiskInGb: {disk}
    }}) {{
        id
        imageName
    }}
}}
""")

result = data.get("saveTemplate", {})
print(f"Template {result.get('id')} updated -> {result.get('imageName')}")
print("New workers will pull the updated image automatically.")
