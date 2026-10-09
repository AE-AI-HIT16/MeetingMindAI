"""Update RunPod serverless endpoint to use a newly built Docker image.

For endpoints with a bound template, updates the template image in-place.
Uses RunPod GraphQL API directly via requests (no SDK internals dependency).

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


def gql(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data.get("data", {})


# Step 1: get the template ID bound to this endpoint
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
print(f"New image: {image}")

# Step 2: update the bound template's image in-place
data = gql("""
mutation($id: String!, $imageName: String!) {
    saveTemplate(input: {
        id: $id,
        imageName: $imageName,
        isServerless: true
    }) {
        id
        imageName
    }
}
""", {"id": template_id, "imageName": image})

result = data.get("saveTemplate", {})
print(f"Template {result.get('id')} updated -> {result.get('imageName')}")
print("New workers will pull the updated image automatically.")
