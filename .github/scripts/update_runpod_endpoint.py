"""Update RunPod serverless endpoint to use a newly built Docker image.

Creates a new serverless template with the given image, then points the
endpoint to it. Called from CI after each successful image build.

Required env vars:
    RUNPOD_API_KEY      - RunPod API key
    RUNPOD_ENDPOINT_ID  - Serverless endpoint ID to update
    NEW_IMAGE           - Full image reference (name@sha256:...)
    TEMPLATE_NAME       - Name for the new template
"""
import os
import sys
import runpod

api_key = os.environ.get("RUNPOD_API_KEY")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
image = os.environ.get("NEW_IMAGE")
template_name = os.environ.get("TEMPLATE_NAME")

if not all([api_key, endpoint_id, image, template_name]):
    print("ERROR: Missing required environment variables", file=sys.stderr)
    sys.exit(1)

runpod.api_key = api_key

print(f"Creating serverless template {template_name!r} with image {image!r}")
new_template = runpod.create_template(
    name=template_name,
    image_name=image,
    is_serverless=True,
)
template_id = new_template["id"]
print(f"Template created: {template_id}")

runpod.update_endpoint_template(endpoint_id, template_id)
print(f"Endpoint {endpoint_id} updated to template {template_id}")
