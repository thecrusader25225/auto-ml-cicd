import functions_framework
from googleapiclient import discovery

PROJECT = "myproject-25225"
ZONE = "us-central1-b"
INSTANCE = "automl-instance"

def get_fingerprint(compute):
    res = compute.instances().get(
        project=PROJECT,
        zone=ZONE,
        instance=INSTANCE
    ).execute()
    return res["metadata"]["fingerprint"]
    
@functions_framework.cloud_event
def start_vm(cloud_event):
    data = cloud_event.data
    file_name = data["name"]

    # Only react to _READY file
    if not file_name.endswith("_READY"):
        print("Not a READY file, ignoring")
        return

    # Extract dataset name
    # datasets/data01/_READY → data01
    parts = file_name.split("/")
    dataset = parts[1]

    print(f"Triggering training for dataset: {dataset}")

    compute = discovery.build("compute", "v1")

    # 🔥 Set metadata (pass dataset to VM)
    metadata_body = {
        "items": [
            {
                "key": "dataset",
                "value": dataset
            }
        ]
    }

    # Update metadata first
    compute.instances().setMetadata(
        project=PROJECT,
        zone=ZONE,
        instance=INSTANCE,
        body={
            "fingerprint": get_fingerprint(compute),
            "items": [
                {"key": "dataset", "value": dataset}
            ]
        }
    ).execute()

    # Start VM
    compute.instances().start(
        project=PROJECT,
        zone=ZONE,
        instance=INSTANCE
    ).execute()

    print("VM start triggered")