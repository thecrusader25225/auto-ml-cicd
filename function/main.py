from googleapiclient import discovery

def start_vm(event, context):
    file_name = event['name']
    dataset = file_name.split('/')[1]

    print(f"Dataset detected: {dataset}")

    compute = discovery.build('compute', 'v1')

    project = 'YOUR_PROJECT_ID'
    zone = 'us-central1-a'
    instance = 'ml-vm'

    compute.instances().start(
        project=project,
        zone=zone,
        instance=instance
    ).execute()