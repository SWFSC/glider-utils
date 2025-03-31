"""
Functions for interacting with GCP
"""

import os
import subprocess
import logging
import google_crc32c
from google.cloud import secretmanager

_log = logging.getLogger(__name__)


def access_secret_version(project_id, secret_id, version_id = 'latest'):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    Originally from
    https://github.com/googleapis/python-secret-manager/blob/main/samples/snippets/access_secret_version.py

    TODO: update this to follow current sample code from Google:
    https://github.com/googleapis/google-cloud-python/blob/main/packages/google-cloud-secret-manager/samples/generated_samples/secretmanager_v1_generated_secret_manager_service_access_secret_version_sync.py

    """

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})

    # Verify payload checksum.
    crc32c = google_crc32c.Checksum()
    crc32c.update(response.payload.data)
    if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
        _log.error("Data corruption detected.")
        return response

    # Print the secret payload.
    #
    # WARNING: Do not print the secret in a production environment - this
    # snippet is showing how to access the secret material.
    # payload = response.payload.data.decode("UTF-8")
    # print("Plaintext: {}".format(payload))

    return response.payload.data.decode("UTF-8")


#---------------------------------------
# GCS bucket mount management
# NOTE: these functions are from
#  https://github.com/us-amlr/shaip/blob/main/shaip/utils.py
def gcs_unmount_bucket(mountpoint):
    """
    Run the command to unmount a bucket mounted at 'mountpoint' using gcsfuse
    mountpoint must be a string
    https://cloud.google.com/storage/docs/gcs-fuse
    """
    subprocess.run(["fusermount", "-u",  mountpoint])

    return 0

def gcs_mount_bucket(bucket, mountpoint, ro = False):
    """
    Run the command to mount a bucket 'bucket' at 'mountpoint' using gcsfuse
    Command is run with '--implicit-dirs' argument.
    https://cloud.google.com/storage/docs/gcs-fuse

    Parameters
    ----------
    bucket : str
        Name of bucket ot mount. Eg, 'amlr-gliders-imagery-raw-dev'

    mountpoint : str
        Path of where to mount 'bucket'. Eg, '.../amlr-gliders-imagery-proc-dev'

    ro : boolean
        Indicates if bucket should be mounted as read only

    """
    # Make mountpoint, if necessary
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)

    # Unmount bucket, just in case
    gcs_unmount_bucket(mountpoint)

    # Mount bucket using gcsfuse
    cmd = ["gcsfuse", "--implicit-dirs", bucket, mountpoint]
    if ro:
        cmd[2:2] = ["-o", "ro"]
    subprocess.run(cmd)

    return 0
