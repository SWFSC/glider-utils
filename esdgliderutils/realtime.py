"""
Scraping and organizing ESD glider data from the SFMC
"""

import os
import logging
import re
from subprocess import call, run

from google.cloud import secretmanager
import google_crc32c

logger = logging.getLogger(__name__)


def access_secret_version(project_id, secret_id, version_id = 'latest'):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    
    https://github.com/googleapis/python-secret-manager/blob/main/samples/snippets/access_secret_version.py
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
        logging.error("Data corruption detected.")
        return response

    # Print the secret payload.
    #
    # WARNING: Do not print the secret in a production environment - this
    # snippet is showing how to access the secret material.
    # payload = response.payload.data.decode("UTF-8")
    # print("Plaintext: {}".format(payload))

    return response.payload.data.decode("UTF-8")


def rt_files_mgmt(sfmc_ext_all, ext_regex, subdir_name, local_path, bucket_path):
    """
    Copy real-time files from the local sfmc folder (local_path)
    to their subdirectory (subdir_path), 
    and then rsync to their place in the bucket (bucket_path)

    ext_regex_path does include * for copying files (eg is '.[st]bd')
    """
    
    if (any(re.search(ext_regex, i) for i in sfmc_ext_all)):
        # Check paths
        if not os.path.isdir(local_path):
            logging.error(f'Necessary path ({local_path}) does not exist')
            return

        subdir_path = os.path.join(local_path, subdir_name)
        if not os.path.isdir(subdir_path):
            logging.error(f'Necessary path ({subdir_path}) does not exist')
            return

        logging.info(f'Moving {subdir_name} files to their local subdirectory')
        ext_regex_path = os.path.join(local_path, f'*{ext_regex}')
        logging.debug(f'Regex extension path: {ext_regex_path}')
        logging.debug(f'Local subdirectory: {subdir_path}')
        retcode_tmp = call(f'cp {ext_regex_path} {subdir_path}', 
            shell = True)

        logging.info(f'Rsyncing {subdir_name} subdirectory with bucket directory')
        logging.debug(f'Bucket directory: {bucket_path}')
        retcode = run(['gcloud', 'storage', 'rsync', subdir_path, bucket_path], 
            capture_output = True)
        if retcode.returncode != 0:
            logging.error(f'Error copying {subdir_name} files to bucket')
            logging.error(f'Args: {retcode.args}')
            logging.error(f'stderr: {retcode.stderr}')
            return
        else:
            logging.info(f'Successfully copied {subdir_name} files to {bucket_path}')
            logging.debug(f'Args: {retcode.args}')
            logging.debug(f'stderr: {retcode.stdout}')
    else: 
        logging.info(f'No {subdir_name} files to copy')