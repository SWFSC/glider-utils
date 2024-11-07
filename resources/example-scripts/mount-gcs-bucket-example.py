# Sample Python code to mount GCS bucket
# Note: could also us os.system 
from subprocess import run

# Specify the bucket name, and local mount location
bucket_name = "bucket_name"
bucket_mtpt = f"/home/jupyter/{bucket_name}"

# Run unmount command, in case bucket is already mounted
run(["fusermount", "-u",  bucket_mtpt])
# Mount bucket with implicit directories (i.e., 'normal' implicit folder structure'
run(["gcsfuse", "--implicit-dirs", bucket_name,  bucket_mtpt])

# Mount bucket with implicit directories as read-only
run(["gcsfuse", "--implicit-dirs", "-o", "ro", bucket_name,  bucket_mtpt])
