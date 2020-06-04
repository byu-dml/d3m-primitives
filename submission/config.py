import os

# This is where the datasets directory lives inside
# the docker container.
DATASETS_DIR = "/datasets/seed_datasets_current"
WORKER_ID = os.getenv('WORKER_ID')
