import os

# This is where the datasets directory lives inside
# the docker container.
DATASETS_DIR = "/datasets/seed_datasets_current"
WORKER_ID = os.getenv('WORKER_ID')

PROBLEM_BLACKLIST = [
    # Large, takes a long time: ~150k rows.
    "SEMI_1217_click_prediction_small_MIN_METADATA"
]
