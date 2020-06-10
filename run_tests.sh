#! /bin/bash

pip3 install . > /dev/null
mv byudml tmp_byudml

python3 submission/primitive_jsons/generate_primitive_jsons.py
python3 submission/pipelines/generate_pipelines.py


# test static file download and use in profiler
mkdir /static

python3 -m d3m primitive download -p d3m.primitives.schema_discovery.profiler.BYU -o /static

predictions_path=./38_sick_predictions.csv
pipeline_run_path=./38_sick_pipeline_run.yml

python3 -m d3m runtime --volumes /static fit-score \
    --pipeline ./submission/pipelines/semantic_profiler/f4ebb9c9-ef15-491d-9a39-595c20f3e78e.json \
    --problem /datasets/seed_datasets_current/38_sick_MIN_METADATA/TRAIN/problem_TRAIN/problemDoc.json \
    --input /datasets/seed_datasets_current/38_sick_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json \
    --test-input /datasets/seed_datasets_current/38_sick_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json \
    --output $predictions_path \
    --output-run $pipeline_run_path \
    > /dev/null

chmod 777 $predictions_path
chmod 777 $pipeline_run_path


python3 run_tests.py
python3 submission/ci_validation.py

mv tmp_byudml byudml

pip3 uninstall -y byudml > /dev/null
