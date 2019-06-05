# d3m-primitives

BYU-DML machine learning algorithms or primitives created for DARPA's D3M project.
These primitives are wrapped to fit within the D3M ecosystem.

## How to update
0. Update the forked d3m repo with the newest changes.
1. Update the primitive items: metalearn, the version number in `byudml/<primitive>/<primitive_code>.py`, the `__python_path__` in the same file (if applicable),  and the Dockerfile with the latest tag from d3m.
2. From the home directory, run this command: `bash run_tests.sh`.  This command will verify that nothing is broken, generate new pipeline and primitive jsons with updated digests and versions, and place them in the correct folder in the submodule of the `primitive` repo.
3. Go into the submodule directory `cd primitives/` and commit the updated json files `git commit -m "your custom message"`.
4. Push the changes to the forked gitlab repo (https://gitlab.com/byu-dml/primitives)
5. Update this repo by committing the changes to the submodule `git add primitives/` and `git commit` and `git push`.
6. Use the forked repo to update the main d3m `primitive` repo so that it is available for d3m.

## metafeature_extraction

Extracts metafeatures from tabular data.

## random_sampling_imputer

Imputes missing values in tabular data by randomly sampling other known values from the same column.
