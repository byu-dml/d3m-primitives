# d3m-primitives

BYU-DML machine learning algorithms or primitives created for DARPA's D3M project.
These primitives are wrapped to fit within the D3M ecosystem.

## metafeature_extraction

Extracts metafeatures from tabular data.

## random_sampling_imputer

Imputes missing values in tabular data by randomly sampling other known values from the same column.

## How to update
0. Clone this repository. `cd` into it.
1. Clone the primitives git submodule `git submodule update --init --recursive`.
2. Update the primitives submodule
    a. `cd submission/primtives`
    b. Pull the master branch of the parent repository into the byu-dml branch `git pull https://gitlab.com/datadrivendiscovery/primitives`
3. Update `Dockerfile` with the latest tag from D3M. Pull the image and start the container: `docker-compose up -d --build`. Note that the image will change, but the tag will not, as primitive authors submit their primitives. When this happens, one solution is to delete the image with `docker rmi test-d3m-primtives` and pull it again.
4. Update the primtives, if necessary. Be sure to update the version numbers in `byudml/<primitive>/<primitive_code>.py` and in `byudml/__init__.py`.
5. Run the tests, generate the primitive json files, and generate the pipelines.
    * `docker exec -it test-d3m-primtives bash`
    * `./run_tests.sh` This command will verify that nothing is broken, generate new pipeline and primitive jsons with updated digests and versions, and place them in the correct folder in the submodule of the `primitive` repo.
    * `exit`
6. Commit the update primitive jsons and pipelines.
7. Update **this** repo by committing the changes to the submodule `git add submission/primitives/` and `git add`, `git commit`, and `git push`.
8. Release this package.
9. Push the primtives submodule `git push origin byu-dml` (push to https://gitlab.com/byu-dml/primitives) and verify that the CI passes.
TODO: fix this problem: currently this package must be released before it can be tested with CI.
10. Create a merge request from the byu-dml branch of https://gitlab.com/byu-dml/primitives to the master branch of https://gitlab.com/datadrivendiscovery/primitives.
