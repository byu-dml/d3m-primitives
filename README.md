# d3m-primitives

BYU-DML machine learning algorithms or primitives created for DARPA's D3M project.
These primitives are wrapped to fit within the D3M ecosystem.

## metafeature_extraction

Extracts metafeatures from tabular data.

## random_sampling_imputer

Imputes missing values in tabular data by randomly sampling other known values from the same column.

## How to update

### When Cloning This Repo

1. Clone this repository. `cd` into it. Clone the primitives git submodule `git submodule update --init --recursive`.
2. `cd` into the `submission/primitives` directory, then verify it is synced to the current state of the D3M primitives repo source, and not just to the BYU's fork. See [configuring a remote fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork) and [syncing a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork). 
3. Create a `.env` file in the repo root. Populate the `DATASETS` and `WORKER_ID` environment variables. `DATASETS` should be a path to the root folder of the D3M datasets repo. `WORKER_ID` should be the ID uniquely identifying the machine any pipelines are run on.

### If The Repo Is Already Cloned

1. Update the primitives submodule (skip if the submodule was just cloned) `git submodule update --recursive`.
2. Update the primitives submodule
    - `cd submission/primitives`
    - Pull the master branch of the parent repository into the byu-dml branch `git pull https://gitlab.com/datadrivendiscovery/primitives`

### Remaining Steps

1. Update `Dockerfile` with the latest tag from D3M. Pull the image and start the container: `docker-compose up -d --build`. Note that the image will change, but the tag will not, as primitive authors submit their primitives. When this happens, one solution is to delete the image with `docker rmi <image id>` and pull it again.
2. Update the primitives, if necessary. At the least you'll likely need to update the dependencies in this repo to honor the dependencies and their version ranges found in the D3M core package. Be sure to update the version numbers in `byudml/__init__.py`.
3. Next, to run the tests, generate the primitive json files, generate the pipelines, and run the pipelines:
    * Execute `docker exec -it test-d3m-primitives bash` to enter the docker container.
    * `./run_tests.sh` This command will verify that nothing is broken, generate new pipeline and primitive jsons with updated digests and versions, run the pipelines, and place them in the correct folder in the submodule of the `primitives` repo. NOTE: Verify that the glob pattern in `submission.utils.get_new_d3m_path` will correctly capture the D3M version in the `primitives` submodule.
    * `exit`
4. Commit the updated primitive jsons and pipelines in the submodule i.e. our fork of the D3M primitives repo. **Note**: Do not commit straight to the master branch, but to a branch that semantically represents the new D3M package version and our organization.
5. Update **this** repo by committing the changes to the submodule `git add submission/primitives/` and `git add`, `git commit`, and `git push`.
6. Release this package.
7. Push the primitives submodule `git push origin byu-dml` (push to https://gitlab.com/byu-dml/primitives) and verify that the CI passes.  If this fails, start over at step 4. NOTE: this package must be released before it can be tested with CI.
8. Create a merge request from the byu-dml branch of https://gitlab.com/byu-dml/primitives to the master branch of https://gitlab.com/datadrivendiscovery/primitives.
