Some notes how to manage this repository and [Docker `images` repository](https://gitlab.com/datadrivendiscovery/images).

# Adding a new core package version

* In `images` repository:
  * Copy latest stable Dockerfile and name it after the new core package version.
    For example, `ubuntu-artful-python36-v2018.6.5.dockerfile` to
    `ubuntu-artful-python36-v2018.7.10.dockerfile`.
  * Update in new Dockerfile `FROM` to point to the new version of the `core` image.
  * Check the `devel` Dockerfile if there is anything to add from it to the
    new Dockerfile (any changes necessary to make the new release work).
  * Move old Dockerfiles to `archive`. Only the latest
    two versions + `devel` Dockerfile should be in `complete` directory.
    This will disable building Docker images for old versions and
    effectively freeze the images for those versions.
  * Make sure these files are moved before moving old primitive annotations
    in `primitives` repository. Otherwise images for old versions
    might be rebuild without any primitives but we want them preserved.
  * Do **not** remove Docker images for files just moved. We want to keep them
    available (but not updating anymore).
* In `primitives` repository:
  * Add a directory for new core package version, e.g., `v2018.4.18`.
    Add `.gitignore` placeholder file into the directory.
  * Update `FIXED_PACKAGE_VERSIONS` in `run_validation.py` script for new version.
  * If the base Docker image is changing to a new version of Ubuntu or Python
    (for example, not using anymore `artful-python36`), update the
    map between core package versions and image versions in `run_validation.py` script.
  * Add:
    * Primitive annotations for common primitives.
    * Primitive annotations for test primitives.
    * Primitive annotations for sklearn wrap primitives.
  * If it is just a bugfix release, try to migrate other existing primitive annotations
    (see below for instructions).
  * Move old primitive annotations directory into `archive`. Only the latest
    two versions should be in the repo: the old latest and the new latest just
    being added.
* Then follow steps for [private primitives and Docker images](https://gitlab.datadrivendiscovery.org/jpl/primitives_repo/blob/master/HOW_TO_MANAGE.md).

# Disabling a primitive annotation

Sometimes an accepted primitive annotations starts failing. For example,
a Docker image cannot be built anymore. This happens for various reasons,
a common one is that a primitive does not specify an upper bound on a
dependency and a new version of a dependency got released which breaks the
primitive. In this case we disable the primitive annotation and remove it
from the image by moving it to `failed` directory. There is a script to help
with this, `disable.py`. It can run in two ways:

* Providing a primitive ID from validation log, found listed in the final
  `ERROR` line during validation, example:

    ```
    ./disable.py v2018.1.26/d3m.primitives.dsbox.MultiTableFeaturization/0.1.3
    ```
* Providing a path to primitive annotation inside the repository, example:

    ```
    ./disable.py ./v2018.1.26/ISI/d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization/0.1.3/primitive.json
    ```

Do not forget to commit both the removed and added files (they have been moved).

# Migrating primitive annotations

When a new core package is released we can try to automatically port existing primitives
to the new version. We can do this by:

* Running `./migrate.sh <old version> <new version>` where `old version` and `new version` are
  directories which we want to migrate from and to, respectively.
  * By default the script just migrates the interface version string in annotations,
    so if there are more changes to annotations (like changes to docstrings) script should
    be temporary updated to do them as well.
* This will create and push branches with migrated primitive annotations for each team.
* [Visit active branches on GitLab](https://gitlab.com/datadrivendiscovery/primitives/branches/active)
  and for each "migration" branch click on *Merge request* and create it for the branch.
* Enable *Merge when pipeline succeeds* and select *Remove source branch* in the merge request.
* Some merge request CIs might fail because required primitives for their pipelines do not yet exist in the image.
  In that case wait for the new image to be rebuild with primitive annotations merged until then
  and retry failed merge request CIs.
