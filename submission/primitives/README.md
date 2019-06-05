# Index of open source D3M primitives

This repository contains JSON-serialized metadata (annotation) of **open source** primitives
and their example pipelines. You can use repository to discover available primitives.

**D3M performers**: If your primitive is not (yet) open source, submit its
annotation to [private repository](https://gitlab.datadrivendiscovery.org/jpl/primitives_repo)
instead.

## Structure of repository

The directory and file structure is defined and controlled:

```
primitives/
  <interface_version>/
    <performer_team>/
      <python_path>/
        <version>/
          pipelines/
            <pipeline 1 id>.json
            <pipeline 1 id>.meta
            <pipeline 2 id>.yml
            ...
          primitive.json
  failed/
    ... structure as above ...
  archive/
    ... old primitive annotations ...
```

* `interface_version` is a version tag of a primitive interfaces package
  you used to generate the annotation and against which the primitive is
  developed, and should match the `primitive_code.interfaces_version` metadata entry
  with `v` prefix added.
* `performer_team` should match the `source.name` metadata entry.
* `python_path` should match the `python_path` metadata entry and should start
  with `d3m.primitives`.
* `version` should match `version` metadata entry.
* `primitive.json` is a JSON-serialized metadata of the primitive
  obtained by running `python3 -m d3m index describe -i 4 <python_path>`.
* All added primitive annotations are regularly re-validated. If they fail validation,
  they are moved under the `failed` directory.
* Pipeline examples in D3M pipeline description language must have a filename
  matching pipeline's ID with `.json` or `.yml` file extensions.
* A pipeline can have a `.meta` file with same base filename. Existence of
  this file makes the pipeline a standard pipeline (inputs are `Dataset` objects
  and output are predictions as `DataFrame`). Other pipelines might be
  referenced as subpipelines with arbitrary inputs and outputs.
* `.meta` file is a JSON file providing a problem ID to be used with the pipeline
  and input training and testing dataset IDs. Only
  [standard problems and datasets](https://gitlab.datadrivendiscovery.org/d3m/datasets)
  are allowed. In the case that a dataset does not have pre-split train/test/score
  splits just provide `problem` and `full_inputs`. Note, MIT-LL "score" splits
  have ID equal to the "test" split, change it to have the `SCORE` suffix.

    ```
    {
        "problem": "185_baseball_problem",
        "full_inputs": ["185_baseball_dataset"],
        "train_inputs": ["185_baseball_dataset_TRAIN"],
        "test_inputs": ["185_baseball_dataset_TEST"],
        "score_inputs": ["185_baseball_dataset_SCORE"]
    }
    ```
* For primitive references in your pipelines, consider not specifying `digest`
  field for primitives you do not control. This way your pipelnes will not
  fail with digest mismatch if those primitives get updated. (But they might
  fail because of behavior change of those primitives, but you cannot do much
  about that.)

## Adding a primitive

* You can add a new primitive or a new version of a primitive by
  creating a merge request against the master branch of this repository
  with `primitive.json` file for the primitive added according to the
  repository structure.
  * To make a merge request make a dedicated branch for that merge request in
    the fork of this repository and make a merge request for it.
  * Do not work or modify anything in the `master` branch of your fork because you will
    have issues if your merge request will not get merged for some reason.
  * Keep sure that your fork, your `master` branch, and a dedicated branch you
    are working from are all up-to-date with the `master` branch of this repository.
* Generate `primitive.json` file using `python3 -m d3m index describe -i 4 <python_path> > primitive.json`
  command. The command will validate generated file automatically and
  will not generate JSON output if there are any issues.
  * Make sure all install dependencies are at least accessible to all other
    performers, if not public, so that they can use them. **CI validation cannot check this**.
* Create any missing directories to adhere to the repository structure.
* Add pipeline examples for every primitive annotation you add.
  * You can use `add.py` script available in this repository to help you with these two steps.
* Do not delete any existing files or modify files which are not your annotations.
* Once a merge request is made, the CI will validate added files automatically.
* After CI validation succeeds (`validate` job), the maintainers of the repository
  will merge the merge request.
* You can submit same primitive and version to multiple primitive interface
  directories if your primitive works well with them.
* There is also CI validation against current development version of core packages
  (`validate_devel` job). Failing that will output a warning but not prevent adding
  a primitive. This job validates your primitive annotation against devel version of
  core packages. In this way you can validate also against the upcoming version of
  core packages and make your annotation ready so that it can be automatically ported
  to the new release once it is published.

## Local validation

You can run the CI validation script also locally so that you do not have to commit
and push and wait for CI to test your primitive annotation. Passing the CI validation script
locally is not authoritative but it can help speed up debugging.

CI validation script has some requirements:

* Linux
* Working Docker installation, logged in into `registry.datadrivendiscovery.org`
* PyPi packages listed in [`requirements.txt`](requirements.txt)
* Internet connection

Run it by providing the path to the primitive annotation file you want to validate. Example:

```bash
$ ./run_validation.py 'v2017.12.27/Test team/d3m.primitives.test.IncrementPrimitive/0.1.0/primitive.json'
```

To validate pipeline description do:

```bash
$ python3 -m d3m pipeline describe <path_to_JSON>
```

It will print out the pipeline JSON if it succeeds, or an error otherwise. You should probably run it inside
a Docker image with all primitives your pipeline references, or have them installed on your system.

You can validate your `.meta` file by running:

```bash
$ python3 -m d3m runtime -d /path/to/all/datasets fit-score -m your-pipeline.meta -p your-pipeline.yml
```

## Requesting a primitive

If you would like to request a primitive, [use private repository](https://gitlab.datadrivendiscovery.org/jpl/primitives_repo#requesting-a-primitive).

## Reporting issues with a primitive of a performer

To report all issues with a primitive of a performer, [use private repository](https://gitlab.datadrivendiscovery.org/jpl/primitives_repo#reporting-issues-with-a-primitive-of-a-performer).

## Note

Do not check in the source code here. Please host your source code in a different repository and
put the link or links in `source.uris` metadata entry.
