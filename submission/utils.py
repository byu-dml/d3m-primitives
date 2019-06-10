import glob
import json
import os
import shutil


def get_new_d3m_path():
    """
    Gets the name of the newest version path for d3m/byudml
    :return:
    """
    new_directory = max(glob.glob('submission/primitives/v????.?.?'))
    byu_path = "byu-dml"
    byu_dir = os.path.join(new_directory, byu_path)
    return byu_dir


def create_meta_script_seed(problem):
    """
    Creates the meta file for a given problem.
    :param problem: the name of the problem for the dataset
    :return: the meta file
    """
    return \
        {
        "problem": "{0}_problem".format(problem),
        "full_inputs": [
            "{0}_dataset".format(problem)
        ],
        "train_inputs": [
            "{0}_dataset_TRAIN".format(problem)
        ],
        "test_inputs": [
            "{0}_dataset_TEST".format(problem)
        ],
        "score_inputs": [
            "{0}_dataset_SCORE".format(problem)
        ]
    }


def clear_directory(dir_path):
    """
    CAREFUL: this will DELETE ALL FILES in dirs_path

    This function clears the submodule directory so that we can add the new information
    :param dir_path: the directory where all files will be deleted
    """
    files = glob.glob(dir_path + '/*')
    for f in files:
        print("Deleting all files in", f)
        shutil.rmtree(f)


def create_and_add_pipelines_for_submission(primitive_dir, new_version_num, pipeline_json, problem_name):
    """
    Adds pipelines to the submodule directory and creates directories if it needs it
    :param primitive_dir: the python path of the primitive
    :param new_version_num: the latest version number of the primitive
    :param pipeline_json: the pipeline to be written to file
    :param problem_name: the name of the problem
    """
    # make folders if they don't exist already
    pipeline_dir = os.path.join(primitive_dir, new_version_num, "pipelines")
    if not os.path.exists(pipeline_dir):
        os.makedirs(pipeline_dir)

    # write json pipeline out
    pipeline_name = os.path.join(pipeline_dir, pipeline_json["id"]+ ".json")
    meta_name = os.path.join(pipeline_dir, pipeline_json["id"]+ ".meta")

    with open(pipeline_name, "w") as f:
        f.write(json.dumps(pipeline_json, indent=4))
        os.chmod(pipeline_name, 0o777)

    with open(meta_name, "w") as f:
        f.write(json.dumps(create_meta_script_seed(problem_name), indent=4))
        os.chmod(pipeline_name, 0o777)