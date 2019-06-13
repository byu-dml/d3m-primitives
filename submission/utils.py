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


def create_meta_script_seed(problem, seed=True):
    """
    Creates the meta file for a given problem.
    :param problem: the name of the problem for the dataset
    :return: the meta file
    """
    if seed:
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
    else:
        return \
            {
                "problem": "{0}_problem".format(problem),
                "full_inputs": [
                    "{0}_dataset".format(problem)
                ],
            }


def clear_directory(dir_path):
    """
    CAREFUL: this will DELETE ALL FILES in dirs_path

    This function clears the submodule directory so that we can add the new information
    :param dir_path: the directory where all files will be deleted
    """
    files = glob.glob(dir_path + '/*')
    for f in files:
        shutil.rmtree(f)


def write_pipeline_for_submission(primitive_dir, new_version_num, pipeline_json, problem_name):
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

def get_pipeline_from_database(pipeline_id, mongo_client):
    """
    This function gets a pipeline from our local database given an id

    Parameters
    ----------
    pipeline_id: the id of the pipeline to grab
    mongo_client: a connection to the database

    :return a pipeline matching the id
    """
    collection = mongo_client.metalearning.pipelines
    pipeline_to_write = collection.find({"id": pipeline_id})
    for pipeline in pipeline_to_write:
        # should only be one pipeline
        return pipeline
    raise FileExistsError("Pipeline ID does not exist in the database")


seed_datasets_exlines = {
    "1491_one_hundred_plants_margin": {"score" : 0.862722, "mit-score": 0.693786, "problem": "accuracy"},
    "1491_one_hundred_plants_margin_clust": {"score" : 0.82062,"mit-score": 0.816731,"problem": "accuracy"},
    "1567_poker_hand": {"score" : 0.36039,"mit-score": 0.0555651,"problem": "accuracy"},
    "185_baseball": {"score" : 0.713804,"mit-score": 0.69137,"problem": "accuracy"},
    "196_autoMpg": {"score" : 5.66376,"mit-score": 7.37077	,"problem": "regression"},
    "22_handgeometry": {"score" : 0.203687,"mit-score": 0.145134,"problem": "regression"},
    "27_wordLevels": {"score" : 0.31072,"mit-score": 0.315046,"problem": "accuracy"},
    "26_radon_seed": {"score" : 0.0401657,"mit-score": 0.765384,"problem": "regression"},
    "299_libras_move": {"score" : 4.53215,"mit-score": 17.2537,"problem": "regression"},
    "30_personae": {"score" : 0.619048,"mit-score": 0.571429,"problem": "accuracy"},
    "313_spectrometer": {"score" : 0.496147,"mit-score": 0.19095, "problem": "accuracy"},
    "31_urbansound": {"score" : 0.929204,"mit-score": 0.483776,"problem": "accuracy"},
    "32_wikiqa": {"score" : 0.453901,"mit-score": 0.72524, "problem": "accuracy"},
    "38_sick": {"score" : 0.957307,"mit-score": 0.849966,"problem": "accuracy"},
    "4550_MiceProtein": {"score" : 1,"mit-score": 1,"problem": "accuracy"},
    "49_facebook": {"score" : 0.92798,"mit-score": 0.92798,"problem": "accuracy"},
    "534_cps_85_wages": {"score" : 19.9594,"mit-score": 19.7419,"problem": "regression"},
    "56_sunspots": {"score" : 30.9789,"mit-score": 30.9789,"problem": "regression"},
    "57_hypothyroid": {"score" : 0.977997,"mit-score": 0.842661,"problem": "accuracy"},
    "59_umls": {"score" : 0.942505,"mit-score": 0.803168,"problem": "accuracy"},
    "60_jester": {"score" : 3.09141,"mit-score": 3.71511,"problem": "regression"},
    "66_chlorineConcentration": {"score" : 0.971237,"mit-score": 0.610168,"problem": "accuracy"},
    "6_70_com_amazon": {"score" : 0.850326,"mit-score": 0.811539,"problem": "accuracy"},
    "6_86_com_DBLP": {"score" : 0.722373,"mit-score": 0.722373,"problem": "accuracy"},
    "LL0_1100_popularkids": {"score" : 0.464314,"mit-score": 0.41428, "problem": "accuracy"},
    "LL0_186_braziltourism": {"score" : 0.393803,"mit-score": 0.112059,"problem": "accuracy"},
    "LL0_207_autoPrice": {"score" : 4.33215e+06,"mit-score": 6.98572e+06,"problem": "regression"},
    "LL0_acled": {"score" : 0.93862	,"mit-score": 0.848077,"problem": "accuracy"},
    "LL0_acled_reduced": {"score" : 0.930426,"mit-score": 0.848077	,"problem": "accuracy"},
    "LL1_336_MS_Geolife_transport_mode_prediction": {"score" : 0.786346,"mit-score": 0.539788,"problem": "accuracy"},
    "LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon": {"score" : 0.560279,"mit-score": 0.560279,"problem": "accuracy"},
    "LL1_3476_HMDB_actio_recognition": {"score" : 0.111111,"mit-score": 0.951613,"problem": "accuracy"},
    "LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction": {"score" : 0.488889,"mit-score": 0.643939,"problem": "accuracy"},
    "LL1_736_stock_market": {"score" : 1.21545,"mit-score": 4.72722, "problem": "regression"},
    "LL1_crime_chicago": {"score" : 0.65, "mit-score": 0.693786,"problem": "accuracy"},
    "LL1_EDGELIST_net_nomination_seed": {"score" :0.3875,"mit-score": 0.85, "problem": "accuracy"},
    "LL1_net_nomination_seed": {"score" : 0.3875,"mit-score": 0.85,"problem": "accuracy"},
    "uu1_datasmash": {"score" : 1, "mit-score": 0.6875, "problem": "accuracy"},
    "uu2_gp_hyperparameter_estimation": {"score" : 0.354031,"mit-score": 0.824358,"problem": "regression"},
    "uu3_world_development_indicators": {"score" : 0.681363,"mit-score": 1.18279,"problem": "regression"},
    "uu4_SPECT": {"score" : 0.897059, "mit-score": 0.892086,"problem": "accuracy"},
    "uu5_heartstatlog": {"score" : 0.657143, "mit-score": 0.657143,"problem": "accuracy"},
    "uu6_hepatitis": {"score" : 0.421053,"mit-score": 0.421053, "problem": "accuracy"},
    "uu7_pima_diabetes": {"score" : 0.56, "mit-score": 0.56, "problem": "accuracy"},
}