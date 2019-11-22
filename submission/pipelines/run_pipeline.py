import subprocess
import os
from typing import Tuple

from d3m import cli

from submission.utils import check_pipeline_run_was_successful, gzip_file

DATASETS_DIR = '/datasets/seed_datasets_current'
WORKER_ID = os.getenv('WORKER_ID')

def run_pipeline(
    pipeline_path: str,
    problem_name: str,
    output_dir: str,
    output_name: str,
    should_output_scores: bool = False
) -> Tuple[str, str]:
    """
    Run a pipeline on a problem using the d3m reference runtime.

    Parameters
    ----------
    pipeline_path
        The full path to the pipeline file to be run.
    problem_name
        The name of the problem to be run. Should be the name as it
        exists under the DATASETS environment variable.
    output_dir
        The path to the directory to save the pipeline run and scores to.
    output_name
        The name to give the pipeline run and scores. The name should not
        include the file extension. 
    should_output_scores
        Whether to compute and save the pipeline run scores as well. If
        `True`, the scores will be saved in `output_dir`.
    
    Returns
    -------
    str
        The full file path to the outputted pipeline run.
    str
        The full file path to the outputted run scores file, which will only
        be present if `should_output_scores=True`.
    """
    problem_dir = os.path.join(DATASETS_DIR, problem_name)
    pipeline_run_path = os.path.join(output_dir, f'{output_name}.yml')
    pipeline_run_scores_path = os.path.join(output_dir, f'{output_name}_scores.csv')

    score_subfolder_postfix = 'SCORE' if os.path.isdir(f'{problem_dir}/SCORE/dataset_SCORE') else 'TEST'
    score_input_path = os.path.join(problem_dir, 'SCORE', f'dataset_{score_subfolder_postfix}', 'datasetDoc.json')

    # Run the pipeline, generating the pipeline run and scores
    run_args = [
        '',
        '--strict-resolving',
        '--strict-digest',
        'runtime',
        '--worker-id', WORKER_ID,
        'fit-score',
        '--pipeline', pipeline_path,
        '--problem', os.path.join(problem_dir, f'{problem_name}_problem', 'problemDoc.json'),
        '--input', os.path.join(problem_dir, 'TRAIN', 'dataset_TRAIN', 'datasetDoc.json'),
        '--test-input', os.path.join(problem_dir, 'TEST', 'dataset_TEST', 'datasetDoc.json'),
        '--score-input', score_input_path,
        '--output-run', pipeline_run_path
    ]
    if should_output_scores:
        run_args += ['--scores', pipeline_run_scores_path]
    cli.main(run_args)

    return pipeline_run_path, pipeline_run_scores_path

def run_pipeline_run(
    pipeline_run_path: str,
    pipelines_dir: str,
    output_dir: str,
    output_name: str,
    should_output_scores: bool = False
) -> Tuple[str, str]:
    """
    Runs a pipeline run (i.e. executes a pipeline  'rerun') using the
    d3m reference runtime.

    Parameters
    ----------
    pipeline_run_path
        The full path to the pipeline run that is to be rerun.
    pipelines_dir
        The directory where the pipeline can be found that the pipeline
        run references.
    output_dir
        The path to the directory to save the pipeline rerun and scores to.
    output_name
        The name to give the pipeline rerun and scores. The name should not
        include the file extension. 
    should_output_scores
        Whether to compute and save the pipeline rerun scores as well. If
        `True`, the scores will be saved in `output_dir`.
    
    Returns
    -------
    str
        The full file path to the outputted pipeline rerun
    str
        The full file path to the outputted rerun scores file, which will only
        be present if `should_output_scores=True`.
    """
    pipeline_rerun_path = f'{output_dir}/{output_name}.yml'
    pipeline_rerun_scores_path = f'{output_dir}/{output_name}_scores.csv'

    # re-run the pipeline run to verify reproducibility
    rerun_args = [
        '',
        '--pipelines-path', pipelines_dir,
        'runtime',
        '--datasets', DATASETS_DIR,
        '--context', 'TESTING',
        '--worker-id', WORKER_ID,
        'fit-score',
        '--input-run', pipeline_run_path,
        '--output-run', pipeline_rerun_path,
    ]
    if should_output_scores:
        rerun_args += ['--scores', pipeline_rerun_scores_path]
    cli.main(rerun_args)

    return pipeline_rerun_path, pipeline_rerun_scores_path

def run_and_save_pipeline_for_submission(
    pipeline_path: str,
    problem_name: str,
    submission_path: str,
    run_output_name: str,
    should_output_scores: bool = False
) -> str:
    """
    Run a pipeline on a problem using the d3m reference runtime and
    save the pipeline run in gzip format to the proper submission_path.
    Also verifies that the run can be rerun successfully. 

    Parameters
    ----------
    pipeline_path
        The full path to the pipeline file to be run.
    problem_name
        The name of the problem to be run. Should be the name as it
        exists under the DATASETS environment variable.
    submission_path
        The directory where the primitive's pipelines and pipeline_runs go
    run_output_name
        The name to give the pipeline run. Should not include the run's file extension.
    should_output_scores
        Whether to compute and save the pipeline run scores as well. If
        `True`, the scores will be saved in `output_dir`.

    Returns
    -------
    str
        The full file path to the outputted pipeline run.
    """
    pipelines_dir = os.path.join(submission_path, 'pipelines')

    # Make folder if it doesn't exist already.
    pipeline_runs_dir = os.path.join(submission_path, 'pipeline_runs')
    if not os.path.exists(pipeline_runs_dir):
        os.makedirs(pipeline_runs_dir)

    # First, run the pipeline and create the pipeline run doc.
    pipeline_run_path, _ = run_pipeline(
        pipeline_path,
        problem_name,
        pipeline_runs_dir,
        run_output_name,
        should_output_scores
    )
    print(f'checking {pipeline_run_path} was successful')
    check_pipeline_run_was_successful(pipeline_run_path)

    # Next, rerun the pipeline run doc to verify reproducibility.
    rerun_output_name = f'{run_output_name}_rerun'
    pipeline_rerun_path, _ = run_pipeline_run(
        pipeline_run_path,
        pipelines_dir,
        pipeline_runs_dir,
        rerun_output_name,
        should_output_scores
    )
    print(f'checking {pipeline_rerun_path} was successful')
    check_pipeline_run_was_successful(pipeline_rerun_path)

    os.remove(pipeline_rerun_path)  # the rerun isn't part of the submission
    gzip_file(pipeline_run_path, remove_original=True)

    return pipeline_run_path
