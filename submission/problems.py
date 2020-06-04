import os
import json
import typing as t


class ProblemReference:

    subsets = {"TRAIN", "TEST", "SCORE"}

    def __init__(self, name: str, directory: str) -> None:
        """
        :param name: The name of the directory where this problem's files
            can be found.
        :param directory: The path to the directory the problem is found in.
        """
        # The name of the problem/dataset. Also the same as the root directory
        # the problem lives in.
        self.name = name
        # The path to the root directory of the problem and dataset.
        self.path = os.path.join(directory, self.name)
        self._load_and_set_attributes()

    @property
    def data_splits_path(self) -> str:
        return os.path.join(self.path, f"{self.name}_problem", "dataSplits.csv")

    @property
    def problem_doc_path(self) -> str:
        return os.path.join(self.path, f"{self.name}_problem", "problemDoc.json")

    @property
    def dataset_doc_path(self) -> str:
        return os.path.join(self.path, f"{self.name}_dataset", "datasetDoc.json")
    
    def get_subset_dataset_doc_path(self, subset: str) -> str:
        """
        Each D3M problem has official TRAIN/TEST/SCORE splits. This method
        will return the file path to any of those desired splits.
        """
        assert subset in self.subsets
        if subset == "SCORE":
            # SCORE is a special case because of some inconsistencies in the datasets
            # repo. Sometimes its subfolder is named SCORE and sometimes its TEST.
            score_subfolder_postfix = "SCORE" if os.path.isdir(os.path.join(self.path, "SCORE", "dataset_SCORE")) else "TEST"
            return os.path.join(self.path, 'SCORE', f'dataset_{score_subfolder_postfix}', 'datasetDoc.json')
        else:
            return os.path.join(self.path, subset, f'dataset_{subset}', 'datasetDoc.json')

    def get_dataset_id(self) -> str:
        return json.load(open(self.dataset_doc_path))["about"]["datasetID"]

    def _load_and_set_attributes(self):
        """
        Gathers various attributes about the problem and its dataset
        using its problem document and dataset document.
        """
        # Gather data about the problem.
        with open(self.problem_doc_path, "r") as f:
            problem_doc = json.load(f)

        self.task_keywords = problem_doc["about"]["taskKeywords"]
        if "classification" in self.task_keywords:
            self.problem_type = "classification"
        elif "regression" in self.task_keywords:
            self.problem_type = "regression"
        else:
            self.problem_type = None

        self.is_tabular = "tabular" in self.task_keywords
        self.problem_id = problem_doc["about"]["problemID"]

        # Gather data about the problem's dataset.
        with open(self.dataset_doc_path, "r") as f:
            dataset_doc = json.load(f)

        self.dataset_id = dataset_doc["about"]["datasetID"]
        self.dataset_digest = dataset_doc["about"]["digest"]


def get_tabular_problems(
    datasets_dir: str,
    problem_types: list = ["classification", "regression"]
) -> t.List[ProblemReference]:
    """
    Gets all tabular problems under `datasets_dir` having problem type
    in `problem_types`.

    :param datasets_dir: The path to a directory containing problems.
    :param problem_types: The problem types to get problems for.

    :return: a dictionary containing two keys: `classification`
        and `regression`.  Each key maps to a list of problems.
    """
    problems = []

    for dataset_name in os.listdir(datasets_dir):
        problem = ProblemReference(
            dataset_name, datasets_dir
        )
        if problem.is_tabular and problem.problem_type in problem_types:
            problems.append(problem)

    print(f"found {len(problems)} tabular problems.")

    return problems
