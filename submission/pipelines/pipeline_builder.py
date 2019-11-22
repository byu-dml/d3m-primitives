"""
TODO: THIS MODULE IS COPIED FROM THE D3M-EXPERIMENTER REPO.
EXTRAC THIS MODULE OUT OF THIS REPO AND THAT REPO INTO A SINGLE
REPO SO IT CAN BE USED AS A DEPENDENCY IN BOTH.
"""

from typing import Tuple, List, Any, Dict, Set, Optional
import itertools

from d3m import utils as d3m_utils
from d3m import index as d3m_index
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.pipeline import PrimitiveStep, StepBase
from d3m.metadata.base import ArgumentType, PrimitiveFamily
from d3m.primitive_interfaces.base import PrimitiveBase
from dsbox.datapreprocessing.cleaner.encoder import Encoder

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer

# TODO: This will no longer be needed once the new version of our primitives
# is in the D3M Docker container.
d3m_index.register_primitive(
    RandomSamplingImputer.metadata.query()['python_path'],
    RandomSamplingImputer
)

# TODO: Once the newest dsbox gets added to the new docker container and we
# update docker containers, we can remove this install.
d3m_index.register_primitive(
    Encoder.metadata.query()['python_path'],
    Encoder
)

# This is the full list of families:
# https://metadata.datadrivendiscovery.org/schemas/v0/definitions.json#/definitions/primitive_family
PRIMITIVE_FAMILIES = {
    "classification": [
        PrimitiveFamily.CLASSIFICATION,
        PrimitiveFamily.SEMISUPERVISED_CLASSIFICATION,
        PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
        PrimitiveFamily.VERTEX_CLASSIFICATION
    ]
}

class PipelineArchDesc:
    """
    Holds data that describes a pipeline's architecture. e.g.
    `PipelineArchDesc(generation_method="ensemble", generation_parameters={"k": 3}) could
    describe an ensemble pipeline that ensembles three classifiers.
    """

    def __init__(self, generation_method: str, generation_parameters: Dict[str, Any] = None):
        """
        :param generation_method: The pipeline's high level structural type e.g.
            "ensemble", "straight", "random", etc.
        :param generation_parameters: An optional dictionary holding data describing
            attributes of the pipeline's architecture e.g.
            { "depth": 4, "max_width": 3 }, etc. Fields in the dictionary
            will likely vary depending on the pipeline's type.
        """
        self.generation_method = generation_method
        self.generation_parameters = dict() if generation_parameters is None else generation_parameters
    
    def to_json_structure(self):
        return {
            "generation_method": self.generation_method,
            "generation_parameters": self.generation_parameters
        }


class EZPipeline(Pipeline):
    """
    A subclass of `d3m.metadata.pipeline.Pipeline` that is arguably easier
    to work with when building a pipeline.
    
    This pipeline class allows for certain steps to be
    tagged with a 'ref', (a.k.a. reference) so they can be easily referenced
    later on in the building of the pipeline. For example it can be used to
    track which steps contain the pipeline's raw dataframe, prepared
    attributes, prepared targets, etc., and to easily get access to those
    steps and their output when future steps need to use the output of those
    steps as input.

    This class also has helper methods for adding primitives to the pipeline
    in a way that's less verbose than the standard d3m way while still
    supporting many (but perhaps not all) common pipeline building use cases.
    It also provides ways to easily concatenate the output of different
    pipeline steps, and keeps track of which concatenations have happened before
    so it can reuse concatenations and optimize. 
    """

    # Constructor

    def __init__(self,
        *args,
        arch_desc: PipelineArchDesc = None,
        add_preparation_steps: bool = False,
        **kwargs
    ):
        """
        :param *args: All positional args are forwarded to the superclass
            constructor.
        :param arch_desc: an optional description of the pipeline's
            architecture.
        :param add_preparation_steps: Whether to add initial data preprocessing
            steps to the pipeline.
        :param **kwargs: All other keyword args are forwarded to the
            superclass constructor.
        """
        super().__init__(*args, **kwargs)

        self._step_i_of_refs: Dict[str, int] = {}
        self.arch_desc = arch_desc
        # A mapping of data reference pairs to the data reference of their
        # concatenation.
        self.concat_cache: Dict[frozenset, str] = {}

        if add_preparation_steps:
            self.add_preparation_steps()
    
    # Public properties

    @property
    def curr_step_i(self) -> None:
        """
        Returns the current step index i.e. the index of the step
        most recently added to the pipeline.
        """
        num_steps = len(self.steps)
        return None if num_steps == 0 else num_steps - 1
    
    @property
    def current_step(self) -> StepBase:
        """
        Allows the most recently added pipeline step to be accessed and
        modified at will e.g. `pipeline.current_step.add_hyperparameter(...)`
        etc.
        """
        return self.steps[self.curr_step_i]
    
    @property
    def curr_step_data_ref(self) -> str:
        return self._data_ref_by_step_i(self.curr_step_i)
    
    # Public methods

    def set_step_i_of(self, ref_name: str, step_i: int = None) -> None:
        """
        Sets the step index of the ref identified by `ref_name`.
        If `step_i` is `None`, the ref's step index will be set to the
        index of the current step.
        """
        if step_i is None:
            self._step_i_of_refs[ref_name] = self.curr_step_i
        else:
            self._step_i_of_refs[ref_name] = step_i
            
    def step_i_of(self, ref_name: str) -> int:
        """
        Returns the index of the step associated with `ref_name`.
        """
        self._check_ref_is_set(ref_name)
        return self._step_i_of_refs[ref_name]
    

    def data_ref_of(self, ref_name: str) -> str:
        """
        Returns a data reference to the output of the step associated
        with `ref_name`. For example if the step index of the `raw_attrs`
        ref is 2, and the output method name of step 2 is 'produce',
        then `data_ref_of('raw_attrs')` == 'step.2.produce'`.
        """
        return self._data_ref_by_step_i(self.step_i_of(ref_name))
    
    def to_json_structure(self, *args, **kwargs) -> Dict:
        """
        An overriden version of the parent class `Pipeline`'s method.
        Adds the pipeline architecture description to the json.
        """
        pipeline_json = super().to_json_structure(*args, **kwargs)
        if self.arch_desc is not None:
            pipeline_json['pipeline_generation_description'] = self.arch_desc.to_json_structure()
            # Update the digest since new information has been added to the description
            pipeline_json['digest'] = d3m_utils.compute_digest(pipeline_json)
        
        return pipeline_json
    
    def add_primitive_step(
        self,
        python_path: str,
        auto_data_ref: str = None,
        *,
        container_args: Dict[str,str] = None,
        value_args: Dict[str,str] = None,
        container_hyperparams: Dict[str,str] = None,
        value_hyperparams: Dict[str,str] = None,
        auto_fill_args: bool = True,
        output_name: str = 'produce',
        is_final_model: bool = True
    ) -> None:
        """
        Adds the results of `self.create_primitive_step` to the pipeline.
        All args passed on to `self.create_primitive_step` except these:

        :param is_final_model: If set, label encoding and semantic type
            changes will not automatically be applied to the output of
            this primitive if its a classifier.
        """
        step = self.create_primitive_step(
            python_path,
            auto_data_ref,
            container_args=container_args,
            value_args=value_args,
            container_hyperparams=container_hyperparams,
            value_hyperparams=value_hyperparams,
            auto_fill_args=auto_fill_args,
            output_name=output_name
        )
        self.add_step(step)

        if (
            not is_final_model
            and step.primitive.metadata.query()['primitive_family']
            in PRIMITIVE_FAMILIES['classification']
        ):
            # Since this primitive is a classifier and its not the final
            # model of the pipeline, its output will likely be used as a
            # feature for some future primitive, so we need to change the
            # output's semantic type to attribute and encode it.
            self.add_primitive_step(
                'd3m.primitives.data_transformation.replace_semantic_types.Common',
                value_hyperparams={
                    'from_semantic_types': ['https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                    'to_semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']
                }
            )

            # Next the encoder step
            self.add_primitive_step(
                "d3m.primitives.data_cleaning.label_encoder.DSBOX"
            )
    
    def create_primitive_step(
        self,
        python_path: str,
        auto_data_ref: str = None,
        *,
        container_args: Dict[str,str] = None,
        value_args: Dict[str,str] = None,
        container_hyperparams: Dict[str,str] = None,
        value_hyperparams: Dict[str,str] = None,
        auto_fill_args: bool = True,
        output_name: str = 'produce'
    ) -> PrimitiveStep:
        """
        Helper for creating a basic `d3m.metadata.pipeline.PrimitiveStep`
        in a less verbose manner. Can also do some additional things for you, like
        automatically populate the primitive's required arguments.

        :param python_path: the python path of the primitive to be added.
        :param auto_data_ref: If `auto_fill_args` is set, this data ref will be
            supplied to all unset required arguments that are not `outputs` (since `outputs`
            is always the pipeline's target column(s)). If not supplied, the data reference
            of the current step will be used instead.
        :param container_args: A map of primitive argument names to the data references
            to use for those arguments. Pass a primitive argument to `container_args` if the
            argument is of type `d3m.metadata.base.ArgumentType.CONTAINER`.
        :param value_args: A map of primitive argument names to the values to use for
            those arguments. Pass a primitive argument to `value_args` if the
            argument is of type `d3m.metadata.base.ArgumentType.VALUE`.
        :param container_hyperparams: A map of primitive hyperparameter names to the data references
            to use for those hyperparams. Pass a primitive hyperparam to `container_hyperparams` if the
            hyperparam is of type `d3m.metadata.base.ArgumentType.CONTAINER`.
        :param value_hyperparams: A map of primitive hyperparameter names to the values to use for
            those hyperparams. Pass a primitive hyperparam to `value_hyperparams` if the
            hyperparam is of type `d3m.metadata.base.ArgumentType.VALUE`.
        :param auto_fill_args: Attempt to automatically populate any required
            args that haven't been explicity supplied by the user.
        :param output_name: an optional method output name to use for the primitive.
        """
        primitive = d3m_index.get_primitive(python_path)
        step = PrimitiveStep(primitive=primitive)

        # Add the primitive's arguments.
        ################################

        primitive_args: Dict[str,Dict[str,Any]] = {}

        if container_args:
            primitive_args.update(
                self._build_d3m_arg_dict(container_args, ArgumentType.CONTAINER)
            )
        
        if value_args:
            primitive_args.update(
                self._build_d3m_arg_dict(value_args, ArgumentType.VALUE)
            )

        if auto_fill_args:
            for arg_name in self._get_required_args(primitive):
                if arg_name not in primitive_args:
                    # This arg hasn't been supplied yet.
                    if arg_name == "outputs":
                        data_ref = self.data_ref_of('target')
                    else:
                        data_ref = auto_data_ref if auto_data_ref else self.curr_step_data_ref

                    primitive_args[arg_name] = {
                        "name": arg_name,
                        "argument_type": ArgumentType.CONTAINER,
                        "data_reference": data_ref
                    }

        # Finally, add the arguments to the primitive
        for arg_dict in primitive_args.values():
            step.add_argument(**arg_dict)
        
        # Add hyperparameters to the primitive.
        #######################################

        primitive_hyperparams: Dict[str,Dict[str,Any]] = {}
        
        if container_hyperparams:
            primitive_hyperparams.update(
                self._build_d3m_arg_dict(container_hyperparams, ArgumentType.CONTAINER)
            )
        
        if value_hyperparams:
            primitive_hyperparams.update(
                self._build_d3m_arg_dict(value_hyperparams, ArgumentType.VALUE)
            )
            
        # Finally, add the hyperparameters to the primitive
        for hyperparam_dict in primitive_hyperparams.values():
            step.add_hyperparameter(**hyperparam_dict)

        step.add_output(output_name)
        return step
    
    def replace_step_at_i(
        self,
        step_i: int,
        new_step_python_path: str
    ) -> None:
        """
        Replaces the step at `step_i` with a new step. The inputs from
        the old step will be used as the inputs of the new step and all
        other required arguments will be populated automatically.
        """
        old_step_inputs_data_ref = self.steps[step_i].arguments['inputs']['data']
        new_step = self.create_primitive_step(
            new_step_python_path,
            old_step_inputs_data_ref
        )
        self.replace_step(step_i, new_step)
    
    def add_preparation_steps(self) -> None:
        """
        Adds a general data preparation pipeline preamble of primitives.
        """
        # Creating pipeline
        self.add_input(name='inputs')

        # dataset_to_dataframe step
        self.add_primitive_step(
            'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
            'inputs.0'
        )
        self.set_step_i_of('raw_df')

        # column_parser step
        self.add_primitive_step(
            'd3m.primitives.data_transformation.column_parser.Common',
            self.data_ref_of('raw_df'),
            value_hyperparams={
                # We don't parse categorical data, since we want it to stay as a string
                # and not be hashed as a long integer. That makes the one hot encoded
                # names more interpretable.
                "parse_semantic_types": (
                    'http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
                )
            }
        )
        self.set_step_i_of('parsed')

        # extract_columns_by_semantic_types(targets) step
        self.add_primitive_step(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
            value_hyperparams={ 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TrueTarget'] }
        )
        self.set_step_i_of('target')

        # extract_columns_by_semantic_types(attributes) step
        self.add_primitive_step(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
            self.data_ref_of('parsed'),
            value_hyperparams={ 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'] }
        )

        # imputer step
        self.add_primitive_step('d3m.primitives.data_preprocessing.random_sampling_imputer.BYU')

        # encoder step
        self.add_primitive_step('d3m.primitives.data_preprocessing.encoder.DSBOX')
        self.set_step_i_of('attrs')
    
    def add_predictions_constructor(self, input_data_ref: str = None) -> None:
        """
        Adds the predictions constructor to the pipeline
        :param input_data_ref: the data reference to be used as the input to the predictions primitive.
            If `None`, the output data reference to the most recently added step will be used. 
        """

        self.add_primitive_step(
            "d3m.primitives.data_transformation.construct_predictions.Common",
            input_data_ref,
            container_args={ "reference": self.data_ref_of('raw_df') }
        )
    
    def concatenate_inputs(self, *data_refs_to_concat) -> str:
        """
        Adds concatenation steps to the pipeline that join the outputs of every data
        reference found in `data_refs_to_concat` until they are all a single data frame.
        If two steps in `data_refs_to_concat` have already been concatenated in another step on
        the pipeline, then that concatenation step will be referenced during this
        concatenation, instead of creating a new duplicate concatenation step. Reduces
        the runtime and memory footprint of the pipeline.

        :param data_refs_to_concat: The data references to the steps whose outputs are to be
            concatentated together. Data references are strings like `steps.4.produce`.
        """
        if len(data_refs_to_concat) == 1:
            # No concatenation necessary
            output_data_ref, = data_refs_to_concat
            return output_data_ref
        
        data_refs = set(data_refs_to_concat)

        while len(data_refs) > 1:
            # first look in the pipeline cache for an already existing
            # concatenation we can use.
            cache_match = self._find_match_in_cache(data_refs)
            
            if cache_match is None:
                # Manually concatenate a pair of data refs, then add them to the cache.
                data_ref_pair = sorted(data_refs)[:2]
                self.add_primitive_step(
                    'd3m.primitives.data_transformation.horizontal_concat.DataFrameCommon',
                    container_args={ "left": data_ref_pair[0], "right": data_ref_pair[1] }
                )
                concat_data_ref = self.curr_step_data_ref

                # Save the link between the data ref pair and their concatenation's
                # output to the pipeline cache.
                data_ref_pair = frozenset(data_ref_pair)
                self.concat_cache[data_ref_pair] = concat_data_ref
                # Now we have a match in the cache.
                cache_match = data_ref_pair

            # Now that we've concatenated `data_ref_pair`, we don't
            # need it in `data_refs` anymore.
            data_refs -= cache_match
            data_refs.add(self.concat_cache[cache_match])
        
        output_data_ref, = data_refs
        return output_data_ref
    
    # Private methods
    
    def _check_ref_is_set(self, ref_name: str) -> None:
        if ref_name not in self._step_i_of_refs:
            raise ValueError(f'{ref_name} has not been set yet')
    
    def _data_ref_by_step_i(self, step_i: int) -> str:
        step_output_names: List[str] = self.steps[step_i].outputs
        if len(step_output_names) != 1:
            raise AttributeError(
                f'step {step_i} has more than one output; which output to use is ambiguous'
            )
        return f'steps.{step_i}.{step_output_names[0]}'
    
    def _get_required_args(self, p: PrimitiveBase) -> List[str]:
        """
        Gets the required arguments for a primitive
        :param p: the primitive to get arguments for
        :return a list of the required args
        """
        required_args = []
        metadata_args = p.metadata.to_json_structure()['primitive_code']['arguments']
        for arg, arg_info in metadata_args.items():
            if 'default' not in arg_info and arg_info['kind'] == 'PIPELINE':  # If yes, it is a required argument
                required_args.append(arg)
        return required_args
    
    def _find_match_in_cache(self, data_refs: Set[str]) -> Optional[frozenset]:
        """
        Uses all unordered combinations from `data_refs` (n choose k where k goes
        from n to 2) to find a match in the pipeline cache, that is to say, a set
        of data refs in `data_refs` who have already been concatenated.

        :param data_refs: The set of data_refs to search in the cache for.
        :return: If a match is found, the matching data ref pair is returned, else
            None is returned. 
        """
        for k in range(len(data_refs), 1, -1):
            # Use all unordered combinations from `data_refs`
            # (n choose k where k goes from n to 2)
            for subset in itertools.combinations(data_refs, k):
                subset = frozenset(subset)
                if subset in self.concat_cache:
                    return subset
        return None
    
    def _build_d3m_arg_dict(
        self,
        simple_args: Dict[str,str],
        arg_type: ArgumentType
    ) -> Dict[str,Dict[str,str]]:
        """
        Builds a dictionary of arguments. Each argument can be passed
        to the `.add_hyperparameter` or `.add_argument` method of a
        d3m pipeline step. Builds it from the simpler arguments that
        are passed to the `create_primitive_step` and `add_primitive_step`
        methods of this class.

        :param simple_args: A dictionary mapping argument names to their
            values. E.g. { 'negate': True }
        """
        data_arg_name_by_type = {
            ArgumentType.VALUE: "data",
            ArgumentType.CONTAINER: "data_reference"
        }

        if arg_type not in data_arg_name_by_type:
            raise ValueError(f'unsupported arg_type {arg_type}')

        data_arg_name = data_arg_name_by_type[arg_type]
        return {
            arg_name: { 
                "name": arg_name,
                "argument_type": arg_type,
                data_arg_name: data
            } for arg_name, data in simple_args.items()
        }

