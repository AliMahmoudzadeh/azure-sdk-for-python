import asyncio
import os
import logging
from typing import Dict, Union, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import reformat_conversation_history, reformat_tool_definitions, \
    reformat_agent_response
from azure.ai.evaluation._common._experimental import experimental

# Import evaluators
from azure.ai.evaluation._evaluators._intent_resolution import IntentResolutionEvaluator
from azure.ai.evaluation._evaluators._tool_call_accuracy import ToolCallAccuracyEvaluator
from azure.ai.evaluation._evaluators._task_adherence import TaskAdherenceEvaluator

from data_analyzer import DataAnalyzer, visualize_data_analyzer_2d

logger = logging.getLogger(__name__)

# Todo: This needs to be put at the right place if sdk version is less than *.b10. We can ship this to available only with the latest version of the SDK.
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    from azure.ai.evaluation._converters._ai_services import AIAgentConverter
except ImportError as import_error:
    raise EvaluationException(
        message=f"Required Azure AI packages not available: {str(import_error)}. Please install azure-ai-projects and azure-identity packages.",
        internal_message=f"Missing required packages for agent evaluation: {str(import_error)}",
        # blame=ErrorBlame.USER_ERROR,
        # category=ErrorCategory.MISSING_FIELD,
        # target=ErrorTarget.CRITIC_AGENT,
    )


@experimental
class CriticAgent(PromptyEvaluatorBase[Dict[str, Union[str, List[str]]]]):
    """The CriticAgent evaluates which evaluator(s) to use based on the conversation history and tool definitions.
    1) It loads _critic_agent.prompty from the current directory, and uses it to select the appropriate evaluators.
    2) The constructor initializes the agent with the model configuration.
    3) The evaluate method supports multiple signatures:
       - evaluation_results = critic_agent.evaluate(thread_id, evaluation=Optional[None])
       - evaluation_results = critic_agent.evaluate(agent_id, azure_ai_project=Optional[str], evaluation=Optional[IntentResolution, , TaskAdherence])

    The data fetching from agent_id automates the code pattern from Azure AI Foundry agent evaluation samples.
    """

    _PROMPTY_FILE = "_critic_agent.prompty"
    _RESULT_KEY = "evaluator_selector"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    # Default agent evaluation metrics
    _DEFAULT_AGENT_EVALUATORS = ["IntentResolution", "ToolCallAccuracy", "TaskAdherence"]

    id = None
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(self, model_config, **kwargs):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        super().__init__(model_config=model_config, prompty_file=prompty_path, result_key=self._RESULT_KEY, **kwargs)
        self.model_config = model_config
        self.evaluator_instances = self._initialize_evaluators(self._DEFAULT_AGENT_EVALUATORS)

    def evaluate(self,
                 agent_id: str = None,
                 thread_id: str = None,
                 data: Optional[Dict[str, Any]] = None,
                 azure_ai_project: Optional[Dict[str, str]] = None,
                 evaluators: Optional[Union[str, List[str]]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Evaluate an agent using the specified evaluators.

        :param agent_id: The ID of the agent to evaluate
        :type agent_id: str
        :param thread_id: The ID of the conversation thread to evaluate
        :type thread_id: str
        :param data: Data for evaluation (not used in this implementation)
        :type data: Optional[Dict[str, Any]]
        :param azure_ai_project: Azure AI project configuration
        :type azure_ai_project: Optional[Dict[str, str]]
        :param evaluators: Specific evaluators to run
        :type evaluators: Optional[Union[str, List[str]]]
        :param kwargs: Additional keyword arguments
        :return: Evaluation results
        :rtype: Dict[str, Any]

        """
        # Todo: This needs to be fixed
        if evaluators is None:
            logger.warning("Evaluators not specified, using default evaluators: %s", self._DEFAULT_AGENT_EVALUATORS)
            evaluators = self._DEFAULT_AGENT_EVALUATORS
        if agent_id is None and thread_id is None and data is None:
            raise EvaluationException(
                message="Either agent_id or thread_id must be provided for evaluation.",
                internal_message="Missing agent_id or thread_id in input.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.CRITIC_AGENT,
            )
        if agent_id is not None:
            return self._evaluate_agent(
                agent_id=agent_id, azure_ai_project=azure_ai_project, evaluators=evaluators, **kwargs
            )
        if thread_id is not None:
            # Agent-based evaluation with data fetching
            return self._evaluate_conversation(thread_id, azure_ai_project, evaluators, **kwargs)
        elif data is not None:
            # Validate the data and run evaluation (to be implemented)
            pass

    def auto_evaluate(self, agent_id: str = None,
                      thread_id: str = None,
                      data: Optional[Dict[str, Any]] = None,
                      azure_ai_project: Optional[Dict[str, str]] = None,
                      **kwargs) -> List[Any]:
        """
        Auto-evaluate an agent or conversation thread.
        """
        if agent_id is None and thread_id is None and data is None:
            raise EvaluationException(
                message="Either agent_id or thread_id must be provided for evaluation.",
                internal_message="Missing agent_id or thread_id in input.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.CRITIC_AGENT,
            )
        if agent_id is not None:
            return self._evaluate_agent(
                agent_id=agent_id, azure_ai_project=azure_ai_project, evaluators=None, **kwargs
            )
        if thread_id is not None:
            # Agent-based evaluation with data fetching
            return self._evaluate_conversation(thread_id, azure_ai_project, **kwargs)
        elif data is not None:
            # Validate the data and run evaluation (to be implemented)
            pass

    def _evaluate_agent(self,
                        agent_id: str,
                        azure_ai_project: Dict[str, str],
                        evaluators: Optional[Union[str, List[str]]] = None,
                        **kwargs) -> List[Any]:
        """
        Evaluate an agent by fetching data from Azure AI Project and running specified evaluators.

        :param agent_id: The ID of the agent to evaluate
        :type agent_id: str
        :param azure_ai_project: Azure AI project configuration
        :type azure_ai_project: Dict[str, str]
        :param evaluators: Specific evaluators to run
        :type evaluators: Optional[Union[str, List[str]]]
        :return: Evaluation results
        :rtype: Dict[str, Any]
        """
        try:
            # Max number of threads to fetch. Default=5. This need not be a parameter
            max_threads = kwargs.get("max_threads", 5)

            project_client = AIProjectClient(
                endpoint=azure_ai_project.get("azure_endpoint"),
                credential=DefaultAzureCredential(),
            )

            thread_ids = self._fetch_agent_threads(project_client, agent_id)
            results = []
            for thread_id in thread_ids:
                if max_threads == 0:
                    break
                evaluated_result = self._evaluate_conversation(
                    thread_id=thread_id,
                    azure_ai_project=azure_ai_project,
                    evaluators_to_run=evaluators,
                    agent_id=agent_id,
                    **kwargs
                )
                if evaluated_result is not None:
                    max_threads -= 1
                    results.append(evaluated_result)
                    logger.info(f"Thread {max_threads} Evaluated thread {thread_id} for agent {agent_id}.")
            return results

        except Exception as e:
            logger.error(f"Error during agent evaluation: {str(e)}")
            raise EvaluationException(
                message=f"Failed to evaluate agent {agent_id}: {str(e)}",
                internal_message=f"Agent evaluation failed: {str(e)}",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                # target=ErrorTarget.CRITIC_AGENT,
            )

    def _evaluate_conversation(self,
                               thread_id: str,
                               azure_ai_project: Optional[Dict[str, str]] = None,
                               evaluators_to_run: Optional[Union[str, List[str]]] = None,
                               agent_id: Optional[str] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Evaluate a specific conversation thread.

        :param thread_id: The ID of the conversation thread
        :type thread_id: str
        :param evaluators_to_run: Specific evaluators to run
        :type evaluators_to_run: Optional[Union[str, List[str]]]
        :return: Evaluation results
        :rtype: Dict[str, Any]
        """
        # This would integrate with the existing async _do_eval method
        # For conversation-based evaluation using the prompty
        # print("THREAD ID", thread_id)
        if not thread_id:
            raise EvaluationException(
                message="Thread ID must be provided for conversation evaluation.",
                internal_message="Missing thread ID in input.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.CRITIC_AGENT,
            )
        project_client = AIProjectClient(
            endpoint=azure_ai_project.get("azure_endpoint"),
            credential=DefaultAzureCredential(),
        )

        converter = AIAgentConverter(project_client)
        try:
            conversation = converter.prepare_evaluation_data(thread_ids=thread_id)[-1]
        except:
            conversation = {}
        # {'query': [
        #     {'createdAt': '2025-07-17T08:56:22Z', 'role': 'user', 'content': [{'type': 'text', 'text': "hey there'"}]},
        #     {'createdAt': '2025-07-17T08:56:23Z', 'run_id': 'run_MVHZIe0TNWWKPx0ppvUz3uAh',
        #      'assistant_id': 'asst_CLx2RNAXhAoFIbkLxZfoM6P4', 'role': 'assistant',
        #      'content': [{'type': 'text', 'text': 'Hey there! How can I help you today? 😊'}]},
        #     {'createdAt': '2025-07-17T08:56:29Z', 'role': 'user',
        #      'content': [{'type': 'text', 'text': 'How are you'}]}], 'response': [
        #     {'createdAt': '2025-07-17T08:56:31Z', 'run_id': 'run_vZU2pZ4a4QzxYB1tLoei7ycB',
        #      'assistant_id': 'asst_CLx2RNAXhAoFIbkLxZfoM6P4', 'role': 'assistant', 'content': [{'type': 'text',
        #                                                                                         'text': 'Thanks for asking! I’m just a bunch of code, but I’m here and ready to help you. How are you doing?'}]}],
        #  'tool_definitions': []}
        # Filter conversations if belongs to a specific agent
        # try:
        #     # This needs converter changes
        #     if not conversation or (
        #         agent_id and not any(
        #             resp.get("assistant_id") == agent_id for resp in conversation.get("response", [])
        #         )
        #     ):
        #         logger.info(f"Skipping conversation {thread_id} for agent {agent_id}.")
        #         return None
        # except Exception as e:
        #     logger.error(f"Error filtering conversation for agent {agent_id} and thread {thread_id}: {str(e)}")
        #     return None
        result = {}
        if evaluators_to_run is None:
            # Fix error: asyncio.run cannot be called from a running event loop
            if not asyncio.get_event_loop().is_running():
                # If not running in an event loop, run the selection synchronously
                evaluation_selection_results = asyncio.run(self._select_evaluators(conversation))
            else:
                # If already in an event loop, run the selection asynchronously
                evaluation_selection_results = asyncio.get_event_loop().run_until_complete(self._select_evaluators(conversation))
            evaluators_to_run = evaluation_selection_results.get("evaluators", self._DEFAULT_AGENT_EVALUATORS)
            print(f"Selected evaluators: {evaluators_to_run} for thread {thread_id}")
            result["justification"] = evaluation_selection_results.get("justification", "")
            result["distinct_assessments"] = evaluation_selection_results.get("distinct_assessments", "")
        if not evaluators_to_run:
            logger.warning(f"No evaluators to run for thread {thread_id}.")
            result["thread_id"] = thread_id
            result["results"] = {}
            result["conversation"] = conversation
            return result
        evaluator_instances = {name: self.evaluator_instances[name] for name in evaluators_to_run}
        print(f"Running evaluators: {list(evaluator_instances.keys())} on thread {thread_id}")
        conversation_results = self._run_evaluators_on_conversation(
            evaluator_instances, conversation
        )
        result["thread_id"] = thread_id
        result["results"] = conversation_results
        result["conversation"] = conversation
        return result

    def _fetch_agent_threads(self, project_client: Any, agent_id: str) -> List[str]:
        """
        Fetch conversation data for an agent from Azure AI Project.
        This implements the pattern from the Azure AI samples for agent evaluation.

        :param project_client: The AI Project client
        :type project_client: AIProjectClient
        :param agent_id: The agent ID
        :type agent_id: str
        :return: List of thread IDs for the agent
        :rtype: List[str]
        """
        try:

            threads = project_client.agents.threads.list()
            # threads = project_client.agents.threads.list(agent_id=agent_id)

            thread_ids = []
            # threads = ["thread_oDhHDz6HgjTqSLE8FVJy6Ord"]
            for thread in threads:
                # print(thread)
                thread_ids.append(thread.id)
                # Convert to evaluation format
            print(f"Fetched {len(thread_ids)} threads for agent {agent_id}.")
            return thread_ids

        except Exception as e:
            logger.error(f"Error fetching agent threads: {str(e)}")
            raise

    def _initialize_evaluators(self, evaluator_names: List[str]) -> Dict[str, Any]:
        """
        Initialize the specified evaluators.

        :param evaluator_names: List of evaluator names to initialize
        :type evaluator_names: List[str]
        :return: Dictionary of initialized evaluators
        :rtype: Dict[str, Any]
        """
        evaluators = {}

        for name in evaluator_names:
            if name == "IntentResolution":
                evaluators[name] = IntentResolutionEvaluator(model_config=self.model_config)
            elif name == "ToolCallAccuracy":
                evaluators[name] = ToolCallAccuracyEvaluator(model_config=self.model_config)
            elif name == "TaskAdherence":
                evaluators[name] = TaskAdherenceEvaluator(model_config=self.model_config)
            else:
                logger.warning(f"Unknown evaluator: {name}")

        return evaluators

    def _run_evaluators_on_conversation(self,
                                        evaluators: Dict[str, Any],
                                        conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run multiple evaluators on a conversation in parallel.

        :param evaluators: Dictionary of evaluator instances
        :type evaluators: Dict[str, Any]
        :param conversation_data: Conversation data to evaluate
        :type conversation_data: Dict[str, Any]
        :return: Evaluation results
        :rtype: Dict[str, Any]
        """
        results = {}

        # Extract evaluation inputs from conversation data
        query = conversation_data.get("query", "")
        response = conversation_data.get("response", "")
        tool_definitions = conversation_data.get("tool_definitions", [])
        tool_calls = [r.get("content", [{}])[0].get("tool_call_id", None) for r in response if r.get("content", [{}])[0].get("tool_call_id", None) is not None]
        
        # Run evaluators in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_evaluator = {}

            for name, evaluator in evaluators.items():
                if name == "IntentResolution":
                    future = executor.submit(
                        evaluator,
                        query=query,
                        response=response,
                        tool_definitions=tool_definitions if tool_definitions else None
                    )
                elif name == "ToolCallAccuracy":
                    if tool_calls:  # Only run if there are tool calls
                        future = executor.submit(
                            evaluator,
                            query=query,
                            response=response,
                            tool_calls=tool_calls,
                            tool_definitions=tool_definitions
                        )
                    else:
                        logger.info("No tool calls found, skipping ToolCallAccuracy evaluation.")
                        continue
                elif name == "TaskAdherence":
                    future = executor.submit(
                        evaluator,
                        query=query,
                        response=response,
                        tool_definitions=tool_definitions if tool_definitions else None
                    )
                else:
                    continue

                future_to_evaluator[future] = name

            # Collect results
            for future in as_completed(future_to_evaluator):
                evaluator_name = future_to_evaluator[future]
                try:
                    result = future.result()
                    results[evaluator_name] = result
                except Exception as e:
                    logger.error(f"Evaluator {evaluator_name} failed: {str(e)}")
                    results[evaluator_name] = {"error": str(e)}

        return results

    async def _select_evaluators(self, eval_input: Dict[str, Any]):
        """
        Select appropriate evaluators based on the conversation history and tool definitions.

        :param eval_input: Input dictionary containing conversation history and tool definitions
        :type eval_input: Dict[str, Any]
        """
        # This would use the prompty to select evaluators
        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger)
        tool_definitions = eval_input.get("tool_definitions", None)
        if tool_definitions:
            eval_input["tool_definitions"] = reformat_tool_definitions(tool_definitions, logger)

        llm_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        if isinstance(llm_output, dict):
            evaluators = llm_output.get("evaluators", [])
            evaluators = [evaluator for evaluator in evaluators if evaluator in self._DEFAULT_AGENT_EVALUATORS]
            return {
                "evaluators": evaluators,
                "justification": llm_output.get("justification", ""),
                "distinct_assessments": llm_output.get("distinct_assessments", {}),
            }
        if logger:
            logger.warning("LLM output is not a dictionary, returning empty result.")
        return {"evaluators": [], "justification": "", "distinct_assessments": {}}

    # async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[str, List[str]]]:
    #     """Perform evaluator selection based on the provided inputs."""
    #     if "conversation_history" not in eval_input:
    #         raise EvaluationException(
    #             message="Conversation history must be provided as input to the CriticAgent.",
    #             internal_message="Missing conversation history in input.",
    #             blame=ErrorBlame.USER_ERROR,
    #             category=ErrorCategory.MISSING_FIELD,
    #             # target=ErrorTarget.CRITIC_AGENT,
    #         )
    #     eval_input["conversation_history"] = reformat_conversation_history(eval_input["conversation_history"], logger)
    #     if "tool_definitions" in eval_input and eval_input["tool_definitions"] is not None:
    #         eval_input["tool_definitions"] = reformat_tool_definitions(eval_input["tool_definitions"], logger)
    #     llm_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
    #     if isinstance(llm_output, dict):
    #         return {
    #             "evaluators": llm_output.get("evaluators", []),
    #             "justification": llm_output.get("justification", ""),
    #             "distinct_assessments": llm_output.get("distinct_assessments", {}),
    #         }
    #     if logger:
    #         logger.warning("LLM output is not a dictionary, returning empty result.")
    #     return {"evaluators": [], "justification": "", "distinct_assessments": {}}

    def analyze_errors(
        self,
        evaluation_results: List[Dict[str, Any]],
        fails_only: bool = True,
        num_clusters: int = 10,
        agent_id: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze errors in the evaluation results using the ErrorAnalyzer.

        :param evaluation_results: List of evaluation results to analyze
        :type evaluation_results: List[Dict[str, Any]]
        :param fails_only: Whether to include only failed evaluations
        :type fails_only: bool
        :param num_clusters: Number of clusters for error analysis
        :type num_clusters: int
        :param use_llm_analysis: Whether to use LLM for error analysis
        :type use_llm_analysis: bool
        :return: Error analysis report
        :rtype: Dict[str, Any]
        """
        data_analyzer = DataAnalyzer()
        # format results for analysis
        def extract_score_from_eval(data):
            """Extract the evaluation score from the analysis report."""
            for k,v in data.items():
                if ("_result" not in k) and ("_threshold" not in k) and ("_reason" not in k):
                    return {"score": v}
            return None
        
        # transform to be used by dataanalyzer
        data_analyzer_input = []
        for result in evaluation_results:
            for eval, evaluation in result.get("results", {}).items():
                metadata = extract_score_from_eval(evaluation)
                metadata["study"] = "evaluation error"
                metadata["evaluator"] = eval
                metadata["thread_id"] = result.get("thread_id")
                metadata["agent_id"] = agent_id
                entry = {
                    "context": evaluation,
                    "conversation": result.get("conversation"),
                    "metadata": metadata,
                    "evaluation": evaluation,
                }
                data_analyzer_input.append(entry)
        # return data_analyzer_input

        if fails_only:
        # filter data_analyzer_input to entries with metadata/score less than 3
            data_analyzer_input = [
                entry for entry in data_analyzer_input if entry.get("metadata", {}).get("score", 0) < 3
            ]
            print(f"filtered results to {len(data_analyzer_input)} evaluations with failed status")

        output_file_name = f"formatted_evaluations_fails_only_{agent_id}" if fails_only else f"formatted_evaluations_all_{agent_id}"
        import json
        with open(f"data/{output_file_name}.json", "w") as f:
            json.dump(data_analyzer_input, f, indent=4)
        # report = data_analyzer.analyze(entries=data_analyzer_input, num_clusters=num_clusters)
        # return report

    def visualize_errors(self, data_analysis_results: Dict[str, Any], figsize: tuple = (12, 10)):
        """
        Visualize the error analysis report.

        :param error_analysis_report: The error analysis report to visualize
        :type error_analysis_report: Dict[str, Any]
        :param figsize: Size of the figure for visualization
        :type figsize: tuple
        """
        from data_analyzer import visualize_data_analyzer_2d

        visualize_data_analyzer_2d(data_analysis_results)
        