from typing import List, Dict, Any
from models.model_loader import ModelLoader
from evaluation.evaluator import Evaluator
import src.utils.logger as logger
import time

class BenchmarkRunner:
    """
    Runs benchmarks on a list of models and tasks.

    :param model_configs: List of dictionaries containing model configurations.
    :param task_configs: List of dictionaries containing task configurations.
    :param evaluation_params: Evaluation parameters like batch size, max length, etc.
    """

    def __init__(self, model_configs: List[Dict[str, Any]], task_configs: List[Dict[str, Any]], evaluation_params: Dict[str, Any]):
        self.model_configs = model_configs
        self.task_configs = task_configs
        self.evaluation_params = evaluation_params
        self.results = {}

    def run(self):
        """
        Runs the benchmark for all models and tasks.
        """
        evaluator = Evaluator(self.evaluation_params)

        for model_config in self.model_configs:
            model_loader = ModelLoader(model_config['checkpoint'], model_config['framework'])
            model_data = model_loader.load()
            model_name = model_config['name']
            self.results[model_name] = {}

            for task_config in self.task_configs:
                logger.info(f"Running {task_config['name']} on {model_name}")
                task_results = self._run_task(model_data['model'], task_config)
                evaluation_results = evaluator.evaluate(task_results)
                self.results[model_name][task_config['name']] = evaluation_results

        return self.results

    def _run_task(self, model, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task on the model (this is a placeholder for the real task logic).

        :param model: The model to run the task with.
        :param task_config: Configuration for the task.
        :return: Results of the task.
        """
        time.sleep(1)  # Simulating task runtime
        return {"accuracy": 0.85, "f1_score": 0.90}  # Mock results for illustration