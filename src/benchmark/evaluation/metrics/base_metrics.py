from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List # Added List

class BaseMetric(ABC):
    """
    Abstract base class for stateful evaluation metrics.
    """
    def __init__(self):
        self._options: Dict[str, Any] = {} # To store options like normalize, ignore_case etc.

    def set_options(self, **options: Any) -> None:
        """
        Sets options for the metric computation. Called once by the Evaluator.
        :param options: Options for the metric computation.
        """
        self._options = options

    @abstractmethod
    def reset_state(self) -> None:
        """
        Resets all internal state of the metric.
        """
        pass

    @abstractmethod
    def update_state(self, predictions: List[Any], labels: List[Any]) -> None:
        """
        Update the metric's state with a batch of predictions and labels.
        Note: No **kwargs here, options are set via set_options.

        :param predictions: List of model predictions for the current batch.
        :param labels: List of ground truth labels for the current batch.
        """
        pass

    @abstractmethod
    def result(self) -> Union[float, Dict[str, float]]:
        """
        Compute and return the final metric value(s) from the accumulated state.

        :return: Computed metric value or a dictionary of metric values.
        """
        pass