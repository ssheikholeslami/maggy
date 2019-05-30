from abc import ABC, abstractmethod


class AbstractAblator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_trial(self, trial=None):
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        pass
