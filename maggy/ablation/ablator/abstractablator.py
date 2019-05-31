from abc import ABC, abstractmethod


class AbstractAblator(ABC):

    def __init__(self, ablation_study, final_store):
        self.ablation_study = ablation_study
        self.final_store = final_store
        self.trial_buffer = []

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_trial(self, trial=None):
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        pass
