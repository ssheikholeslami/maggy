from abc import ABC, abstractmethod


class AbstractAblator(ABC):

    def __init__(self, ablation_study, final_store):
        self.ablation_study = ablation_study
        self.final_store = final_store
        self.trial_buffer = []
        self.number_of_trials = 0  # XXX wtf

    @abstractmethod
    def calculate_number_of_trials(self):
        pass

    @abstractmethod
    def get_dataset_generator(self, hops_dataset, dataset_type='tfrecord'):
        pass

    @abstractmethod
    def get_model_generator(self):
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

