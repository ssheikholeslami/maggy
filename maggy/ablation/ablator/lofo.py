from maggy.ablation.ablator import AbstractAblator


class LOFO(AbstractAblator):

    def calculate_number_of_trials(self):
        pass

    def get_dataset_generator(self, hops_dataset, dataset_type='tfrecord'):
        """
        Create and return a dataset generator function based on the ablation policy to be used in a trial.
        The returned function will be executed on the executor per each trial.

        :param hops_dataset:
        :param dataset_type: type of the dataset. For now, we only support 'tfrecord'.
        :return: A function that generates a TFRecordDataset
        :rtype: function
        """

        if self.ablation_study.custom_dataset_generator:
            pass # XXX process and change the custom dataset generator
        else:
            # XXX RESUME
            # create and return a dataset generator
            pass

    def get_model_generator(self):
        pass

    def initialize(self):

        pass

    def get_trial(self, trial=None):
        # set a new model
        pass

    def finalize_experiment(self, trials):
        pass
