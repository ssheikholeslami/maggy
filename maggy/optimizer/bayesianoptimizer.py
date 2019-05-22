"""
A Bayesian Optimizer with Expected Improvement (EI) as the acquisition function.

"""
import random

from maggy.optimizer import AbstractOptimizer
from maggy.searchspace import Searchspace
from maggy.trial import Trial

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import simplefilter, catch_warnings

from sklearn.gaussian_process import GaussianProcessRegressor


class BayesianOptimizer(AbstractOptimizer):

    def __init__(self, num_trials, searchspace, final_store, num_init_points, random_seed=None, should_load_from_logs=False):
        super().__init__(num_trials, searchspace, final_store)

        print("my random_seed is: {}".format(random_seed))
        print("my other stuff are: {0}\n{1}\n{2}".format(self, num_trials, searchspace))

        self.num_init_points = num_init_points
        self.trial_counter = 0 # TODO can be replaced with len(final_store)
        self.should_load_from_logs = should_load_from_logs
        print("Bayesian Optimizer instance created.")

        # TODO leave self._gp here?
        # Internal GP Regressor, defaults inspired by https://github.com/fmfn/BayesianOptimization
        # this can be shared among threads in case of distributed/multi-threaded optimizer
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state= random.uniform(random_seed)  # TODO check
        )

    def initialize(self):
        # TODO flag + method to reinitialize/reconstruct by loading results from JSON/HDFS
        if self.reconstruct_from_logs:
            raise NotImplementedError("Reinitialization from logs not yet implemented.")


        # Suggest random points from the

        print("Bayesian Optimizer initialized.")

    def get_suggestion(self, trial=None):
        """
        Update the internal GP regressor, probe the acquisition function, and return a new trial.
        :param trial: the latest finished trial by the calling executor. Used for models where
        incremental updates are used, otherwise the final_store in the experiment driver is accessible
        by the optimizer and the optimizer can iterate over it.
        :type trial: Trial
        :return: A new trial to be assigned to the executor by the experiment driver.
        :rtype: Trial
        """
        assert isinstance(trial, Trial), \
            ("Instance of {} was passed to get_suggestion() while expected maggy.Trial".format(type(trial)))

        if trial.status != Trial.FINALIZED:
            raise ValueError("Unfinished trial {0} with status {1} was used for getting a new suggestion.".format(
                trial.trial_id,
                trial.status
            ))

        # else

    def finalize_experiment(self, trials):
        pass

    # acquisition function
    # TODO rename parameters to fit Maggy vocabulary
    # TODO make sure no repeated suggestions are provided
    # TODO refactor as an abstract acquisition function
    # TODO do we need trial_buffer here? since get_suggestion might actually be time consuming. "Explore" in this case?

    def expected_improvement(self, points, ):
        pass

