"""
The experiment driver implements the functionality for scheduling trials on maggy.
"""
import queue
import threading
import json
import os
import secrets
from datetime import datetime
from sys import maxsize as integer_max
from maggy import util
from maggy.optimizer import AbstractOptimizer, RandomSearch
from maggy.core import rpc
from maggy.trial import Trial
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule, NoStoppingRule
from maggy.searchspace import Searchspace
from maggy.ablation.ablator import AbstractAblator, LOFO
from maggy.ablation.ablationstudy import AblationStudy

from hops import constants as hopsconstants
from hops import hdfs as hopshdfs
from hops import util as hopsutil

import traceback


class ExperimentDriver(object):

    SECRET_BYTES = 8
    # TODO rewrite with @classmethod

    # for now, we infer the experiment type (an optimization experiment or an ablation study)
    # using keyword arguments, and set self.experiment_type to an according string.
    # Some of these arguments are required for any maggy experiment:
    # num_trials, name, num_executors, hb_interval, description, app_dir, log_dir, trial_dir
    # while some are specific to the type of experiment. For example, if the ExperimentDriver constructor
    # is called with a `searchspace` parameter, we infer that it is a hyperparameter optimization task,
    # and if it is called with `ablationstudy` parameter, we infer that it's an ablation study.

    # So we first setup the type-specific experiment requirements, and then the general ones

    def __init__(self, experiment_type, **kwargs):

        # COMMON EXPERIMENT SETUP
        self._final_store = []
        self._trial_store = {}
        self.num_executors = kwargs.get('num_executors')
        self._message_q = queue.Queue()
        self.name = kwargs.get('name')
        self.experiment_done = False
        self.worker_done = False
        self.hb_interval = kwargs.get('hb_interval')
        self.description = kwargs.get('description')
        self.experiment_type = experiment_type
        self.es_interval = integer_max  # XXX not the cleanest way
        self.es_min = integer_max

        # TYPE-SPECIFIC EXPERIMENT SETUP
        if self.experiment_type == 'optimization':
            # set up an optimization experiment

            self.num_trials = kwargs.get('num_trials')

            searchspace = kwargs.get('searchspace')
            if isinstance(searchspace, Searchspace):
                self.searchspace = searchspace
            else:
                raise Exception(
                    "The experiment's search space should be an instance of maggy.Searchspace, "
                    "but it is {0} (of type '{1}')."
                    .format(str(searchspace), type(searchspace).__name__))

            optimizer = kwargs.get('optimizer')
            if isinstance(optimizer, str):
                if optimizer.lower() == 'randomsearch':
                    self.optimizer = RandomSearch(self.num_trials, self.searchspace, self._final_store)
            elif isinstance(optimizer, AbstractOptimizer):
                self.optimizer = optimizer
                print("Custom Optimizer initialized.")  # TODO do we need this print?
            else:
                raise Exception(
                    "The experiment's optimizer should either be an string indicating the name "
                    "of an implemented optimizer (such as 'randomsearch') or an instance of "
                    "maggy.optimizer.AbstractOptimizer, "
                    "but it is {0} (of type '{1}')."
                    .format(str(optimizer), type(optimizer).__name__))

            direction = kwargs.get('direction')
            if isinstance(direction, str):
                if direction.lower() in ['min', 'max']:
                    self.direction = direction.lower()
            else:
                raise Exception(
                    "The experiment's direction should be an string (either 'min' or 'max') "
                    "but it is {0} (of type '{1}')."
                    .format(str(direction), type(direction).__name__))

            es_policy = kwargs.get('es_policy')
            if isinstance(es_policy, str):
                if es_policy.lower() == 'median':
                    self.earlystop_check = MedianStoppingRule.earlystop_check
                    # XXX should also throw an exception if it's a string but not 'median'!
                    # also the same thing for optimizers, etc.
                    # XXX check self.early_stop_check vs. self.earlystop_check
                elif es_policy.lower() == 'none':
                    self.earlystop_check = NoStoppingRule.earlystop_check
            elif isinstance(es_policy, AbstractEarlyStop):
                self.earlystop_check = es_policy.earlystop_check
                print("Custom Early Stopping policy initialized.")  # TODO do we need this print?
            else:
                raise Exception(
                    "The experiment's early stopping policy should either be string ('median' or 'none') "
                    "or a custom policy that is an instance of maggy.earlystop.AbstractEarlyStop, "
                    "but it is {0} (of type '{1}')."
                    .format(str(es_policy), type(es_policy).__name__))

            self.es_interval = kwargs.get('es_interval')
            self.es_min = kwargs.get('es_min')

            self.result = {'best_val': 'n.a.',
                           'num_trials': 0,
                           'early_stopped': 0}

        elif self.experiment_type == 'ablation':
            # set up an ablation study experiment
            self.earlystop_check = NoStoppingRule.earlystop_check
            ablation_study = kwargs.get('ablation_study')
            ablator = kwargs.get('ablator')  # XXX wtf ablator... maybe planner is a better name
            if isinstance(ablator, str):
                if ablator.lower() == 'lofo':
                    self.ablator = LOFO(ablation_study, self._final_store)
                    self.num_trials = self.ablator.get_number_of_trials()
                    if self.num_executors > self.num_trials:
                        self.num_executors = self.num_trials
                else:
                    raise Exception(
                        "The experiment's ablation study policy should either be string ('lofo') "
                        "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                        "but it is {0} (of type '{1}')."
                        .format(str(ablator), type(ablator).__name__))
            elif isinstance(ablator, AbstractAblator):
                self.ablator = ablator
                print("Custom Ablator initialized. \n")
            else:
                raise Exception(
                    "The experiment's ablation study policy should either be string ('lofo') "
                    "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                    "but it is {0} (of type '{1}')."
                    .format(str(ablator), type(ablator).__name__))

            # XXX setup ablation result schema
            self.result = {'best_val': 'n.a.',
                           'num_trials': 0,
                           'early_stopped': 'n.a'}
        else:
            raise Exception(
                "Unknown experiment type. experiment_type should be either 'optimization' or 'ablation', "
                "but it is {0}."
                .format(str(self.experiment_type)))

            # throw exception and exit

        # FINALIZE EXPERIMENT SETUP
        self.server = rpc.Server(self.num_executors)
        self._secret = self._generate_secret(ExperimentDriver.SECRET_BYTES)
        self.job_start = datetime.now()
        self.executor_logs = ''
        self.maggy_log = ''
        self.log_lock = threading.RLock()
        self.log_file = kwargs.get('log_dir') + '/maggy.log'
        self.trial_dir = kwargs.get('trial_dir')
        self.app_dir = kwargs.get('app_dir')

        # Open File desc for HDFS to log
        if not hopshdfs.exists(self.log_file):
            hopshdfs.dump('', self.log_file)
        self.fd = hopshdfs.open_file(self.log_file, flags='w')

    def init(self):

        self.server_addr = self.server.start(self)

        if self.experiment_type == 'optimization':
            self.optimizer.initialize()
        elif self.experiment_type == 'ablation':
            self.ablator.initialize()
            util.quick_log("INITIALIZED ABLATION!")

        try:
            self._start_worker()
        except Exception as e:
            util.quick_log("EXCEPTION: " + traceback.format_exc())

    def finalize(self, job_start, job_end):

        _ = self.optimizer.finalize_experiment(self._final_store)

        self.job_end = datetime.now()

        self.duration = hopsutil._time_diff(self.job_start, self.job_end)


        results = '\n------ ' + str(self.optimizer.__class__.__name__) + ' results ------ direction(' + self.direction + ') \n' \
            'BEST combination ' + json.dumps(self.result['best_hp']) + ' -- metric ' + str(self.result['best_val']) + '\n' \
            'WORST combination ' + json.dumps(self.result['worst_hp']) + ' -- metric ' + str(self.result['worst_val']) + '\n' \
            'AVERAGE metric -- ' + str(self.result['avg']) + '\n' \
            'EARLY STOPPED Trials -- ' + str(self.result['early_stopped']) + '\n' \
            'Total job time ' + self.duration + '\n'
        print(results)

        self._log(results)

        hopshdfs.dump(json.dumps(self.result), self.app_dir + '/result.json')
        sc = hopsutil._find_spark().sparkContext
        hopshdfs.dump(self.json(sc), self.app_dir + '/maggy.json')

        return self.result

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def add_message(self, msg):
        self._message_q.put(msg)

    def _start_worker(self):

        util.quick_log('entered _start_worker, experiment_type is: ' + self.experiment_type)

        def _target_function(self):

            time_earlystop_check = datetime.now()  # only used by earlystop-supporting experiments

            util.quick_log("inside the thread...")

            while not self.worker_done:
                trial = None
                # get a message
                try:
                    # util.quick_log("worker is not done and  trying to get a message")
                    msg = self._message_q.get_nowait()
                    util.quick_log("success with the first try clause... msg is" + str(msg))
                except:
                    msg = {'type': None}

                if (datetime.now() - time_earlystop_check).total_seconds() >= self.es_interval:
                    time_earlystop_check = datetime.now()

                # pass currently running trials to early stop component
                    if len(self._final_store) > self.es_min:
                        self._log("Check for early stopping.")
                        try:
                            to_stop = self.earlystop_check(
                                self._trial_store, self._final_store, self.direction)
                        except Exception as e:
                            self._log(e)
                            to_stop = []
                        if len(to_stop) > 0:
                            self._log("Trials to stop: {}".format(to_stop))
                        for trial_id in to_stop:
                            self.get_trial(trial_id).set_early_stop()

                # depending on message do the work
                # 1. METRIC
                if msg['type'] == 'METRIC':
                    # append executor logs if in the message
                    logs = msg.get('logs', None)
                    if logs is not None:
                        with self.log_lock:
                            self.executor_logs = self.executor_logs + logs

                    if msg['trial_id'] is not None and msg['data'] is not None:
                        self.get_trial(msg['trial_id']).append_metric(msg['data'])

                # 2. BLACKLIST the trial
                elif msg['type'] == 'BLACK':
                    trial = self.get_trial(msg['trial_id'])
                    with trial.lock:
                        trial.status = Trial.SCHEDULED
                        self.server.reservations.assign_trial(
                            msg['partition_id'], msg['trial_id'])

                # 3. FINAL
                elif msg['type'] == 'FINAL':
                    # set status
                    # get trial only once
                    trial = self.get_trial(msg['trial_id'])

                    logs = msg.get('logs', None)
                    if logs is not None:
                        with self.log_lock:
                            self.executor_logs = self.executor_logs + logs

                    # finalize the trial object
                    with trial.lock:
                        trial.status = Trial.FINALIZED
                        trial.final_metric = msg['data']
                        trial.duration = hopsutil._time_diff(
                            trial.start, datetime.now())

                    # move trial to the finalized ones
                    self._final_store.append(trial)
                    self._trial_store.pop(trial.trial_id)

                    # update result dictionary
                    self._update_result(trial)
                    # keep for later in case tqdm doesn't work
                    self.maggy_log = self._update_maggy_log()
                    self._log(self.maggy_log)

                    util.quick_log("Finalized trial... before JSON dump")

                    if self.experiment_type == 'optimization':
                        hopshdfs.dump(trial.to_json(), self.trial_dir + '/' + trial.trial_id + '/trial.json')

                    # assign new trial
                    if self.experiment_type == 'optimization':
                        trial = self.optimizer.get_suggestion(trial)
                    elif self.experiment_type == 'ablation':
                        trial = self.ablator.get_trial(trial)
                    if trial is None:
                        self.server.reservations.assign_trial(
                            msg['partition_id'], None)
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)

                            util.quick_log("TRIAL ASSIGNED AFTER PREVIOUS FINALIZATION: " + str(msg))

                # 4. REG
                elif msg['type'] == 'REG':
                    if self.experiment_type == 'optimization':
                        trial = self.optimizer.get_suggestion()
                    elif self.experiment_type == 'ablation':
                        trial = self.ablator.get_trial()
                    if trial is None:
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)
                            util.quick_log("TRIAL ASSIGNED WITH REGISTRATION: " + str(msg))

        t = threading.Thread(target=_target_function, args=(self,))
        t.daemon = True
        util.quick_log("starting the thread...")
        t.start()

    def stop(self):
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()
        self.fd.flush()
        self.fd.close()

    def json(self, sc):
        """Get all relevant experiment information in JSON format.
        """
        user = None
        if hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR in os.environ:
            user = os.environ[hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR]

        experiment_json = {'project': hopshdfs.project_name(),
            'user': user,
            'name': self.name,
            'module': 'maggy',
            'app_id': str(sc.applicationId),
            'start': self.job_start.isoformat(),
            'memory_per_executor': str(sc._conf.get("spark.executor.memory")),
            'gpus_per_executor': str(sc._conf.get("spark.executor.gpus")),
            'executors': self.num_executors,
            'logdir': self.trial_dir,
            # 'versioned_resources': versioned_resources,
            'description': self.description}
        if self.experiment_type == 'optimization':
            experiment_json['hyperparameter_space'] = json.dumps(self.searchspace.to_dict())
            experiment_json['function'] = self.optimizer.__class__.__name__,

        if self.experiment_done:
            experiment_json['status'] = "FINISHED"
            experiment_json['finished'] = self.job_end.isoformat()
            experiment_json['duration'] = self.duration
            if self.experiment_type == 'optimization':
                experiment_json['hyperparameter'] = json.dumps(self.result['best_hp'])
            experiment_json['metric'] = self.result['best_val']

        else:
            experiment_json['status'] = "RUNNING"

        return json.dumps(experiment_json)

    def _generate_secret(self, nbytes):
        """Generates a secret to be used by all clients during the experiment
        to authenticate their messages with the experiment driver.
        """
        return secrets.token_hex(nbytes=nbytes)

    def _update_result(self, trial):
        """Given a finalized trial updates the current result's best and
        worst trial.
        """

        metric = trial.final_metric
        param_string = trial.params
        trial_id = trial.trial_id

        # First finalized trial
        if self.result.get('best_id', None) is None:
            self.result = {'best_id': trial_id, 'best_val': metric,
                'best_hp': param_string, 'worst_id': trial_id,
                'worst_val': metric, 'worst_hp': param_string,
                'avg': metric, 'metric_list': [metric], 'num_trials': 1,
                'early_stopped': 0}

            if trial.early_stop:
                self.result['early_stopped'] += 1

            return
        # TODO handle for ablation
        if self.direction == 'max':
            if metric > self.result['best_val']:
                self.result['best_val'] = metric
                self.result['best_id'] = trial_id
                self.result['best_hp'] = param_string
            if metric < self.result['worst_val']:
                self.result['worst_val'] = metric
                self.result['worst_id'] = trial_id
                self.result['worst_hp'] = param_string
        elif self.direction == 'min':
            if metric < self.result['best_val']:
                self.result['best_val'] = metric
                self.result['best_id'] = trial_id
                self.result['best_hp'] = param_string
            if metric > self.result['worst_val']:
                self.result['worst_val'] = metric
                self.result['worst_id'] = trial_id
                self.result['worst_hp'] = param_string

        # update average
        self.result['metric_list'].append(metric)
        self.result['num_trials'] += 1
        self.result['avg'] = sum(self.result['metric_list'])/float(
            len(self.result['metric_list']))

        if trial.early_stop:
            self.result['early_stopped'] += 1

    def _update_maggy_log(self):
        """Creates the status of a maggy experiment with a progress bar.
        """
        finished = self.result['num_trials']

        log = 'Maggy ' + str(finished) + '/' + str(self.num_trials) + \
            ' (' + str(self.result['early_stopped']) + ') ' + \
            util._progress_bar(finished, self.num_trials) + ' - BEST ' + \
            json.dumps(self.result['best_hp']) + ' - metric ' + \
            str(self.result['best_val'])

        return log

    def _get_logs(self):
        """Return current experiment status and executor logs to send them to
        spark magic.
        """
        with self.log_lock:
            temp = self.executor_logs
            # clear the executor logs since they are being sent
            self.executor_logs = ''
            return self.result, temp

    def _log(self, log_msg):
        """Logs a string to the maggy driver log file.
        """
        msg = datetime.now().isoformat() + ': ' + str(log_msg)
        self.fd.write((msg + '\n').encode())

