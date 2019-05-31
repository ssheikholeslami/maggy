import json
import threading
import hashlib


class AblationTrial(object):
    """ Represents a single trial.
    It is used as shared memory between
    the worker thread and rpc server thread. The server thread performs only
    lookups on the `params` attribute.
    """

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    FINALIZED = "FINALIZED"

    def __init__(self, params):
        """Create a new trial object from a hyperparameter combination
        ``params``.

        :param params: A dictionary of Hyperparameters as key value pairs.
        :type params: dict
        """
        self.trial_id = AblationTrial._generate_id(params)
        self.params = params
        self.status = AblationTrial.PENDING
        self.final_metric = None
        self.metric_history = []
        self.start = None
        self.duration = None
        self.lock = threading.RLock()


    def append_metric(self, metric):
        """Append a metric from the heartbeats to the history."""
        with self.lock:
            self.metric_history.append(metric)

    @classmethod
    def _generate_id(cls, params):
        """
        Class method to generate a hash from a hyperparameter dictionary.

        All keys in the dictionary have to be strings. The hash is a to 16
        characters truncated md5 hash and stable across processes.

        :param params: Hyperparameters
        :type params: dictionary
        :raises ValueError: All hyperparameter names have to be strings.
        :raises ValueError: Hyperparameters need to be a dictionary.
        :return: Sixteen character truncated md5 hash
        :rtype: str
        """

        # ensure params is a dictionary
        if isinstance(params, dict):
            # check that all keys are strings
            if False in set(isinstance(k, str) for k in params.keys()):
                raise ValueError(
                    'All hyperparameter names have to be strings.')

            return hashlib.md5(
                json.dumps(params, sort_keys=True).encode('utf-8')
            ).hexdigest()[:16]

        raise ValueError("Hyperparameters need to be a dictionary.")

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        obj_dict = {
            "__class__": self.__class__.__name__
        }

        temp_dict = self.__dict__.copy()
        temp_dict.pop('lock')
        temp_dict.pop('start')

        obj_dict.update(temp_dict)

        return obj_dict
