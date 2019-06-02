import json


class AblationStudy(object):
    def __init__(self, training_dataset_name, training_dataset_version,
                 label_name, num_epochs, batch_size, **kwargs):
        self.features = Features()
        # TODO call the featurestore and save the list of features of the dataset
        self.models = Models()
        self.hops_training_dataset_name = training_dataset_name
        self.hops_training_dataset_version = training_dataset_version
        self.label_name = label_name
        self.custom_dataset_generator = kwargs.get('dataset_generator', False)
        self.num_epochs = num_epochs
        self.batch_size = batch_size


class Features(object):
    # TODO type-checking for all the methods
    def __init__(self):
        self.included_features = set()  # TODO set or list?

    def include(self, *args):
        for arg in args:
            if type(arg) is list:
                for feature in arg:
                    self._include_single_feature(feature)
            else:
                self._include_single_feature(arg)

    def _include_single_feature(self, feature):
        if type(feature) is str:
            self.included_features.add(feature)  # TODO should check with the list retrieved from the featurestore
            # print("included {}".format(feature))  # this still prints even if was duplicate
        else:
            raise ValueError("features.include() only accepts strings or lists of strings, "
                             "but it received {0} which is of type '{1}'."
                             .format(str(feature), type(feature).__name__))

    def exclude(self, *args):
        for arg in args:
            if type(arg) is list:
                for feature in arg:
                    self._exclude_single_feature(feature)
            else:
                self._exclude_single_feature(arg)

    def _exclude_single_feature(self, feature):
        if type(feature) is str:
            if feature in self.included_features:
                self.included_features.remove(feature)
        else:
            raise ValueError("features.exclude() only accepts strings or lists of strings, "
                             "but it received {0} (of type '{1}')."
                             .format(str(feature), type(feature).__name__))

    def list_all(self):
        for feature in self.included_features:
            print(feature)


class Models(object):
    def __init__(self):
        self._included_layers = set()
