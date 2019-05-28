import json
import random


class AblationStudy(object):
    def __init__(self, **kwargs):
        self.features = Features()

        # TODO call the featurestore and save the list of features of the dataset


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
            print("included {}".format(feature))  # TODO this still prints even if was duplicate
        else:
            raise ValueError("features.include() only accepts strings or lists of strings, "
                             "but it received {0} which is of type '{1}'."
                             .format(str(feature), type(feature).__name__))

    def exclude(self, feature):
        if feature in self.included_features:
            self.included_features.remove(feature)

    def list_all(self):
        for feature in self.included_features:
            print(feature)
