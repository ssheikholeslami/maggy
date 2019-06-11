class AblationStudy(object):
    def __init__(self, training_dataset_name, training_dataset_version,
                 label_name, **kwargs):
        self.features = Features()
        # TODO call the featurestore and save the list of features of the dataset
        self.model = Model()
        self.hops_training_dataset_name = training_dataset_name
        self.hops_training_dataset_version = training_dataset_version
        self.label_name = label_name
        self.custom_dataset_generator = kwargs.get('dataset_generator', False)

    def to_dict(self):
        """
        Returns the ablation study configuration as a Python dictionary.
        :return: A dictionary with ablation study configuration parameters as keys
        (i.e. 'training_dataset_name', 'included_features', etc.)
        :rtype: dict
        """
        ablation_dict = {
            'training_dataset_name': self.hops_training_dataset_name,
            'training_dataset_version': self.hops_training_dataset_version,
            'label_name': self.label_name,
            'included_features': list(self.features.included_features),
            'included_layers': list(self.model.layers.included_layers),
            'custom_dataset_generator': True if self.custom_dataset_generator else False,
        }

        return ablation_dict


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
            print(feature)  # TODO proper printing


class Model(object):
    def __init__(self):
        self.layers = Layers()
        self.base_model_generator = None

    def set_base_model_generator(self, base_model_generator):
        self.base_model_generator = base_model_generator


class Layers(object):
    def __init__(self):
        self.included_layers = set()
        self.included_groups = set()

    def include(self, *args):
        for arg in args:
            if type(arg) is list:
                for layer in arg:
                    self._include_single_layer(layer)
            else:
                self._include_single_layer(arg)

    def _include_single_layer(self, layer):
        if type(layer) is str:
            self.included_layers.add(layer)
            # print("included {}".format(layer))  # this still prints even if was duplicate
        else:
            raise ValueError("layers.include() only accepts strings or lists of strings, "
                             "but it received {0} which is of type '{1}'."
                             .format(str(layer), type(layer).__name__))

    def exclude(self, *args):
        for arg in args:
            if type(arg) is list:
                for layer in arg:
                    self._exclude_single_layer(layer)
            else:
                self._exclude_single_layer(arg)

    def _exclude_single_layer(self, layer):
        if type(layer) is str:
            if layer in self.included_layers:
                self.included_layers.remove(layer)
        else:
            raise ValueError("layers.exclude() only accepts strings or lists of strings, "
                             "but it received {0} (of type '{1}')."
                             .format(str(layer), type(layer).__name__))

    def include_groups(self, *args):
        """
        Adds a group of layers that should be removed from the model together. The groups are specified either
        by being passed as a list of layers, or with a string as a common prefix of their layer names.
        :param args: Strings that should be a common prefix of the name of all the layers in the desired group, or lists
        of strings (as layer names) to indicate groups of layers.
        :type args: str or list
        """
        for arg in args:
            if type(arg) is list:
                self.included_groups.add(frozenset(arg))
            elif type(arg) is str:
                self.included_groups.add(arg)
            else:
                raise ValueError("layers.include_groups() only accepts strings or lists of strings, "
                                 "but it received {0} (of type '{1}')."
                                 .format(str(arg), type(arg).__name__))

    def exclude_groups(self, *args):
        """
        Removes a group of layers from being included in the ablation study. The groups are specified
        either by being passed as a set of layers, or with a string as a common prefix of their layer names.
        :param args: Strings that should be a common prefix of the name of all the layers in the desired group, or sets
        of strings (of layer names) to indicate groups of layers.
        :type args: str or list
        """
        for arg in args:
            if type(arg) is list:
                if frozenset(arg) in self.included_groups:
                    self.included_groups.remove(frozenset(arg))
            elif type(arg) is str:
                if arg in self.included_groups:
                    self.included_groups.remove(arg)
            else:
                raise ValueError("layers.exclude_groups() only accepts strings or lists of strings, "
                                 "but it received {0} (of type '{1}')."
                                 .format(str(arg), type(arg).__name__))

    def print_all(self):
        """
        Prints all single layers that are included in the current ablation study configuration.
        """
        if len(self.included_layers) > 0:
            print("Included single layers are: \n")  # TODO proper printing
            for layer in self.included_layers:
                print(layer)
        else:
            print("There are no single layers in this ablation study configuration.")

    def print_all_groups(self):
        """
        Prints all layer groups that are included in the current ablation study configuration.
        """
        if len(self.included_groups) > 0:
            print("Included layer groups are: \n")  # TODO proper printing
            for layer_group in self.included_groups:
                if type(layer_group) is frozenset:
                    print("--- " + str(list(layer_group)))
                elif type(layer_group) is str:
                    print('---- All layers prefixed "' + layer_group + '"')
        else:
            print("There are no layer groups in this ablation study configuration.")
