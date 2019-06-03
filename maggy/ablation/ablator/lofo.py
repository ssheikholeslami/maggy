from maggy.ablation.ablator import AbstractAblator
from maggy.ablation.ablationstudy import AblationStudy
from hops import featurestore
import tensorflow as tf
from maggy import Trial
import json


class LOFO(AbstractAblator):

    # TODO add this logic
    def get_number_of_trials(self):
        return len(self.ablation_study.features.included_features)

    def get_dataset_generator(self, ablated_feature, dataset_type='tfrecord'):
        """
        Create and return a dataset generator function based on the ablation policy to be used in a trial.
        The returned function will be executed on the executor per each trial.

        :param ablated_feature: the name of the feature to be excluded from the training dataset.
        Must match a feature name in the corresponding feature group in the feature store.
        :type ablated_feature: str
        :param dataset_type: type of the dataset. For now, we only support 'tfrecord'.
        :return: A function that generates a TFRecordDataset
        :rtype: function
        """

        if self.ablation_study.custom_dataset_generator:
            pass  # TODO process and change the custom dataset generator
        else:
            training_dataset_name = self.ablation_study.hops_training_dataset_name
            training_dataset_version = self.ablation_study.hops_training_dataset_version
            label_name = self.ablation_study.label_name
            batch_size = self.ablation_study.batch_size
            num_epochs = self.ablation_study.num_epochs

            def create_tf_dataset():
                # TODO @Moritz: go with shadowing? i.e., def create_tf_dataset(ablated_feature)?
                SHUFFLE_BUFFER_SIZE = 10000  # XXX parametrize?
                dataset_dir = featurestore.get_training_dataset_path(training_dataset_name,
                                                                     training_dataset_version)
                input_files = tf.gfile.Glob(dataset_dir + '/part-r-*')
                dataset = tf.data.TFRecordDataset(input_files)
                tf_record_schema = featurestore.get_training_dataset_tf_record_schema(training_dataset_name)
                meta = featurestore.get_featurestore_metadata()  # XXX move outside the function?
                training_features = [feature.name
                                     for feature
                                     in meta.training_datasets[training_dataset_name +
                                                               '_'
                                                               + str(training_dataset_version)].features]
                training_features.remove(ablated_feature)
                training_features.remove(label_name)

                def decode(example_proto):
                    example = tf.parse_single_example(example_proto, tf_record_schema)
                    x = []
                    for feature_name in training_features:
                        x.append(tf.cast(example[feature_name], tf.float32))  # XXX parametrize tf.dtype?
                    y = [tf.cast(example[label_name], tf.float32)]
                    return x, y
                dataset = dataset.map(decode).shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).repeat(num_epochs)
                return dataset

            return create_tf_dataset

    def get_model_generator(self, ablated_layer=None):

        base_model_generator = self.ablation_study.model.base_model_generator

        def model_generator():
            base_model = base_model_generator()

            list_of_layers = [layer for layer in base_model.get_config()['layers']]
            for layer in list_of_layers:
                if layer['config']['name'] == ablated_layer:
                    list_of_layers.remove(layer)

            base_json = base_model.to_json()  # TODO support YAML and maybe custom serializers
            new_dict = json.loads(base_json)
            new_dict['config']['layers'] = list_of_layers
            new_json = json.dumps(new_dict)
            new_model = tf.keras.models.model_from_json(new_json)


            return new_model
        return model_generator

    def initialize(self):
        """
        Prepares all the trials for LOFO policy. Trials will consist of `n` dataset generator callables,
        where `n` is equal to the number of features that are included in the ablation study (i.e. the features that
        will be removed one-at-a-time).
        """

        for feature in self.ablation_study.features.included_features:
            trial_dict = {'dataset_function': self.get_dataset_generator(ablated_feature=feature),
                          'model_function': self.ablation_study.base_model_generator}
            # since this is LOFO, not LOLO or LOMO :D
            # model_function = self.get_model_generator()  # TODO check this later
            self.trial_buffer.append(Trial(trial_dict))

    def get_trial(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        pass
