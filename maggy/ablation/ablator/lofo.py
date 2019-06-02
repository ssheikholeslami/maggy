from maggy.ablation.ablator import AbstractAblator
from maggy.ablation.ablationstudy import AblationStudy
from hops import featurestore
import tensorflow as tf
from maggy.ablation import AblationTrial


class LOFO(AbstractAblator):

    # TODO add this logic
    def calculate_number_of_trials(self):
        pass

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
            training_dataset_name = self.ablation_study.training_dataset_name
            training_dataset_version = self.ablation_study.training_dataset_version
            label_name = self.ablation_study.label_name
            batch_size = self.ablation_study.batch_size
            num_epochs = self.ablation_study.num_epochs

            def create_tf_dataset():
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

    def get_model_generator(self):
        pass

    def initialize(self):
        """
        Prepares all the trials for LOFO policy. Trials will consist of `n` dataset generator callables,
        where `n` is equal to the number of features that are included in the ablation study (i.e. the features that
        will be removed one-at-a-time).
        :return:
        """
        for feature in self.ablation_study.features.included_features:
            pass
            # TODO resume here

        pass

    def get_trial(self, trial=None):
        # get a new dataset generator
        # get a new model generator
        # wrap it in a trial
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        pass