"""Common tools for all of mtil package."""
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tf_agents.networks.categorical_projection_network import \
    CategoricalProjectionNetwork
from tf_agents.specs import distribution_spec, tensor_spec


class MultiCategoricalProjectionNetwork(CategoricalProjectionNetwork):
    """Joint distribution over independent categorical variables. Useful for
    the milbench suite because the action space is
    {forward,stop,back}x{left,stop,right}x{open,close}."""
    def __init__(self,
                 sample_spec,
                 logits_init_output_factor=0.1,
                 name='MultiCategoricalProjectionNetwork'):
        # TODO: do I have to do anything to support batching?
        action_counts = sample_spec.maximum - sample_spec.minimum + 1
        total_num_outputs = tf.reduce_sum(action_counts)
        if tf.reduce_any(action_counts <= 0) or action_counts.ndim != 1:
            raise ValueError("must have at least two actions available & "
                             "have a 1D action spec")

        output_shape = sample_spec.shape[:-1].concatenate([total_num_outputs])
        output_spec = self._output_distribution_spec(output_shape, sample_spec)

        # call up to class above this one
        super(CategoricalProjectionNetwork,
              self).__init__(input_tensor_spec=None,
                             state_spec=(),
                             output_spec=output_spec,
                             name=name)

        if not tensor_spec.is_bounded(sample_spec):
            raise ValueError('sample_spec must be bounded. Got: %s.' %
                             type(sample_spec))

        if not tensor_spec.is_discrete(sample_spec):
            raise ValueError('sample_spec must be discrete. Got: %s.' %
                             sample_spec)

        self._sample_spec = sample_spec
        self._output_shape = output_shape

        self._projection_layer = tf.keras.layers.Dense(
            self._output_shape.num_elements(),
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=logits_init_output_factor),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='all_logits')

    def _output_distribution_spec(self, output_shape, sample_spec):
        input_param_spec = {
            'all_logits':
            tensor_spec.TensorSpec(shape=output_shape, dtype=tf.float32),
        }
        action_counts = sample_spec.maximum - sample_spec.minimum + 1

        def builder(logits):
            logits_slices = tf.split(logits, action_counts, axis=1)
            return tfp.distributions.Blockwise([
                tfp.distributions.Categorical(logits=logits_slice)
                for logits_slice in logits_slices
            ])

        return distribution_spec.DistributionSpec(builder,
                                                  input_param_spec,
                                                  sample_spec=sample_spec)


class ImageScaleStackLayer(keras.layers.Layer):
    def call(self, x):
        # move uint8 pixel values into [-1, 1]
        float_values = tf.cast(x, tf.float32) / 128.0 - 1
        if len(x.shape) == 5:
            # also stack images in sequence along channels axis
            float_values = tf.concat(
                [float_values[:, i] for i in range(x.shape[1])], axis=-1)
        return float_values

    def compute_output_shape(self, input_shape):
        # "s" represents stacked sequence of recent images; "c" is channels
        if len(input_shape) == 4:
            return input_shape
        assert len(input_shape) == 5, input_shape
        b, s, h, w, c = input_shape
        return (b, h, w, c * s)
