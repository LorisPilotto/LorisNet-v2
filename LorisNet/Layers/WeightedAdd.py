import tensorflow as tf


class WeightedAdd(tf.keras.layers.Layer):
    """Do a trainable weighted addition of several tensors."""
    
    def __init__(self,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """Initilaizes the WeightedAdd.
        
        Parameters
        ----------
        TODO"""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias_initializer = tf.keras.initializers.get(bias_initializer)
            self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
            self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
    def build(self, input_shape):
        self.nbr_masks = input_shape[1]
        self.units = input_shape[2]
        self.kernel = self.add_weight('kernel',
                                      shape=[self.nbr_masks, 1],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        ret = inputs * self.kernel
        if self.nbr_masks != 1:
            ret = tf.reduce_sum(ret, 1)
        else:
            ret = ret[:, 0]
        if self.use_bias:
            ret += self.bias
        return ret
    
    
class PositiveAndNormalizedConstraint(tf.keras.constraints.Constraint):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, w):
        w = tf.abs(w)
        return w / tf.reduce_sum(w)
    
    
class PositiveAndNormalizedInitializer(tf.keras.initializers.Initializer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, shape, dtype=None, **kwargs):
        ret = tf.random.uniform(shape=shape,
                                minval=.9,
                                maxval=1.,
                                dtype=dtype)
        return ret / tf.reduce_sum(ret)
    
    
class NormalizedWeightedAdd(WeightedAdd):
    
    def __init__(self,
                 **kwargs):
        super().__init__(use_bias=False,
                         kernel_initializer=PositiveAndNormalizedInitializer(),
                         kernel_constraint=PositiveAndNormalizedConstraint(),
                         **kwargs)