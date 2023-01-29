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
        nbr_masks = input_shape[1]
        self.kernel = self.add_weight('kernel',
                                      shape=[nbr_masks,],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[1,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        ret = tf.squeeze(inputs) * self.kernel
        if inputs.shape[1] != 1:
            ret = tf.reduce_sum(ret, 1)
        if self.use_bias:
            ret += self.bias
        return ret