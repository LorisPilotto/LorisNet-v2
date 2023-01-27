import tensorflow as tf


class LinearSeparators(tf.keras.layers.Layer):
    
    def __init__(self,
                 activation='sigmoid',
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
    def build(self, input_shape):
        number_masks = input_shape[1]
        an_input_tensor_dim = input_shape[2]
        self.kernel = self.add_weight('kernel',
                                      shape=[number_masks, an_input_tensor_dim],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        self.bias = self.add_weight('bias',
                                    shape=[number_masks,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        return tf.expand_dims(self.activation(tf.reduce_sum(inputs * self.kernel, -1) + self.bias), -1)