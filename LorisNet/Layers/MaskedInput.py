import tensorflow as tf


class ActivationMaskedInputNoFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 activation,
                 masks_weights_initializer="glorot_uniform",
                 masks_weights_regularizer=None,
                 masks_weights_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.number_masks = number_masks
        self.activation = tf.keras.activations.get(activation)
        self.masks_weights_initializer = tf.keras.initializers.get(masks_weights_initializer)
        self.masks_weights_regularizer = tf.keras.regularizers.get(masks_weights_regularizer)
        self.masks_weights_constraint = tf.keras.constraints.get(masks_weights_constraint)
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[1]
        self.masks_weights = self.add_weight('masks_weights',
                                             shape=[self.number_masks, an_input_tensor_dim],
                                             initializer=self.masks_weights_initializer,
                                             regularizer=self.masks_weights_regularizer,
                                             constraint=self.masks_weights_constraint,
                                             dtype=self.dtype,
                                             trainable=True)
        super().build(input_shape)
    
    def get_masks(self):
        return self.activation(self.masks_weights)
        
    def call(self, inputs):
        return self.get_masks() * tf.expand_dims(inputs, 1), self.get_masks()
    
    
class ActivationMaskedInputWithFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 activation,
                 masks_weights_initializer="glorot_uniform",
                 masks_weights_regularizer=None,
                 masks_weights_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.number_masks = number_masks
        self.activation = tf.keras.activations.get(activation)
        self.masks_weights_initializer = tf.keras.initializers.get(masks_weights_initializer)
        self.masks_weights_regularizer = tf.keras.regularizers.get(masks_weights_regularizer)
        self.masks_weights_constraint = tf.keras.constraints.get(masks_weights_constraint)
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[0][1]
        self.masks_weights = self.add_weight('masks_weights',
                                             shape=[self.number_masks, an_input_tensor_dim],
                                             initializer=self.masks_weights_initializer,
                                             regularizer=self.masks_weights_regularizer,
                                             constraint=self.masks_weights_constraint,
                                             dtype=self.dtype,
                                             trainable=True)
        super().build(input_shape)
    
    def get_masks(self):
        return self.activation(self.masks_weights)
        
    def call(self, inputs):
        inputs_tensor, prior_masks_tensor = inputs
        return self.get_masks() * tf.expand_dims(inputs_tensor, 1), self.get_masks()
    
    
class SigmoidMaskedInputNoFeedback(ActivationMaskedInputNoFeedback):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(number_masks=number_masks,
                         activation='sigmoid',
                         **kwargs)


class SigmoidMaskedInputWithFeedback(ActivationMaskedInputWithFeedback):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(number_masks=number_masks,
                         activation='sigmoid',
                         **kwargs)
        
        
class SoftmaxMaskedInputNoFeedback(ActivationMaskedInputNoFeedback):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(number_masks=number_masks,
                         activation='softmax',
                         **kwargs)


class SoftmaxMaskedInputWithFeedback(ActivationMaskedInputWithFeedback):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(number_masks=number_masks,
                         activation='softmax',
                         **kwargs)
    
    
class FracOnesMaskedInputNoFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 frac,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        self.frac = frac
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[1]
        self.masks = tf.cast(tf.random.uniform((self.number_masks, an_input_tensor_dim)) <= self.frac,
                             dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        return self.masks * tf.expand_dims(inputs, 1), self.masks
    
    
class FracOnesMaskedInputWithFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 frac,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        self.frac = frac
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[0][1]
        self.masks = tf.cast(tf.random.uniform((self.number_masks, an_input_tensor_dim)) <= self.frac,
                             dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_tensor, prior_masks_tensor = inputs
        return self.masks * tf.expand_dims(inputs_tensor, 1), self.masks
    
    
class AllOnesMaskedInputNoFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[1]
        self.masks = tf.ones((self.number_masks, an_input_tensor_dim), dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, 1), self.number_masks, 1), self.masks
        
        
class AllOnesMaskedInputWithFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[0][1]
        self.masks = tf.ones((self.number_masks, an_input_tensor_dim), dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_tensor, prior_masks_tensor = inputs
        return tf.repeat(tf.expand_dims(inputs_tensor, 1), self.number_masks, 1), self.masks
        
        
class RandomMaskedInputNoFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[1]
        self.masks = tf.random.uniform((self.number_masks, an_input_tensor_dim),
                                       dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        return self.masks * tf.expand_dims(inputs, 1), self.masks
    
    
class RandomMaskedInputWithFeedback(tf.keras.layers.Layer):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
        
    def build(self, input_shape):
        an_input_tensor_dim = input_shape[0][1]
        self.masks = tf.random.uniform((self.number_masks, an_input_tensor_dim),
                                       dtype=self.dtype)
        super().build(input_shape)
        
    def call(self, inputs):
        inputs_tensor, list_prior_masks_tensors = inputs
        return self.masks * tf.expand_dims(inputs_tensor, 1), self.masks