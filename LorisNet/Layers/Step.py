import tensorflow as tf


class StepNoFeedback(tf.keras.layers.Layer):
    
    def __init__(self,
                 masked_input_no_feedback_layer,
                 vectorial_space_separators,
                 prediction_neurons,
                 **kwargs):
        super().__init__(**kwargs)
        self.masked_input_no_feedback_layer = masked_input_no_feedback_layer
        self.vectorial_space_separators = vectorial_space_separators
        self.prediction_neurons = prediction_neurons
        
    def call(self, inputs):
        masked_inputs, masks = self.masked_input_no_feedback_layer(inputs)
        vectorial_space_indicators = self.vectorial_space_separators(masked_inputs)
        predictions = self.prediction_neurons(vectorial_space_indicators)
        return predictions, masks, vectorial_space_indicators
    
    
class StepWithFeedback(tf.keras.layers.Layer):
    
    def __init__(self,
                 masked_input_with_feedback_layer,
                 vectorial_space_separators,
                 prediction_neurons,
                 **kwargs):
        super().__init__(**kwargs)
        self.masked_input_with_feedback_layer = masked_input_with_feedback_layer
        self.vectorial_space_separators = vectorial_space_separators
        self.prediction_neurons = prediction_neurons
        
    def call(self, inputs):
        inputs_tensor, prior_masks_tensor, prior_vectorial_space_indicators_tensor = inputs  # prior_vectorial_space_indicators_tensor has dim == (nbr samples, nbr masks, nbr previous steps)
        masked_inputs, masks = self.masked_input_with_feedback_layer([inputs_tensor, prior_masks_tensor])
        vectorial_space_indicators = self.vectorial_space_separators(masked_inputs)
        predictions = self.prediction_neurons(tf.concat([prior_vectorial_space_indicators_tensor,
                                                         vectorial_space_indicators], -1))
        return predictions, masks, vectorial_space_indicators