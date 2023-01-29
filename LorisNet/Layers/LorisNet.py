import tensorflow as tf


class LorisNetLayer(tf.keras.layers.Layer):
    
    def __init__(self,
                 steps,
                 weighted_addition,
                 use_batch_normalization=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.use_batch_normalization = use_batch_normalization
        if self.use_batch_normalization:
            self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.weighted_addition = weighted_addition
        
    def forward(self, inputs):
        if self.use_batch_normalization:
            inputs = self.batch_normalization(inputs)
        predictions, masks, vectorial_space_indicators = self.steps[0](inputs)
        prior_masks_tensor = tf.expand_dims(masks, 1)
        for step in self.steps[1:]:
            tmp_predictions, tmp_masks, tmp_vectorial_space_indicators = step([inputs, prior_masks_tensor, vectorial_space_indicators])
            predictions += tmp_predictions
            prior_masks_tensor = tf.concat([prior_masks_tensor, tf.expand_dims(tmp_masks, 1)], 1)
            vectorial_space_indicators = tf.concat([vectorial_space_indicators, tmp_vectorial_space_indicators], -1)
        final_prediction = self.weighted_addition(predictions)
        return final_prediction, prior_masks_tensor, vectorial_space_indicators
    
    def call(self, inputs):
        return self.forward(inputs)[0]