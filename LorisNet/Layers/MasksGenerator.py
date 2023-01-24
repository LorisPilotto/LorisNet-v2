import tensorflow as tf
from abc import ABCMeta, abstractmethod


class AbstractMasksGenerator(tf.keras.layers.Layer, metaclass=ABCMeta):
    """"""
    
    def __init__(self,
                 number_masks,
                 **kwargs):
        super().__init__(**kwargs)
        self.number_masks = number_masks
    
    @abstractmethod
    def call(self):
        pass
        
    @abstractmethod
    def masks_generation(self):
        pass


class AbstractMasksGeneratorNoFeedback(AbstractMasksGenerator):
    """"""
    
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return self.masks_generation(inputs)