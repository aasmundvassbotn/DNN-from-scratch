import numpy as np

def random_initializer(layer_units: int, input_units: int) -> np.ndarray:
  weights = np.random.random(size=(layer_units, input_units))
  return weights
