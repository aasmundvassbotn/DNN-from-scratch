import math
import numpy as np

def mean_squared_error(y_true, y_pred) -> float:
  loss: float = np.mean(np.pow((y_true - y_pred), 2))
  return loss
