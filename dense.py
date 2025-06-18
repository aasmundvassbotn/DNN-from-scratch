import numpy as np

class DenseLayer():
  def __init__(self, 
               n_units: int,
               bias: bool = True):
    
    if n_units is None or n_units <= 0:
      raise ValueError('Invalid input parametres')
    
    self.units = n_units

    if bias:
      self.biasmatrix = np.zeros(n_units)
    else:
      self.biasmatrix = None
  
  def forward_prop(self,
                   weights,
                   z_value):
    # TODO
    pass

  def backward_prop(self):
    # TODO
    pass

    

