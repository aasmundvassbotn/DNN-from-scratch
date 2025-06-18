from dense import DenseLayer
from neural_network import DenseModel

def main(args=None):
    dense = DenseLayer(n_units=10)
    model = DenseModel(input_shape=(10))
    model.add(dense)
    

if __name__ == '__main__':
    main()