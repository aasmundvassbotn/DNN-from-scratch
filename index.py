from dense import DenseLayer
from neural_network import DenseModel
from loss_functions import mean_squared_error
from keras.datasets import mnist

def main(args=None):
    lr = 0.0001

    model = DenseModel(input_shape=(10))

    model.add(DenseLayer(n_units=128))
    model.add(DenseLayer(n_units=64))
    model.add(DenseLayer(n_units=10))

    model.compile(learning_rate=lr, loss_function=mean_squared_error)
    
    model.fit(None, None, 10, 128)

if __name__ == '__main__':
    main()