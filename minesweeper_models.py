import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def alexnet3(width, height, lr, outputs=2):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, outputs, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')
    
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

def deepq_network(width, height, lr=0.001, outputs=2):
        network = input_data(shape=[None, width, height, 1], name='input')
        network = conv_2d(network, 32, 8, strides=2, activation='relu')
        network = conv_2d(network, 64, 4, strides=2, activation='relu')
        network = fully_connected(network, 256, activation='softmax')
        network = fully_connected(network, outputs, activation='softmax')
        network = regression(network, loss='categorical_crossentropy',
                             optimizer='adam', learning_rate=0.001, name='targets')

        model = tflearn.DNN(network, checkpoint_path='model_deepq',
                            max_checkpoints=1, tensorboard_verbose=2,
                            tensorboard_dir='log')
        

        return model

