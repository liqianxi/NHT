import tensorflow as tf
from tensorflow.keras import layers
from nht.utils import common_arg_parser


class MLP(tf.keras.Model):
    def __init__(self, inputs, hiddens, out, activation): 
        super(MLP, self).__init__()

        MLP_layers = [tf.keras.Input(shape=(inputs,))]
        for n_units in hiddens:
            MLP_layers = MLP_layers + [layers.Dense(n_units, activation=activation)]

        MLP_layers = MLP_layers + [layers.Dense(out, activation=None)]

        self.net = tf.keras.Sequential(MLP_layers)

    def call(self, x):
        return self.net(x)
        
    def freeze(self):
        for layer in self.net.layers:
            layer.trainable = False
    

if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    my_MLP = MLP(7,[64,64],3,'tanh')
    
    my_MLP.net.summary()
    print('layer config')
    for layer in my_MLP.net.layers:
        print(layer.get_config())