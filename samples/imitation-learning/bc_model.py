import math
import tensorflow as tf
import numpy as np

class BCModel(object):
    """Behavioral Cloning Model

    This model implement a imitation learning algorithm named 
    behavioral cloning by utilizing TensorFlow Neural Network.
    """

    def __init__(self, hidden_layers, learning_rate=0.01):
        """Constructor

        Args:
            hidden_layers: A list of integers which indicates the nubmer
                of nodes at each hidden layer.
            learning_rate: [float] learning rate of the learning model, e.g., 
                step size for gradient descent
        """
        self._hidden_layers = hidden_layers
        self._learning_rate = learning_rate
        
        # tensorflow session
        self._sess = None


    def __del__(self):
        """Destructor

        Free TensorFlow session
        """
        if self._sess:
            self._sess.close()


    def train(self, train_x, train_y, n_epoch=20, batch_size=100):
        """Training Model

        Args:
            train_x:
            train_y:
        """
        train_idxs = np.arange(train_x.shape[0])
        np.random.shuffle(train_idxs)

        input_x = tf.placeholder(tf.float32, [None, train_x.shape[1]])
        expected_y = tf.placeholder(tf.float32, [None, train_y.shape[1]])
        
        y_out = self._inference(input_x, train_x.shape[1], train_y.shape[1])

        loss = tf.reduce_mean(tf.square(y_out - expected_y))
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_op = optimizer.minimize(loss)

        # initialize the variables
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        for i in range(n_epoch):
            for j in range(train_x.shape[0] // batch_size):
                start_idx = j * batch_size % train_x.shape[0]
                idx = train_idxs[start_idx : start_idx + batch_size]
                feed_dict = {input_x : train_x[idx, :], expected_y : train_y[idx, :]}

                _, loss_val = self._sess.run([train_op, loss], feed_dict=feed_dict)

            print("Epoch %d loss = %r" % (i + 1, loss_val))


    def predict(self, predict_x):
        """Prediction Process
        
        Args:
            predict_x

        Returns:
            predict_y
        """
        pass


    def _inference(self, layer_in, x_dimension, y_dimension):
        """Inference

        Building nueral network topology
        """
        assert(len(self._hidden_layers) > 0)

        # generate hidden layer
        hidden_layer = layer_in
        for i, n_nodes in enumerate(self._hidden_layers):
            hidden_layer = self._add_layer(i, hidden_layer, n_nodes)
            # add activation function at each node
            hidden_layer = tf.nn.relu(hidden_layer)
            
        # generate output layer
        output_layer = self._add_layer(len(self._hidden_layers), hidden_layer, y_dimension)

        return output_layer


    def _add_layer(self, layer_cnt, layer_in, n_hidden_nodes):
        """Add layer
        """
        input_dimension = int(layer_in.shape[1])

        weights = tf.Variable(
            tf.truncated_normal(shape=(input_dimension, n_hidden_nodes),
                stddev=1.0 / math.sqrt(float(input_dimension)))
        )
        biases = tf.Variable(tf.zeros(shape=(1, n_hidden_nodes)) + 0.01)

        # for each node run: x^T \cdot weight + bias
        xTw_plus_b = tf.matmul(layer_in, weights) + biases

        return xTw_plus_b