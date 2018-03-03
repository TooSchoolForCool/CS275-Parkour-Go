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
        self._tf_sess = None
        # tensorflow x input placeholder
        self._tf_x_in = None
        # tensorflow y output operator
        self._tf_y_out = None


    def __del__(self):
        """Destructor

        Free TensorFlow session
        """
        if self._tf_sess:
            self._tf_sess.close()


    def train(self, train_x, train_y, n_epoch=20, batch_size=100):
        """Training Model

        Args:
            train_x: [np.ndarray] A list of training sample x (observation),
                each x is a vector stored in the format of nd.ndarray
            train_y: [np.ndarray] A list of actions according to each
                sample x. Each action a vector stored in the format 
                of nd.ndarray
        """
        input_x = tf.placeholder(tf.float32, [None, train_x.shape[1]])
        expected_y = tf.placeholder(tf.float32, [None, train_y.shape[1]])
        
        # build inference graph (network topology)
        y_out = self._inference(input_x, train_x.shape[1], train_y.shape[1])

        loss = tf.reduce_mean(tf.square(y_out - expected_y))
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_op = optimizer.minimize(loss)

        # initialize the variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(n_epoch):
            for idx in self._shuffle_samples(train_x.shape[0], batch_size):
                feed_dict = {input_x : train_x[idx, :], expected_y : train_y[idx, :]}
                _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)

            print("Epoch %d loss = %r" % (i + 1, loss_val))

        # save metadata
        self._tf_sess = sess
        self._tf_x_in = input_x
        self._tf_y_out = y_out


    def predict(self, observation):
        """Prediction Process
        
        Args:
            observation: [np.ndarray] A list of training sample x (observation),
                each x is a vector stored in the format of nd.ndarray

        Returns:
            predicted_action: [np.ndarray] A list of predicted actions
        """
        predicted_action = self._tf_sess.run(self._tf_y_out, 
            feed_dict={self._tf_x_in : observation})

        return predicted_action


    def _inference(self, input_x, x_dimension, y_dimension):
        """Inference

        Building nueral network topology

        Args:
            input_x: input layer of neural network
            x_dimension: input data dimension, i.e., dimension of each
                data sample
            y_dimension: output data dimension, i.e., dimension of each
                output data sample

        Returns:
            output layer operator
        """
        assert(len(self._hidden_layers) > 0)

        # generate hidden layer
        hidden_layer = input_x
        for i, n_nodes in enumerate(self._hidden_layers):
            hidden_layer = self._add_layer(hidden_layer, n_nodes)
            # add activation function at each node
            hidden_layer = tf.nn.relu(hidden_layer)

        # generate output layer
        output_layer = self._add_layer(hidden_layer, y_dimension)

        return output_layer


    def _add_layer(self, layer_in, n_hidden_nodes):
        """Add layer

        Add a layer into neural network

        Args:
            layer_in: previous layer operator
            n_hidden_nodes: [integer] number of nodes in current layer

        Returns:
            output layer operator without activation function
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


    def _shuffle_samples(self, n_samples, batch_size):
        """Shuffle Dataset

        Shuffle the training dataset by shuffling its indices

        Args:
            n_samples: number of samples, i.e., size of dataset
            batch_size: batch size of training process

        Returns:
            A list of np.ndarrays. Each item, which is a 
            np.ndarray whose size is batch_size, contains indices
            of a batch of training samples
        """
        shuffle_set = []

        dataset_idxs = np.arange(n_samples)
        np.random.shuffle(dataset_idxs)

        for i in range(n_samples // batch_size):
            start_idx = i * batch_size
            idx = dataset_idxs[start_idx : start_idx + batch_size]
            shuffle_set.append(idx)

        if n_samples % batch_size:
            idx = dataset_idxs[n_samples - (n_samples % batch_size) : ]
            shuffle_set.append(idx)

        return shuffle_set