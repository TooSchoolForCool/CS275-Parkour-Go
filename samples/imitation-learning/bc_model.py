import tensorflow as tf

class BCModel(object):
    """Behavioral Cloning Model

    This model implement a imitation learning algorithm named 
    behavioral cloning by utilizing TensorFlow Neural Network.
    """

    def __init__(self, n_hidden_layers=5, n_hidden_nodes=40, learning_rate=0.01):
        """Constructor

        Args:
            n_hidden_layers: [integer] number of hidden layers
            n_hidden_nodes: [integer] number of nodes at each hidden layers
            learning_rate: [float] learning rate of the learning model, e.g., 
                step size for gradient descent
        """
        self.n_hidden_layers_ = n_hidden_layers
        self.n_hidden_nodes_ = n_hidden_nodes
        self.learning_rate_ = learning_rate
        
        # tensorflow session
        self.sess_ = None


    def __del__(self):
        """Destructor

        Free TensorFlow session
        """
        if self.sess_:
            self.sess_.close()


    def train(self, train_x, train_y):
        """Training Model

        Args:
            train_x:
            train_y
        """
        pass


    def predict(self, predict_x):
        """Prediction Process
        
        Args:
            predict_x

        Returns:
            predict_y
        """
        pass
