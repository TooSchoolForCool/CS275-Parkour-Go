import tensorflow as tf
import numpy as np
import gym


class ActorCritic(object):

    def __init__(self, env, sess):
        self._sess = sess
        
        self._obs_space_dim = env.observation_space.shape[0]
        self._act_space_dim = env.action_space.shape[0]
        self._act_space_lower_bound = env.action_space.low
        self._act_space_upper_bound = env.action_space.high

        self._entropy_beta = 0.005
        self._actor_learning_rate = 0.00002
        self._critic_learning_rate = 0.0001

        self._actor_hidden_layers = [500, 300, 200]
        self._critic_hidden_layers = [500, 300, 200]


    def _actor_inference(self, layer_in, hidden_layers, activation_func=tf.nn.relu6):
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("actor"):
            hidden_out = self._build_hidden_layer(layer_in, hidden_layers, activation_func)
            
            mu = tf.layers.dense(hidden_out, self._act_space_dim, tf.nn.tanh,
                kernel_initializer=xavier_initializer, name="mu")
            sigma = tf.layers.dense(hidden_out, self._act_space_dim, tf.nn.softplus,
                kernel_initializer=xavier_initializer, name="sigma")

            return mu, sigma


    def _critic_inference(self, layer_in, hidden_layers, activation_func=tf.nn.relu6):
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("critic"):
            hidden_out = self._build_hidden_layer(layer_in, hidden_layers, activation_func)
            
            value = tf.layers.dense(hidden_out, 1, kernel_initializer=xavier_initializer,
                name="value")

            return value

    
    def _build_hidden_layer(self, layer_in, hidden_layers, activation_func=tf.nn.relu6):
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        for i, n_units in enumerate(hidden_layers):
            hidden_layer = tf.layers.dense(layer_in, n_units, activation_func, 
                kernel_initializer=xavier_initializer, name="hidden_" + str(i))
            layer_in = hidden_layer

        return hidden_layer


class ActorCriticMaster(ActorCritic):

    def __init__(self, scope, env, sess):
        ActorCritic.__init__(self, env, sess)

        with tf.variable_scope(scope):
            self._obs_input = tf.placeholder(tf.float32, [None, self._obs_space_dim], name='obs_input')
            
            self._mu, self._sigma = self._actor_inference(self._obs_input, self._actor_hidden_layers)
            self._val = self._critic_inference(self._obs_input, self._critic_hidden_layers)

            self._actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=scope + "/actor")
            self._critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=scope + "/critic")

            self._actor_optimizer = tf.train.RMSPropOptimizer(self._actor_learning_rate)
            self._critic_optimizer = tf.train.RMSPropOptimizer(self._critic_learning_rate)

        print("Init ActorCriticMaster", scope)


class ActorCriticSlave(ActorCritic):

    def __init__(self, scope, env, sess, master_ac):
        ActorCritic.__init__(self, env, sess)

        with tf.variable_scope(scope):
            self._obs_input = tf.placeholder(tf.float32, [None, self._obs_space_dim], name='obs_input')
            self._actions = tf.placeholder(tf.float32, [None, self._act_space_dim], name="actions")
            self._target_vals = tf.placeholder(tf.float32, [None, 1], name="target_val")

            self._mu, self._sigma = self._actor_inference(self._obs_input, self._actor_hidden_layers)
            self._val = self._critic_inference(self._obs_input, self._critic_hidden_layers)

            self._actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=scope + "/actor")
            self._critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=scope + "/critic")

            self._gaussian_dist = tf.contrib.distributions.Normal(self._mu, self._sigma)

            with tf.name_scope("action_control"):
                self._action_selector = tf.clip_by_value(tf.squeeze(self._gaussian_dist.sample(1), axis=0),
                    self._act_space_lower_bound, self._act_space_upper_bound)

            actor_loss, critic_loss = self._loss_function(self._mu, self._sigma, self._val, 
                self._target_vals, self._actions)

            with tf.name_scope("local_gradients"):
                actor_gradients = tf.gradients(actor_loss, self._actor_params)
                critic_gradients = tf.gradients(critic_loss, self._critic_params)

        with tf.name_scope("sync"):
            with tf.name_scope("pull"):
                self._pull_actor_params_op = [lp.assign(gp) for lp, gp in 
                    zip(self._actor_params, master_ac._actor_params)]
                self._pull_critic_params_op = [lp.assign(gp) for lp, gp in 
                    zip(self._critic_params, master_ac._critic_params)]
            with tf.name_scope("push"):
                self._update_actor_master_op = master_ac._actor_optimizer.apply_gradients(
                    zip(actor_gradients, master_ac._actor_params))
                self._update_critic_master_op = master_ac._critic_optimizer.apply_gradients(
                    zip(critic_gradients, master_ac._critic_params))

        print("Init ActorCriticSlave", scope)


    def predict_action(self, observation):
        observation = observation[np.newaxis, :]
        action = self._sess.run(self._action_selector, {self._obs_input : observation})[0]
        
        return action


    def calc_value_func(self, observation):
        observation = observation[np.newaxis, :]
        val = self._sess.run(self._val, {self._obs_input : observation})[0, 0]

        return val


    def update_master(self, observations, actions, target_vals):
        feed_dict = {
            self._obs_input : observations,
            self._actions : actions,
            self._target_vals : target_vals
        }

        self._sess.run([self._update_actor_master_op, self._update_critic_master_op],
            feed_dict=feed_dict)


    def pull_master(self):
        self._sess.run([self._pull_actor_params_op, self._pull_critic_params_op])


    def _loss_function(self, mu, sigma, val, target_val, actions):
        error = tf.subtract(target_val, val, name="error")

        with tf.name_scope("actor_loss"):
            # mu *= self._act_space_upper_bound
            # sigma += 1e-5
            log_prob = self._gaussian_dist.log_prob(actions)
            exp_val = log_prob * error
            entropy = self._gaussian_dist.entropy()
            exp_val = self._entropy_beta * entropy + exp_val
            actor_loss = tf.reduce_mean(-exp_val)

        with tf.name_scope("critic_loss"):
            critic_loss = tf.reduce_mean(tf.square(error))

        return actor_loss, critic_loss
        

def main():
    env = gym.make("BipedalWalker-v2")
    sess = tf.Session()

    master_ac = ActorCriticMaster("master", env, sess)

    a = ActorCriticSlave("slave_0", env, sess, master_ac)
    b = ActorCriticSlave("slave_1", env, sess, master_ac)
    c = ActorCriticSlave("slave_2", env, sess, master_ac)

    sess.close()


if __name__ == '__main__':
    main()