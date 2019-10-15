import numpy as np
import tensorflow as tf
import random
import os
import gym
import imageio
from skimage.transform import resize

tf.compat.v1.disable_eager_execution()

train = True

ENV_NAME = 'SpaceInvaders-v0'
#ENV_NAME = 'PongDeterministic-v4'
# You can increase the learning rate to 0.00025 in Pong for quicker results



class FrameProcessor(object):
    def __init__(self, frame_height=84, frame_width=84):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.compat.v1.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize(self.processed,
                                                [self.frame_height, self.frame_width])

    def __call__(self, session, frame):
        return session.run(self.processed, feed_dict={self.frame:frame})



class DQN(object):
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                frame_height=84, frame_width=84, agent_history_length=4):

        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.compat.v1.placeholder(shape=[None, self.frame_height,
                                    self.frame_width, self.agent_history_length], dtype=tf.float32)

        self.inputscaled = self.input/255

        self.conv1 = tf.nn.conv2d(
            input=self.inputscaled, filters=32, kernel_size=[8,8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding='valid', activation=tf.nn.relu, use_bias=False, name='conv1'
        )

        self.conv2 = tf.nn.conv2d(
            input=self.conv1, filters=64, kernel_size=[4,4], strides=2,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding='valid', activation=tf.nn.relu, use_bias=False, name='conv2'
        )

        self.conv3 = tf.nn.conv2d(
            input=self.conv2, filters=64, kernel_size=[3,3], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding='valid', activation=tf.nn.relu, use_bias=False, name='conv3'
        )

        self.conv4 = tf.nn.conv2d(
            input=self.conv3, filters=hidden, kernel_size=[7,7], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding='valid', activation=tf.nn.relu, use_bias=False, name='conv4'
        )

        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)

        self.advantagestream = tf.nn.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='advantage'
        )

        self.value = tf.nn.dense(
            inputs=self.valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value'
        )

        self.q_values = self.value + tf.subtract(self.advantage,
            self.reduce_mean(self.advantage, axis=1, keepdims=True))

        self.best_actions = tf.argmax(self.q_values, 1)

        self.target_q = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

        self.action = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)

        self.Q = tf.reduce_sum(tf.multiply(self.q_values,
            tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)

        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)


class ExplorationExploitationScheduler(object):
    def __init__(self, DQN, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1000000,
                 replay_memory_start_size=50000, max_frames=25000000):

        self.DQN = DQN
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) \
                        /(self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final - self.slope_2 * self.max_frames

    def get_action(self, session, frame_number, state, evaluation=False):

        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < \
                    self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * self.frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * self.frame_number + self.intercept_2

        if(np.random.rand(1) < eps):
            return np.random.randint(0, self.n_actions)
        return session.run(self.DQN.best_action, feed_dict={self.DQN.input:[state]})[0]

class ReplayMemory(object):
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                    agent_history_length=4, batch_size=32):

        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):

        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.curent] = action
        self.frames[self.current] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):

        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states = self._get_state(idx)

        return np.transpose(self.states, axes=(0,2,3,1)), self.actions[self.indices],\
            self.rewards[self.indices], np.transpose(self.new_states, axes=(0,2,3,1)),\
            self.terminal_flags[self.indices]



def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    # Take some minibatch from replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()

    # Below actions helps reduce the overestimation from single network choosing
    # both target and prediction - by having the target pull the values, and the
    # prediction pull the index we reduce the likelihood of divergence and local minima

    # use main dqn to pull best action
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # use target dqn to pull q vals
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]

    # Now apply Belmann equation to give prediction vs observed
    # 1-terminal returns 0 if terminal state, leaving just reward

    target_q = rewards + (gamme * double_q * (1 - terminal_flags))

    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                            feed_dict={main_dqn.input:states,
                            main_dqn.target_q:target_q,
                            main_dqn.action:actions})

    return loss

class TargetNetworkUpdater(object):
    def __init__(self, main_dqn_vars, target_dqn_vars):
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def __call__(self):
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)


class Atari(object):
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True # Set to true so that the agent starts
                                  # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.process_frame(sess, frame)   # (★★★)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life_lost

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5★)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process_frame(sess, new_frame)   # (6★)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) # (6★)
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1

tf.compat.v1.reset_default_graph()

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 30000000            # Total number of frames the agent sees
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.create_file_writer(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))




# main DQN and target DQN networks:
with tf.compat.v1.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)   # (★★)
with tf.compat.v1.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               # (★★)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')


LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage",
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalar summaries for tensorboard: loss, average reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.compat.v1.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.compat.v1.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.compat.v1.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramm summaries for tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)


def train():
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    explore_explot_sched = ExplorationExploitationScheduler(MAIN_DQN,
                            atari.env.action_space.n, replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
                            max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        sess.run(init)

        frame_number = 0
        rewards = []
        loss_list = []

        while frame_number < MAX_FRAMES:

            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    action = explore_explot_sched.get_action(sess, frame_number, atari.state)

                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)

                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    clipped_reward = clip_reward(reward)

                    my_replay_memory.add_experience(action=action,
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=clipped_reward,
                                                    terminal=terminal_life_lost)

                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)
                        loss_list.append(loss)
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess) # (9★)

                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)

                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH:np.mean(loss_list),
                                                   REWARD_PH:np.mean(rewards[-100:])})

                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)

                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number,
                              np.mean(rewards[-100:]), file=reward_file)

            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False

                # Fire (action 1), when a life was lost or the game just started,
                # so that the agent does not stand around doing nothing. When playing
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else explore_exploit_sched.get_action(sess, frame_number,
                                                                                       atari.state,
                                                                                       evaluation=True)

                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif:
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif

            print("Evaluation score:\n", np.mean(eval_rewards))
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")

            #Save the network parameters
            saver.save(sess, PATH+'/my_model', global_step=frame_number)
            frames_for_gif = []

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)


if(train):
    train()
