# borrowing A2C TF2.0 functionality from 
# http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/

import numpy as np 
import tensorflow as tf

import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from PIL import Image
from vizdoom import *


class probabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis = -1)


def process_frame(frame):
    # transforms a frame for model processing.
    # *does not convert it to a tensor, that happens in Model.call*
    s = frame[10:-10, 30:-30]
    s = np.array(Image.fromarray(s).resize((84,84)))
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    
    return s


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        
        # Convolutional layers
        self.conv1 = kl.Conv2D(filters = 16, kernel_size=[8,8], strides = [4,4], padding='valid', activation=tf.nn.elu)
        self.conv2 = kl.Conv2D(filters = 32, kernel_size=[4,4], strides = [2,2], padding='valid', activation=tf.nn.elu)
        self.flattened = kl.Flatten()
        self.cnn_out = kl.Dense(256, activation=tf.nn.elu)
        #self.conv_seq = tf.keras.Sequential([self.conv1, self.conv2, self.flattened, self.cnn_out])

        # LSTM

        # Value Head
        self.value = kl.Dense(1, name='value')

        # Policy (logits) Head
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = probabilityDistribution()


    def call(self, inputs):
        # convert to tensor and reshape
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = tf.reshape(x, shape=[-1,84,84,1]) # batch, row, col, channels (using greyscale so 1)

        # common trunk operations
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flattened(x)
        x = self.cnn_out(x)

        # separate hidden layers from the same input tensor
        hidden_logs = self.logits(x)
        hidden_vals = self.value(x)
        return hidden_logs, hidden_vals

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class Worker():
    def __init__(self, game, name, num_actions, vis=False):
        self.name = f'worker_{str(name)}'
        self.episode_rewards=[]
        self.episode_lengths=[]
        self.episode_mean_values=[]

        self.local_AC = Model(num_actions)

        # set up Doom params
        #game.set_doom_scenario_path("defend_the_center.wad") #This corresponds to the simple task we will pose our agent
        game.set_doom_scenario_path("basic.wad")
        game.set_doom_map("map01")
        #game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_resolution(ScreenResolution.RES_400X300)
        game.set_screen_format(ScreenFormat.GRAY8)
        #game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        #game.add_available_button(Button.MOVE_LEFT)
        #game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(vis)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        #game.set_death_penalty(1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = np.identity(num_actions,dtype=bool).tolist()
        #End Doom set-up
        self.env = game

    def work(self, max_episode_length):

        while True: # outer loop, keep starting game episodes
            self.env.new_episode() # start doom episode
            
            while self.env.is_episode_finished() == False: # inner loop, play episode
                # get new screen
                s = self.env.get_state().screen_buffer
                s = process_frame(s)

                # get action and value from model
                action, value = self.local_AC.action_value(s[None,:])
                print('action, value', action, value)

                # make action and get reward
                r = self.env.make_action(self.actions[action]) / 100.0
                print('reward', r)

                

                



if __name__ == '__main__':
    my_little_worker = Worker(DoomGame(), name=0, num_actions=3, vis=True)
    my_little_worker.work(300)

