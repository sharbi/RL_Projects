import gym
import torch
import numpy as np
import random
import imageio
from progbar import ProgBar


from DDQN import DuelingAgent
from wrappers import make_atari, wrap_deepmind, wrap_pytorch




def generate_gif(frame_number, frames_for_gif, reward, path):

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)



def train(env, agent, max_episodes, max_steps, batch_size, target_update):
    episode_rewards = []

    frame_number = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        frames_for_gif = []
        gif = False

        for step in range(max_steps):
            #env.render()
            action = agent.get_action(state, frame_number)
            next_state, reward, done, _ = env.step(action)
            if episode % 100:
                gif = True
                frames_for_gif.append(next_state)
            frame_number += 1
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size and episode % 10 == 0:
                agent.update(batch_size, True, episode)
            elif len(agent.replay_buffer) > batch_size:
                agent.update(batch_size, False, episode)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        if episode % target_update == 0:
            agent.run_target_update()

        if gif:
            try:
                generate_gif(frame_number, frames_for_gif, episode_rewards[0], '/output/gifs/')
                frames_for_gif = []
            except IndexError:
                print("Game did not finish")
                gif = False


    return episode_rewards

env_id = "SpaceInvadersNoFrameskip-v0"
MAX_EPISODES = 1000000
MAX_STEPS = 30000000
BATCH_SIZE = 32
TARGET_UPDATE = 10

env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)


agent = DuelingAgent(env)
episode_rewards = train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, TARGET_UPDATE)
