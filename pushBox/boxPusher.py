import gymnasium
import pybullet as p
import pybullet_data
import time
import push_box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from typing import Any, Dict
import torch as th
import time

env = gymnasium.make('pushBox-v0') #render_mode='human'


episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        # time.sleep(1/240)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode: {} Score: {}'.format(episode, score))
# p.disconnect()

# log_path = os.path.join('Training', 'Logs')

# env = gymnasium.make('targetFinder-v0')
# env = DummyVecEnv([lambda: env])

# # Add a callback to training stage for early stopping
# save_path = os.path.join('Training', 'SavedModels')
# stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 100, verbose = 1)
# eval_callback = EvalCallback(env, 
#                             callback_on_new_best = stop_callback,
#                             eval_freq = 10000, 
#                             best_model_save_path = save_path, 
#                             verbose = 1)

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=10000000, callback=eval_callback)

# PPO_Path = os.path.join('Training', 'SavedModels', 'PPO_10M_slide-hinge_noOut')

# model.save(PPO_Path)

