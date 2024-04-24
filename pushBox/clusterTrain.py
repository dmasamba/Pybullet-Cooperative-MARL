import gymnasium
import push_box
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from typing import Any, Dict
import torch as th

log_path = os.path.join('Training', 'clusterResults', 'clusterLogs')

env = gymnasium.make('pushBox-v0')
env = DummyVecEnv([lambda: env])

# Add a callback to training stage for early stopping
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 100, verbose = 1)
eval_callback = EvalCallback(env, 
                            callback_on_new_best = stop_callback,
                            eval_freq = 10000, 
                            verbose = 1)

# Learning rate schedule: linearly decreasing from 0.0007 to 0.0001
def linear_lr(progress_remaining: float):
    start_lr = 0.00001
    end_lr = 0.00005
    return end_lr + (start_lr - end_lr) * progress_remaining

# creating the model
model = PPO('MlpPolicy', env, learning_rate=linear_lr, verbose=1, tensorboard_log=log_path)

# load previous model
#PPO_Path = os.path.join('Training', 'clusterResults', 'clusterSavedModels', 'PPO_22_10M_15dgr_cylinder')
#model = PPO.load(PPO_Path, env=env)

# train the model
model.learn(total_timesteps=10000000, callback=eval_callback)

PPO_Path = os.path.join('Training', 'clusterResults', 'clusterSavedModels', 'PPO_42_10M_30dgr_cylinder')

model.save(PPO_Path)
