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

# Tuned hyperparameters from optuna optimizer round 3
gamma = 0.0023395703784496024
max_grad_norm = 1.1818923402292356
gae_lambda = 0.17410391318934526
exponent_n_steps = 5
lr = 0.0004038539267312125
ent_coef = 0.0010308019410639854

policy_kwargs = dict(ortho_init=True,
                     activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64], vf=[64]))

# creating the model
model = PPO('MlpPolicy', env, learning_rate=lr, n_steps=2**exponent_n_steps, gamma=gamma, gae_lambda=gae_lambda,
            ent_coef=ent_coef, max_grad_norm=max_grad_norm, policy_kwargs=policy_kwargs,
            verbose=1, tensorboard_log=log_path)

# train the model
model.learn(total_timesteps=10000000, callback=eval_callback)

PPO_Path = os.path.join('Training', 'clusterResults', 'clusterSavedModels', 'PPO_21_10M_15dgr_cylinder')

model.save(PPO_Path)
