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

log_path = os.path.join('Training', 'Logs')

env = gymnasium.make('pushBox-v0')
env = DummyVecEnv([lambda: env])

# Add a callback to training stage for early stopping
stop_callback = StopTrainingOnRewardThreshold(reward_threshold = 100, verbose = 1)
eval_callback = EvalCallback(env, 
                            callback_on_new_best = stop_callback,
                            eval_freq = 10000, 
                            verbose = 1)

# Tuned hyperparameters from optuna optimizer OP-25 
gamma = 0.0011032955028127819
max_grad_norm = 0.789225841801385
gae_lambda = 0.0015956023234189407
exponent_n_steps = 7
lr = 6.500487369849844e-05
ent_coef = 0.001546407406983241
exponent_batch_size = 2
n_epochs = 28
use_sde = True

policy_kwargs = dict(ortho_init=True,
                     activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

model = PPO('MlpPolicy', env, learning_rate=lr, n_steps=2**exponent_n_steps, batch_size=2**exponent_batch_size,
            n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, max_grad_norm=max_grad_norm, 
            use_sde=use_sde, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=10000000, callback=eval_callback)

PPO_Path = os.path.join('Training', 'SavedModels', 'PPO_102_10M_15_dgr_cylinder_tuner')

model.save(PPO_Path)