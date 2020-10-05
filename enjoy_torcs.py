import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from gym_torcs_multi import TorcsEnv

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det
device = torch.device("cuda:0")

args.env_name = "Torcs-v1"
args.load_dir = "saved/ppo/"
args.algo = 'ppo'

print(os.path.join(args.load_dir, args.env_name + ".pt"))

# We need to use the same statistics for normalization as used in training

actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + "_new_mp.pt"))
actor_critic = actor_critic[0]

# vec_norm = get_vec_normalize(env)
# if vec_norm is not None:
#     vec_norm.eval()
#     vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

env = TorcsEnv(0, None, throttle=True, gear_change=False)
obs = env.reset()
obs = torch.from_numpy(obs)

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    obs = torch.from_numpy(obs).float().to(device)

    masks.fill_(0.0 if done else 1.0)

    if done:
        obs = envs.reset()
        obs = torch.from_numpy(obs)

