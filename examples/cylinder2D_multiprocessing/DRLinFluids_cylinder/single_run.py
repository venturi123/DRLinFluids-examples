
import socket
import os
import numpy as np
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import envobject
import sys
import os
import re
import envobject

# from simulation_base.env import resume_env, nb_actuations
# from RemoteEnvironmentClient import RemoteEnvironmentClient

# ap = argparse.ArgumentParser()
# ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
# ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
# ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)
#
# args = vars(ap.parse_args())
# number_servers = args["number_servers"]
# ports_start = args["ports_start"]
# host = args["host"]

number_servers=1
nb_actuations = 400 #Nombre d'actuations du reseau de neurones par episode
nstate=1
naction=1

# define parameters
foam_params = {
    'delta_t': 0.0005,
    'solver': 'pimpleFoam',
    'num_processor': 4,
    'of_env_init': 'source ~/OpenFOAM/OpenFOAM-8/etc/bashrc',
    'cfd_init_time': 0.034,  # 初始化流场，初始化state
    'num_dimension': 2,
    'verbose': False
}

entry_dict_q0 = {
    'U': {
        'JET1': {
            'q0': '{x}',
        },
        'JET2': {
            'q0': '{-x}',
        }
    }
}

entry_dict_q1 = {
    'U': {
        'JET1': {
            'q1': '{y}',
        },
        'JET2': {
            'q1': '{-y}',
        }
    }
}

entry_dict_t0 = {
    'U': {
        'JET1': {
            't0': '{t}'
        },
        'JET2': {
            't0': '{t}'
        }
    }
}

agent_params = {
    'entry_dict_q0': entry_dict_q0,
    'entry_dict_q1': entry_dict_q1,
    'entry_dict_t0': entry_dict_t0,
    'deltaA': 0.05,
    'minmax_value': (-1.5, 1.5),
    'interaction_period': 0.025,
    'purgeWrite_numbers': 0,
    'writeInterval': 0.025,
    'deltaT': 0.0005,
    'variables_q0': ('x',),
    'variables_q1': ('y',),
    'variables_t0': ('t',),
    'verbose': False,
    "zero_net_Qs": True,
}

state_params = {
    'type': 'velocity'
}
print("Create CFD env")

# \u83b7\u53d6\u5de5\u4f5c\u76ee\u5f55\u8def\u5f84
root_path = os.getcwd()
# \u83b7\u53d6Environment\u6587\u4ef6\u5939\u540d\u79f0\uff0c\u5e76\u6309\u7167\u5347\u5e8f\u6392\u5217\uff0croot_path + env_path_list\u5c31\u80fd\u83b7\u53d6\u6bcf\u4e00\u4e2a\u73af\u5883\u6587\u4ef6\u5939\u7684\u7edd\u5bf9\u8def\u5f84
env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
# \u65b0\u5efa\u4e00\u4e2aenvironment\u5bf9\u8c61\u5217\u8868
environments = []
for env_name in env_name_list:
    env = envobject.FlowAroundSquareCylinder2D(
        foam_root_path='/'.join([root_path, env_name]),
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    environments.append(env)

deterministic = True

print("define network specs")
network_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512)
]
baseline_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512)
]
print(env.states())
print(env.actions())
print(network_spec)
saver_restore = dict(directory=os.getcwd() + "/saved_models",frequency=1)
print("define agent")
# agent = Agent.create(
#     states=env.states(),
#     actions=env.actions(),
#     max_episode_timesteps=400,
#     agent='ppo',
#     environment=env,
#     batch_size=20,
#      network=network_spec,
#     learning_rate=0.001,state_preprocessing=None,
#     entropy_regularization=0.01, likelihood_ratio_clipping=0.2,
#     subsampling_fraction=0.2,
#     predict_terminal_values=True,
#     # baseline=dict(type='1', size=[32, 32]),
#     baseline=baseline_spec,
#     baseline_optimizer=dict(
#         type='multi_step',
#         optimizer=dict(
#             type='adam',
#             learning_rate=1e-3
#         ),
#         num_steps=5
#     ),
#     multi_step=25,
#     parallel_interactions=1,
#     saver=saver_restore,
# )
# agent.initialize()
# print(str(os.getcwd() + "/saved_models"),'/'.join([root_path, '/saved_models/checkpoint']))
agent = Agent.load(directory=str(os.getcwd() + "/best_model"),
                                 format='checkpoint',
                                 environment=env
                   )

def one_run():
    print("start simulation")
    states = env.reset()
    # env.render = True

    for k in range(400):
        #environment.print_state()
        internals = agent.initial_internals()
        actions,internals  = agent.act(states=states,internals =internals , deterministic=deterministic, independent=True)
        print(actions)
        states, terminal, reward = env.execute(actions=actions)

    env.single_run_write()

if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()
