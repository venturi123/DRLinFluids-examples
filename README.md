# DRLinFluids-examples 

Here is the repository of two DRL cases mentioned in the [article](https://aip.scitation.org/doi/10.1063/5.0103113) with the **primitive** DRLinFluids package. We provide this repository for quick understanding and testing, and the DRLinFluids package is now available on [This page](https://github.com/venturi123/DRLinFluids).

> **Note**  
> This repository is for backup only. We highly suggest running the experiments in a docker or singularity container. A short introduction to creating containers, exporting them, and using them in your own **Linux** device is described as follows.

## Using through Docker container (recommended)

```bash
 docker pull dolores1900/drlinfluids:v2
 docker image ls
 docker run -itd -u 98765 -v /root:/root new_drlinfluids:latest /bin/bash  # Run Docker as a non-root user
 docker ps 
 docker exec -it $(container_id) /bin/bash    # Run Docker as a non-root user
 #cylinder_training
 cd DRLinfluids/cylinder2D_multiprocessing
 python DRLinFluids_cylinder/launch_multiprocessing_traning_cylinder.py
 #square_training
 cd DRLinfluids/square2D_multiprocessing
 python DRLinFluids_square/launch_multiprocessing_traning_square.py
```

## Using through Singularity container (recommended)

```bash
 # must run singularity as a non-root user
 sudo singularity build DRLinfluids.sif docker://dolores1900/drlinfluids:v2
 singularity build --sandbox DRLinfluids_sandbox/ DRLinfluids.sif
 singularity shell -w DRLinfluids_sandbox/ 
 # Setting in OpenFOAMv8 in newbc
 source /opt/openfoam8/etc/bashrc    #OpenFOAM compile boundary conditions
 cd DRLinfluids/newbc
 wclean
 wmake
 # cylinder_training
 cd DRLinfluids/cylinder2D_multiprocessing
 python DRLinFluids_cylinder/launch_multiprocessing_traning_cylinder.py
 #square_training
 cd DRLinfluids/square2D_multiprocessing
 python DRLinFluids_square/launch_multiprocessing_traning_square.py
```

## Installing by hand (discouraged)

> **Note**  
> Before launching any script, check that you have installed completed dependencies. (tested on Ubuntu 20.04)

```
numpy==1.19.5
pandas==1.3.3
PeakUtils==1.3.3
scipy==1.7.1
Tensorforce==0.6.5
```

### Step 1: Compiling new boundary conditions in OpenFOAM v8

The *wmake* script is executed by typing:

```
# Setting proper OpenFOAM environment variables in Linux (according to actual situation)
source ~/OpenFOAM/OpenFOAM-8/etc/bashrc

cd DRLinFluids-examples/newbc/jetParabolicVelocity
wmake
```

For more details, see https://doc.cfd.direct/openfoam/user-guide-v8/compiling-applications#dx10-78001

### Step 2: Running Python scripts

#### QuickStart example code in cylinder2D_multiprocessing

```python
from tensorforce import Runner, Agent,Environment

# Pre-defined or custom environment
root_path = os.getcwd()
env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
environments = []
for env_name in env_name_list:
    env = envobject_cylinder.FlowAroundCylinder2D(
        foam_root_path='/'.join([root_path, env_name]),
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    environments.append(env)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='ppo',
    environment=env,max_episode_timesteps=shell_args['max_episode_timesteps'],
    batch_size=20,
     network=network_spec,
    learning_rate=0.001,state_preprocessing=None,
    entropy_regularization=0.01, likelihood_ratio_clipping=0.2, subsampling_fraction=0.2,
    predict_terminal_values=True,
    discount=0.97,
    baseline=baseline_spec,
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    multi_step=25,
    parallel_interactions=number_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saved_models/checkpoint'),frequency=1  
    ),
    summarizer=dict(
        directory='summary',
        # list of labels, or 'all'
        labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
    ),
)

# Train for set episodes
runner = Runner(
    agent=agent,
    environments=environments,
    max_episode_timesteps=shell_args['max_episode_timesteps'],
    evaluation=use_best_model,
    remote='multiprocessing',
)
runner.run(num_episodes=shell_args['num_episodes'],
           save_best_agent ='best_model',
           )

agent.close()
environment.close()
```

#### QuickStart example code in square2D_multiprocessing

```python
# Pre-defined or custom environment
train_envs = SubprocVectorEnv(
        [lambda x=i: gym.make(args.task,foam_root_path=x,
                              foam_params=foam_params,
                              agent_params=agent_params,
                              state_params=state_params,
                              ) for i in env_path_list[0:args.training_num]],
        wait_num=args.training_num, timeout=0.2
    )

# Instantiate a Tianshou policy
policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

# Train for set episodes
result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        resume_from_log=args.resume,
    )
```

#### Command line usage in cylinder2D_multiprocessing

```bash
python DRLinFluids_cylinder/launch_multiprocessing_traning_cylinder
```

#### Command line usage in square2D_multiprocessing

```bash
python DRLinFluids_square/launch_multiprocessing_traning_square
````

