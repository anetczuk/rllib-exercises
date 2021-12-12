#!/usr/bin/env python3

try:
    ## following import success only when file is directly executed from command line
    ## otherwise will throw exception when executing as parameter for "python -m"
    # pylint: disable=W0611
    import __init__
except ImportError as error:
    ## when import fails then it means that the script was executed indirectly
    ## in this case __init__ is already loaded
    pass

import os
from datetime import datetime

import matplotlib.pyplot as pyplot
# import plot

from rllibexercises import trainer


## disable import warning
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


ENV_NAME = "Pendulum-v1"


def main():
    start_time = datetime.now()

# "CQL"                  ## AssertionError
# "Apex"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-2.], [2.], (1,), float32) is not supported for DQN.
# "DQN"                  ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-2.], [2.], (1,), float32) is not supported for DQN.
# "R2D2"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-2.], [2.], (1,), float32) is not supported for DQN.
# "SimpleQ"              ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-2.], [2.], (1,), float32) is not supported for DQN.
# "MAML"                 ## AttributeError: 'PendulumEnv' object has no attribute 'sample_tasks'
# "MBMPO"                ## AttributeError: 'NoneType' object has no attribute 'cuda'
# "QMix"                 ## ValueError: Obs space must be a Tuple, got Box([-1. -1. -8.], [1. 1. 8.], (3,), float32). Use MultiAgentEnv.with_agent_groups() to group related agents for QMix.
# "SAC"                  ## AssertionError
    
    ## working, but without good results
    ## "A2C", "A3C", "ApexDDPG", "Apex", "Impala", "BC", "MARWIL", "PG", "PPO", "APPO"
    
    ## working with good results
    ## "TD3", "DDPG"
    
    best_seed = None
    best_iters = 200
    best_layers = [ 16 ]

    trainer.learn( ENV_NAME, "TD3", layers_size=best_layers, n_iter=best_iters, seed=best_seed )

#    succeed_algs = [ "TD3", "DDPG" ]
#     for alg in works:
#         learn( alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed )
    
    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


# import ray
# from ray.rllib.agents.ppo import PPOTrainer
# 
# 
# def main():
#     config = {
#         "env": "CartPole-v0",
#         "framework": "tf2",
#         "model": {
#           "fcnet_hiddens": [32],
#           "fcnet_activation": "linear",
#         },
#     }
#     stop = {"episode_reward_mean": 195}
#     ray.shutdown()
#     ray.init(
#       num_cpus=3,
#       include_dashboard=False,
#       ignore_reinit_error=True,
#       log_to_driver=False,
#     )
#     
#     # execute training 
#     analysis = ray.tune.run(
#       "PPO",
#       config=config,
#       stop=stop,
#       checkpoint_at_end=True,
#     )


if __name__ == '__main__':
    main()
