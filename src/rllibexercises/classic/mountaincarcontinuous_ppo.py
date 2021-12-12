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


ENV_NAME = "MountainCarContinuous-v0"


def main():
    start_time = datetime.now()

# "CQL"                  ## AssertionError
# "Apex"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-1.], [1.], (1,), float32) is not supported for DQN.
# "DQN"                  ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-1.], [1.], (1,), float32) is not supported for DQN.
# "R2D2"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-1.], [1.], (1,), float32) is not supported for DQN.
# "SimpleQ"              ## ray.rllib.utils.error.UnsupportedSpaceException: Action space Box([-1.], [1.], (1,), float32) is not supported for DQN.
# "MAML"                 ## AttributeError: 'Continuous_MountainCarEnv' object has no attribute 'sample_tasks'
# "MBMPO"                ## ValueError: Env <TimeLimit<MountainCarEnv<MountainCar-v0>>> doest not have a `reward()` method, needed for MB-MPO!
# "QMix"                 ## ValueError: Obs space must be a Tuple, got Box([-1.2  -0.07], [0.6  0.07], (2,), float32). Use MultiAgentEnv.with_agent_groups() to group related agents for QMix.
# "SAC"                  ## AssertionError

    ### works
    ## MARWIL                             episode takes   0.5 sec (mean does not learn)
    ## A2C                                episode takes  10 sec
    ## A3C                                episode takes   5 sec
    ## Impala                             episode takes  10 sec
    ## PPO                                episode takes   4 sec
    ## APPO                               episode takes  10 sec 
    ## TD3                                episode takes ~21 sec
    ## DDPG                               episode takes  ~1 min
    ## ApexDDPG                           episode takes  ~1 min
    
    MARWIL_config = {
        "num_gpus": 0,
        'lr': 0.1,
        'rollout_fragment_length': 5000,
#         'train_batch_size': 1
#         'rollout_fragment_length': 1,
#         'train_batch_size': 1,
    }
    custom_configs = { "MARWIL": MARWIL_config }
    
    best_alg = "PPO"
    best_seed = None
    best_iters = 100
    best_layers = [ 16, 4 ]
#     best_layers = [ 32, 8 ]
#    best_layers = [ 128, 32 ]
#     best_layers = [ 512, 512, 512 ]
    best_learning = 0.02

    specific_config = custom_configs.get( best_alg, {} ).copy()
    specific_config['lr'] = best_learning
    param = "lr: {:.5f}".format( best_learning )
    
    trainer.learn( ENV_NAME, best_alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed, specific_config=specific_config, custom_params=param )

# #     layers = [ [8], [16], [32], [48], [64], [96], [128], [192], [256] ]
#     layers = [ [16, 4], [32, 8], [48, 12], [64, 16], [96, 24], [128, 32], [192, 48], [256, 64] ]
#     for net in layers:
#         trainer.learn( ENV_NAME, best_alg, layers_size=net, n_iter=best_iters, seed=best_seed, 
#                        specific_config=specific_config, custom_params=param )

    ## for PPO
    #learning_rate = [ 0.001, 0.006, 0.01, 0.06, 0.1 ]
    #learning_rate = [ 0.002, 0.007, 0.02, 0.07, 0.2 ]
    #learning_rate = [ 0.07, 0.2 ]
    
#     specific_config = custom_configs.get( best_alg, {} )
# #     learning_rate = [ 0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1.0, 6.0, 11.0 ]
#     learning_rate = [ 0.6, 1.0, 6.0, 11.0 ]
#     for learning in learning_rate:
#         specific_config['lr'] = learning
#         param = "lr: {:.5f}".format( learning )
#         trainer.learn( ENV_NAME, best_alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed, 
#                        specific_config=specific_config, custom_params=param )

#     layers = [ [8], [16], [32], [48], [64], [96], [128], [192], [256] ]
# #     layers = [ [16, 4], [32, 8], [48, 12], [64, 16], [96, 24], [128, 32], [192, 48], [256, 64] ]
#     for net in layers:
#         trainer.learn( ENV_NAME, "Impala", layers_size=net, n_iter=best_iters, seed=best_seed )
    
    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
