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
import argparse

import matplotlib.pyplot as pyplot
# import plot

from rllibexercises import trainer


## disable import warning
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


ENV_NAME = "MountainCarContinuous-v0"


def int_hex_dec(x):
    return int(x, 0)


def main():
    parser = argparse.ArgumentParser(description=ENV_NAME+' solution')
    parser.add_argument( '--seed', action="store", type=int_hex_dec, default=None, help='RNG seed (dec or hex)' )
    
    args = parser.parse_args()
    
    ## preparing and training
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
        'batch_mode': 'complete_episodes',
        "num_gpus": 0,
#         'rollout_fragment_length': 5000,
#         'train_batch_size': 1
#         'rollout_fragment_length': 1,
#         'train_batch_size': 1,
    }
    PPO_config = {
        'batch_mode': 'complete_episodes',
        "num_gpus": 0,
    }
    custom_configs = { "MARWIL": MARWIL_config, "PPO": PPO_config }
    
    best_alg = "PPO"
    best_seed = args.seed
    best_iters = 5000
    best_layers = [ 16, 4 ]
    best_learning = 0.02            ## default is 5e-5
    metrics_smooth_size=100
    metrics_stop_condition = {
        'limit': 90.0,
        'metrics': 'min'
    }
    custom_params = ""

    specific_config = custom_configs.get( best_alg, {} ).copy()
    specific_config['lr'] = best_learning
    custom_params = "lr: {:.5f}".format( best_learning )
    
    trainer.learn( ENV_NAME, best_alg, layers_size=best_layers, 
                   n_iter=best_iters, metrics_stop_condition=metrics_stop_condition, metrics_smooth_size=metrics_smooth_size, 
                   seed=best_seed, specific_config=specific_config, draw_interval=5, custom_params=custom_params )

    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
