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


ENV_NAME = "Pendulum-v1"


def int_hex_dec(x):
    return int(x, 0)


def main():
    parser = argparse.ArgumentParser(description=ENV_NAME+' solution')
    parser.add_argument( '--seed', action="store", type=int_hex_dec, default=None, help='RNG seed (dec or hex)' )
    
    args = parser.parse_args()
    
    ## preparing and training
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
    
    specific_config = {
        'batch_mode': 'complete_episodes',
    }
    
    best_seed = args.seed
    best_iters = 50
    best_layers = [ 16 ]
#     best_layers = [ 16 ]
    best_learning = 0.0001            ## default is 0.0001
    metrics_smooth_size=100
    metrics_stop_condition = None
#     metrics_stop_condition = {
#         'limit': -120.0,
#         'metrics': 'avg'
#     }
#     custom_params = ""

    specific_config['lr'] = best_learning
    custom_params = "lr: {:.5f}".format( best_learning )
    
    trainer.learn( ENV_NAME, "TD3", layers_size=best_layers, 
                   n_iter=best_iters, metrics_stop_condition=metrics_stop_condition, metrics_smooth_size=metrics_smooth_size,
                   seed=best_seed, draw_interval=1, custom_params=custom_params )

#    succeed_algs = [ "TD3", "DDPG" ]
#     for alg in works:
#         learn( alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed )
    
    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
