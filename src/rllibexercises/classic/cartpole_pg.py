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


ENV_NAME = "CartPole-v1"


def int_hex_dec(x):
    return int(x, 0)


def main():
    parser = argparse.ArgumentParser(description=ENV_NAME+' solution')
    parser.add_argument( '--seed', action="store", type=int_hex_dec, default=None, help='RNG seed (dec or hex)' )
    
    args = parser.parse_args()
    
    ## preparing and training
    start_time = datetime.now()
    
    specific_config = {
        'batch_mode': 'complete_episodes',
#         'lr': 0.001,
#         'rollout_fragment_length': 500,
#         'train_batch_size': 200
    }
    
    best_seed = args.seed
    best_layers = [64, 32]
    best_iters = 5000
    best_learning = 0.0006
    metrics_smooth_size=100
    metrics_stop_condition = 490
#     custom_params = ""

#     trainer.learn( ENV_NAME, "PG", layers_size=best_layers, n_iter=best_iters, seed=best_seed, specific_config=specific_config )

    specific_config['lr'] = best_learning
    custom_params = "lr: {:.5f}".format( best_learning )
    trainer.learn( ENV_NAME, "PG", layers_size=best_layers, 
                   n_iter=best_iters, metrics_stop_condition=metrics_stop_condition, metrics_smooth_size=metrics_smooth_size,
                   seed=best_seed, specific_config=specific_config, custom_params=custom_params )
   
    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
