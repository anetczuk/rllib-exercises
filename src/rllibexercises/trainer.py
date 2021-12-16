##
##
##

import os
from datetime import datetime
import pprint

import numpy
import pandas as pd

import importlib
import inspect

import matplotlib.pyplot as pyplot
# import plot


SCRIPT_DIR      = os.path.abspath( os.path.dirname(__file__) )
TMP_DIR         = os.path.abspath( os.path.join( SCRIPT_DIR, "..", "..", "tmp" ) )


# "ARS"                  ## invalid results
# "DREAMER"              ## ValueError: Distributed Dreamer not supported yet!
# "ES"                   ## invalid results
# "DDPPO"                ## WARNING worker.py:1215 -- The actor or task with ID {} cannot be scheduled right now.
# "RNNSAC"               ## AssertionError: RNNSAC requires its model to be a recurrent one!
# "SlateQ"               ## AttributeError: 'NoneType' object has no attribute 'cuda'


ALGORITHMS_CONFIG = [
    ( "A2C", "a3c" ),
    ( "A3C", ),
    ( "ARS", ),
    ( "CQL", ),
    ( "DDPG", ),
    ( "ApexDDPG", "ddpg" ),
    ( "TD3", "ddpg" ),
    ( "Apex", "dqn" ),
    ( "DQN", ),
    ( "R2D2", "dqn" ),
    ( "SimpleQ", "dqn" ),
    ( "DREAMER", ),
    ( "ES", ),
    ( "Impala", ),
    ( "MAML", ),
    ( "BC", "marwil" ),
    ( "MARWIL", ),
    ( "MBMPO", ),
    ( "PG", ),
    ( "PPO", ),
    ( "APPO", "ppo" ),
    ( "DDPPO", "ppo" ),
    ( "QMix", ),
    ( "SAC", ),
    ( "RNNSAC", "sac" ),
    ( "SlateQ", )
]


ALGORITHMS_FAILING = [ "ARS", "DREAMER", "ES", "DDPPO", "RNNSAC", "SlateQ" ]
ALGORITHMS_WORKING = [ "A2C", "A3C", "CQL", "DDPG", "ApexDDPG", "TD3", "Apex",
                       "DQN", "R2D2", "SimpleQ", "Impala", "MAML", "BC", 
                       "MARWIL", "MBMPO", "PG", "PPO", "APPO", "QMix", "SAC" ]

ALGORITHMS_ALL = ALGORITHMS_FAILING + ALGORITHMS_WORKING


def find_algorithm_entry( algorithm ):
    for item in ALGORITHMS_CONFIG:
        if item[0] == algorithm:
#             print( "found algorithm:", item )
            return item
    return None


def load_algorithm( algorithm, module_name=None ):
    item = find_algorithm_entry( algorithm )
    if item is None:
        return None
    algorithm = item[0]
    if len(item) > 1:
        module_name = item[1]
    if module_name is None:
        module_name = algorithm.lower()
        
    mod = "ray.rllib.agents." + module_name
    import_module = importlib.import_module( mod )

    trainer_name = algorithm + "Trainer"
#     def_config  = getattr( import_module, 'DEFAULT_CONFIG' )
    mod_trainer = getattr( import_module, trainer_name )
#     def_config = mod_trainer._default_config
#     return ( mod_trainer, def_config )
    return mod_trainer


def list_trainers():
    class_list = []
    mod = "ray.rllib.agents"
    import_module = importlib.import_module( mod )
    submodules = inspect.getmembers( import_module, inspect.ismodule )
    for mod_item in submodules:
        submodule = mod_item[1]
        mod_file = os.path.basename( submodule.__file__ )
        if mod_file != "__init__.py":
            continue
        ## algorithm candidates
        classes = inspect.getmembers( submodule, inspect.isclass )
        for class_item in classes:
            class_name = class_item[0]
            if class_name.endswith( "Trainer" ):
                ## print( "found trainer:", class_name, class_item )
                class_list.append( class_item[1] )
    return class_list


## =================================================================


def calculate_reward_stats( n, result ):
#     iter_rewards_num = result['episodes_this_iter']
#      
#     ### it seems that statistic results (e.g. result['episode_reward_min']) are not accurate
#     ### and spans results from multiple training iterations
#     iter_rewards = result['hist_stats']['episode_reward'][-iter_rewards_num:]
#      
# #         print("rewards:", iter_rewards_num, iter_rewards )
#     rewards_min  = min( iter_rewards )
#     rewards_max  = max( iter_rewards )
#     rewards_mean = numpy.mean( iter_rewards )
     
    ## RL way -- based on 'metrics_smoothing_episodes' config value   
    rewards_min  = result['episode_reward_min']
    rewards_max  = result['episode_reward_max']
    rewards_mean = result['episode_reward_mean']
    
    episode = {'iter': n, 
               'episode_reward_min':  rewards_min, 
               'episode_reward_mean': rewards_mean, 
               'episode_reward_max':  rewards_max
               }
    
    return episode


def print_results( result ):
    train_result = result.copy()
#         del result['config']
    train_result['config'] = None
    print( "" )
    pprint.pprint( train_result )
#         results.append( result )


def concat_params( algorithm, layers, customParams, itersNumber, seed ):
    params = "alg: {} layers: {}".format( algorithm, layers )
    if customParams is not None and len( customParams ) > 0:
        params += " " + customParams
    params += " iters: {:05d} seed: 0x{:X}".format( itersNumber, seed )
    return params


def draw_plot( plot_fig, subplot, episode_data, envName, params ):
    if len(episode_data) < 2:
        return
    df = pd.DataFrame( data=episode_data )
    subplot.clear()
    subplot.locator_params( integer=True )
#             df.plot( ax=subplot, title=envName + ":\n" + params, x="iter", y=["episode_reward"] )
    df.plot( ax=subplot, title=envName + ":\n" + params, x="iter", y=["episode_reward_min", "episode_reward_mean", "episode_reward_max"] )
#             plot_fig.canvas.draw()
    plot_fig.canvas.flush_events()
#             plot.process_events( 0.1 )


# n_iter -- number of training runs.
def learn( envName, algorithm, layers_size, n_iter=32, metrics_stop_condition=None, seed=None, framework="tf", specific_config=None, custom_params="",
           metrics_smooth_size=None ):
    start_time = datetime.now()
    
    checkpoint_root = os.path.abspath( os.path.join( TMP_DIR, "run", envName ) )
    checkpoint_video = os.path.abspath( os.path.join( checkpoint_root, "video" ) )
    checkpoint_model = os.path.abspath( os.path.join( checkpoint_root, "model" ) )
    
    os.makedirs( checkpoint_root, exist_ok=True )
    os.makedirs( checkpoint_video, exist_ok=True )
    os.makedirs( checkpoint_model, exist_ok=True )
    
    AlgTrainer = load_algorithm( algorithm )
#     pprint.pprint( dir(AlgTrainer) )
    
    print( "running algorithm:", algorithm )

    if seed is None:
        seed = numpy.random.randint( 0, 2**32 - 1 )
    print( "random seed:", seed )
    
    if metrics_smooth_size is None or metrics_smooth_size < 1:
        metrics_smooth_size = max( int(n_iter / 20), 1 )
    print( "metrics smoothing window size:", metrics_smooth_size )
    
    
#     config = DEFAULT_CONFIG.copy()
    config = AlgTrainer._default_config.copy()
    
    if algorithm == "SlateQ":
        framework = "torch"

    ## common parameters
    common_config = {
        "log_level": "WARN",                    # Suppress too many messages, but try "INFO" to see what can be printed.
#         "framework": "tf2",
        "framework": framework,
        "seed": seed,
        "num_workers": 1,                       # Use > 1 for using more CPU cores, including over a cluster
        "model": {
          "fcnet_hiddens": layers_size,
#           "fcnet_activation": "tanh",
        },
        'metrics_smoothing_episodes': metrics_smooth_size,
        "num_gpus": 0,
        "num_cpus_per_worker": 0,               # This avoids running out of resources in the notebook environment when this cell is re-executed
#        "train_batch_size": 100,
#         "rollout_fragment_length": 1
    }
    config.update( common_config )
    
    config['env'] = envName

    eval_config = {
        # Evaluate once per training iteration.
        "evaluation_interval": 9999999999,                      ## disable automatic evaluation and trigger it explicit
#         "evaluation_interval": n_iter,
        # Run evaluation on (at least) two episodes
        "evaluation_num_episodes": 2,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Store videos in this relative directory here inside
            # the default output dir (~/ray_results/...).
            # Alternatively, you can specify an absolute path.
            # Set to True for using the default output dir (~/ray_results/...).
            # Set to False for not recording anything.
            "record_env": checkpoint_video,
  
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
            "explore": False
        }
    }
    config.update( eval_config )

#     ## algorithm specific config
    if specific_config is not None:
        config.update( specific_config )

#     config["num_sgd_iter"] = 10                     # Number of SGD (stochastic gradient descent) iterations per training minibatch.
#                                                     # I.e., for each minibatch of data, do this many passes over it to train. 
#     config["sgd_minibatch_size"] = 250              # The amount of data records per minibatch

#     config["sgd_minibatch_size"] = 1
#     config["num_sgd_iter"] = 1

    pprint.pprint( config )
    
    params = ""
    
    print( "\ncreating agent" )
    agent = AlgTrainer( config=config, env=envName )
    
    episode_data = []

    pyplot.ion()
    ## ax is 'Line2D'
    plot_fig, ax = pyplot.subplots()
    subplot = ax.axes                   ## of type 'AxesSubplot'
    
    result = None
    print( "\ntraining:" )
    for n in range(1, n_iter+1):
        result = agent.train()
        
#         print_results( result )
        
#         print( "ooooooo:", result['timesteps_total'], result['timesteps_total'] )
#         print( "ooooooo:", len( result['hist_stats']['episode_lengths'] ) )
        
        if result['episodes_total'] is None:
            print( f'{n:3d}: invalid results (episodes_total is None)' )
            continue
        
        iter_rewards_num = result['episodes_this_iter']
        if iter_rewards_num < 1:
            print( f'{n:3d}: no results (episodes_this_iter < 1)' )
            continue
        
        episode = calculate_reward_stats( n, result )

        episode_data.append( episode )
#         file_name = agent.save( checkpoint_model )
        
        rewards_min  = episode['episode_reward_min']
        rewards_max  = episode['episode_reward_max']
        rewards_mean = episode['episode_reward_mean']

        elapsed_time = datetime.now() - start_time

        recent_reward = result['hist_stats']['episode_reward'][-1:][0]
        
        ep_done = result['done']
        
        ## agent_timesteps_total iterations_since_restore episodes_this_iter episodes_total done
        print( f'{n:3d}: Recent/Min/Mean/Max reward: {recent_reward:8.4f}/{rewards_min:8.4f}/{rewards_mean:8.4f}/{rewards_max:8.4f} episodes_this_iter: {iter_rewards_num} elapsed: {elapsed_time} done: {ep_done}' )
#         print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')

        params = concat_params( algorithm, layers_size, custom_params, n, seed )

        if (n % 10 == 0):
            draw_plot( plot_fig, subplot, episode_data, envName, params )
#             df = pd.DataFrame( data=episode_data )
#             subplot.clear()
#             subplot.locator_params( integer=True )
# #             df.plot( ax=subplot, title=envName + ":\n" + params, x="iter", y=["episode_reward"] )
#             df.plot( ax=subplot, title=envName + ":\n" + params, x="iter", y=["episode_reward_min", "episode_reward_mean", "episode_reward_max"] )
# #             plot_fig.canvas.draw()
#             plot_fig.canvas.flush_events()
# #             plot.process_events( 0.1 )

        if metrics_stop_condition is not None and rewards_min > metrics_stop_condition:
            break 

    ## redraw plot before evaluation
    draw_plot( plot_fig, subplot, episode_data, envName, params )

    agent.evaluate()

#     print_results( result )

    execution_time = datetime.now() - start_time

#     policy = agent.get_policy()
#     model = policy.model
#     
#     print( "\nvariables:" )
#     pprint.pprint( model.variables() )
#     
#     print( "\nvalue function:" )
#     pprint.pprint( model.value_function() )
#     
#     print( "\nmodel summary:" )
#     print( model.base_model.summary() )
    
    print( params )
    
    fileName = params
    fileName = fileName.replace( ", ", "_" )
    fileName = fileName.replace( ": ", "-" )
    fileName = fileName.replace( ":", "-" )
    fileName = fileName.replace( " ", "_" )
    
    agentDataPath = agent.save( checkpoint_model )
    print( "agent state stored to", agentDataPath )
    
    cfgOutput = os.path.abspath( os.path.join( checkpoint_root, fileName + ".cfg" ) )
    with open( cfgOutput, 'wt') as cfgOut:
        pprint.pprint( config, stream=cfgOut )
        
    print( "duration: {}\n".format( execution_time ) )
    
    figOutput = os.path.abspath( os.path.join( checkpoint_root, fileName + ".png" ) )
    plot_fig.savefig( figOutput )
