from pogema_toolbox import fix_num_threads_issue

import json
from pathlib import Path

import time

import numpy as np

from pogema_toolbox.config_variant_generator import generate_variants
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry

from pogema_toolbox.views.view_multi_plot import process_multi_plot_view, MultiPlotView
from pogema_toolbox.views.view_plot import process_plot_view, PlotView
from pogema_toolbox.views.view_tabular import process_table_view, TabularView
from pogema_toolbox.results_holder import ResultsHolder

from concurrent.futures import ProcessPoolExecutor

def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        actions = algo.act(obs)
        obs, rew, terminated, truncated, infos = env.step(actions)
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break
    return results_holder.get_final()


def get_num_of_available_cpus():
    """
    Returns the number of available CPUs.

    Returns:
        int: Number of available CPUs.
    """
    import multiprocessing
    return multiprocessing.cpu_count()

def get_env_config_cost(raw_config):
    gc = Environment(**raw_config)
    return gc.num_agents * gc.max_episode_steps


def get_balanced_buckets_indexes(env_configs, num_buckets):
    """
    Distributes environment indexes into balanced buckets based on their costs.

    Args:
        env_configs: List of environment configurations.
        num_buckets: Number of buckets to distribute the indexes into.

    Returns:
        List[List[int]]: Balanced buckets containing environment indexes.
    """
    buckets = [[] for _ in range(num_buckets)]
    bucket_costs = [0 for _ in range(num_buckets)]
    env_costs = [get_env_config_cost(ec) for ec in env_configs]
    indexes = np.argsort(env_costs)[::-1]

    for idx in indexes:
        min_bucket_idx = np.argmin(bucket_costs)
        buckets[min_bucket_idx].append(idx)
        bucket_costs[min_bucket_idx] += env_costs[idx]

    # remove empty buckets
    buckets = [bucket for bucket in buckets if len(bucket) > 0]

    return buckets

def sequential_backend(algo_configs, env_configs, full_algo_name, registry_state=None):
    """
    Runs the algorithm sequentially on multiple environments.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.
        registry_state:

    Returns:
        List: Results of running the algorithm on the environments.
    """
    registry = ToolboxRegistry
    if registry_state is not None:
        registry.recreate_from_state(registry_state)

    results = []
    for idx, env_config in enumerate(env_configs):
        algo_name = algo_configs[idx]['name']
        algo = registry.create_algorithm(algo_name, **algo_configs[idx])
        ToolboxRegistry.info(f'Running: {full_algo_name} [{idx + 1}/{len(env_configs)}]')
        env = registry.create_env(env_config['name'], **env_config)
        results.append(run_episode(env, algo))

        if env_config.get('with_animation', None):
            from pathlib import Path

            directory = Path(f'renders/{full_algo_name}/')
            name = env.pick_name(env.grid_config)

            directory.mkdir(parents=True, exist_ok=True)
            ToolboxRegistry.debug(f'Saving animation to "{directory / name}"')
            env.save_animation(name=directory / name)
    return results





def balanced_dask_backend(algo_configs, env_configs, full_algo_name):
    """
    Runs the algorithm in a balanced manner using Dask for distributed computing.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.

    Returns:
        List: Results of running the algorithm on the environments.
    """
    ToolboxRegistry.debug('Running experiment with balanced task backend')
    import dask
    import dask.distributed as dd
    from dask.config import set as dask_set



    initialized_algo_config = ToolboxRegistry.create_algorithm_config(algo_configs[0]['name'], **algo_configs[0])

    num_process = min(initialized_algo_config.num_process, get_num_of_available_cpus())
    num_process = 12
    balanced_buckets = get_balanced_buckets_indexes(env_configs, num_process)


    cluster = dd.LocalCluster(n_workers=num_process, threads_per_worker=1, nthreads=1)
    client = dd.Client(cluster, timeout="120s")  # Connect the client to the cluster

    futures = []

    ToolboxRegistry.get_maps()
    registry_state = ToolboxRegistry.get_state()
    for bucket in balanced_buckets:
        bucket_configs = [env_configs[idx] for idx in bucket]
        algo_bucket_configs = [algo_configs[idx] for idx in bucket]

        future = client.submit(
            sequential_backend,
            algo_bucket_configs,
            bucket_configs,
            full_algo_name,
            registry_state,
            pure=False
        )
        futures.append(future)

    results = client.gather(futures)
    client.close()
    cluster.close()


    # with dask_set({
    #     "distributed.comm.timeouts.connect": "20s",
    #     "distributed.comm.timeouts.tcp": "120s",
    #     "distributed.comm.timeouts.handshake": "10s",
    #     "distributed.comm.timeouts.heartbeat": "60s",
    #     "distributed.scheduler.allowed-failures": 10,
    #     "distributed.comm.retry.count": 15,
    #     "distributed.comm.retry.delay.min": "1s",
    #     "distributed.comm.retry.delay.max": "10s",
    # }):
    #     with dd.LocalCluster(
    #         n_workers=num_process,
    #         threads_per_worker=1,
    #         memory_limit="auto",
    #         processes=True,
    #         dashboard_address=None
    #     ) as cluster:
            
    #         with dd.Client(cluster, timeout="180s") as client:
                
    #             ToolboxRegistry.get_maps()
    #             registry_state = ToolboxRegistry.get_state()

    #             futures = []
    #             for bucket in balanced_buckets:
    #                 bucket_configs = [env_configs[idx] for idx in bucket]
    #                 algo_bucket_configs = [algo_configs[idx] for idx in bucket]

    #                 future = client.submit(
    #                     sequential_backend,
    #                     algo_bucket_configs,
    #                     bucket_configs,
    #                     full_algo_name,
    #                     registry_state,
    #                     pure=False
    #                 )
    #                 futures.append(future)

    #             # Wait for all tasks to finish
    #             results = client.gather(futures)
    #             dd.wait(futures)


    # Reorder the results according to the original order of env_configs
    ordered_results = [None for _ in range(len(env_configs))]
    for idx, bucket in enumerate(balanced_buckets):
        bucket_results = results[idx]
        for i, env_idx in enumerate(bucket):
            ordered_results[env_idx] = bucket_results[i]

    return ordered_results


def join_metrics_and_configs(metrics, evaluation_configs, env_grid_search, algo_configs, algo_name):
    """
    Joins metrics, evaluation configurations, environment grid search, and algorithm name into a result dictionary.

    Args:
        metrics: List of metrics.
        evaluation_configs: List of evaluation configurations.
        env_grid_search: List of environment grid search configurations.
        algo_config: Configuration for the algorithm.
        algo_name: Name of the algorithm.

    Returns:
        List[dict]: List of result dictionaries.
    """
    env_grid_search = [{key[-1]: value for key, value in x.items()} for x in env_grid_search]
    results = []
    for idx, metric in enumerate(metrics):
        results.append({'metrics': metrics[idx], 'env_grid_search': env_grid_search[idx], 'algorithm': algo_name})
    return results



def evaluation(environment_configs,env_grid_search, algo_configs, key , eval_dir=None):
    ToolboxRegistry.create_algorithm_config(algo_configs[0]['name'], **algo_configs[0])
    ToolboxRegistry.info(f'Starting: omggggg')
    start_time = time.monotonic()
    results = []
    metrics = balanced_dask_backend( algo_configs, environment_configs, key)
    algo_results = join_metrics_and_configs(metrics, environment_configs, env_grid_search,  algo_configs, key)
    if eval_dir:
        save_path = Path(eval_dir) / f'{key}.json'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(algo_results, f)
    results += algo_results
    ToolboxRegistry.success(f'Finished: {key}, runtime: {time.monotonic() - start_time}')


