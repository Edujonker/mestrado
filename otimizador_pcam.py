import argparse
import os
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import erfinv

import otimizador_impressao as oi

LOG_DIR = "log_pcam"
INPUT_DISTANCE_MATRIX = os.path.join("log", "distance_matrix.csv")
INPUT_ROUTE_FILE = os.path.join("log", "route_nn.csv")

PCAM_CONFIG = {
    "mc_samples_per_worker": 50000,
    "mc_workers": min(4, mp.cpu_count()),
    "sa_runs": 12,
    "ga_runs": 12,
    "sa_iterations": oi.SA_PARAMS['iterations'],
    "ga_generations": oi.GA_PARAMS['generations'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa o paralelismo PCAM com Monte Carlo, SA e GA."
    )
    parser.add_argument("--log-dir", default=LOG_DIR, help="Diretório de saída dos logs PCAM")
    parser.add_argument("--distance-matrix", default=INPUT_DISTANCE_MATRIX, help="Arquivo CSV da matriz de distância gerado pelo otimizador principal")
    parser.add_argument("--mc-samples-per-worker", type=int, default=PCAM_CONFIG['mc_samples_per_worker'], help="Número de amostras Monte Carlo por worker")
    parser.add_argument("--mc-workers", type=int, default=PCAM_CONFIG['mc_workers'], help="Número de workers Monte Carlo")
    parser.add_argument("--sa-runs", type=int, default=PCAM_CONFIG['sa_runs'], help="Número de execuções de Simulated Annealing")
    parser.add_argument("--ga-runs", type=int, default=PCAM_CONFIG['ga_runs'], help="Número de execuções de Genetic Algorithm")
    parser.add_argument("--sa-iterations", type=int, default=PCAM_CONFIG['sa_iterations'], help="Número de iterações do Simulated Annealing")
    parser.add_argument("--ga-generations", type=int, default=PCAM_CONFIG['ga_generations'], help="Número de gerações do Genetic Algorithm")
    parser.add_argument("--population-size", type=int, default=oi.GA_PARAMS['population_size'], help="Tamanho da população do GA")
    parser.add_argument("--elite-size", type=int, default=oi.GA_PARAMS['elite_size'], help="Tamanho da elite do GA")
    parser.add_argument("--mutation-rate", type=float, default=oi.GA_PARAMS['mutation_rate'], help="Taxa de mutação do GA")
    return parser.parse_args()


def ensure_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)


def load_distance_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de matriz de distância não encontrado: {path}")
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float) / 111.12


def monte_carlo_chunk(worker_id, seed, distance_matrix, samples):
    rng = np.random.default_rng(seed)
    n = len(distance_matrix)
    costs = []
    base_route = np.arange(n)

    for _ in range(samples):
        route = np.concatenate(([0], rng.permutation(np.arange(1, n))))
        cost = oi.route_distance(route.tolist(), distance_matrix) * 111.12
        costs.append(cost)

    costs = np.array(costs, dtype=float)
    return {
        "worker_id": worker_id,
        "seed": seed,
        "samples": samples,
        "mean_km": float(costs.mean()),
        "std_km": float(costs.std(ddof=1)),
        "min_km": float(costs.min()),
        "max_km": float(costs.max()),
        "median_km": float(np.median(costs)),
        "skew_km": float(((costs - costs.mean())**3).mean() / costs.std(ddof=1)**3) if costs.std(ddof=1) > 0 else 0.0,
        "kurt_km": float(((costs - costs.mean())**4).mean() / costs.std(ddof=1)**4) if costs.std(ddof=1) > 0 else 0.0,
    }


def sa_worker(seed, distance_matrix, initial_route, iterations, temperature, cooling_rate):
    _, final_cost, _, best_cost = oi.simulated_annealing(
        distance_matrix=distance_matrix,
        initial_route=initial_route,
        seed=seed,
        iterations=iterations,
        initial_temperature=temperature,
        cooling_rate=cooling_rate,
    )
    return {
        "seed": seed,
        "final_cost_km": float(final_cost * 111.12),
        "best_cost_km": float(best_cost * 111.12),
    }


def ga_worker(seed, distance_matrix, population_size, elite_size, mutation_rate, generations):
    _, _, best_cost = oi.genetic_algorithm(
        distance_matrix=distance_matrix,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        generations=generations,
        seed=seed,
    )
    return {
        "seed": seed,
        "best_cost_km": float(best_cost * 111.12),
    }


def z_score(p):
    return np.sqrt(2) * erfinv(2 * p - 1)


def save_dataframe(name, records, log_dir):
    df = pd.DataFrame(records)
    path = os.path.join(log_dir, name)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Salvo: {path}")
    return path


def run_parallel_pcam(config):
    ensure_log_dir(config.log_dir)
    distance_matrix = load_distance_matrix(config.distance_matrix)
    initial_route = oi.nearest_neighbor_route(distance_matrix, start=0)

    mc_seeds = [1000 + i for i in range(config.mc_workers)]
    mc_worker = partial(
        monte_carlo_chunk,
        distance_matrix=distance_matrix,
        samples=config.mc_samples_per_worker,
    )

    print("Executando Monte Carlo paralelo...")
    with mp.Pool(processes=config.mc_workers) as pool:
        mc_results = pool.starmap(mc_worker, [(i, seed) for i, seed in enumerate(mc_seeds)])

    save_dataframe('pcam_mc_runs.csv', mc_results, config.log_dir)

    sa_seeds = [2000 + i for i in range(config.sa_runs)]
    sa_worker_fn = partial(
        sa_worker,
        distance_matrix=distance_matrix,
        initial_route=initial_route,
        iterations=config.sa_iterations,
        temperature=oi.SA_PARAMS['initial_temperature'],
        cooling_rate=oi.SA_PARAMS['cooling_rate'],
    )

    print("Executando Simulated Annealing em paralelo...")
    with mp.Pool(processes=min(config.sa_runs, mp.cpu_count())) as pool:
        sa_results = pool.map(sa_worker_fn, sa_seeds)

    save_dataframe('pcam_sa_runs.csv', sa_results, config.log_dir)

    ga_seeds = [3000 + i for i in range(config.ga_runs)]
    ga_worker_fn = partial(
        ga_worker,
        distance_matrix=distance_matrix,
        population_size=config.population_size,
        elite_size=config.elite_size,
        mutation_rate=config.mutation_rate,
        generations=config.ga_generations,
    )

    print("Executando Genetic Algorithm em paralelo...")
    with mp.Pool(processes=min(config.ga_runs, mp.cpu_count())) as pool:
        ga_results = pool.map(ga_worker_fn, ga_seeds)

    save_dataframe('pcam_ga_runs.csv', ga_results, config.log_dir)

    print("PCAM finalizado.")


if __name__ == '__main__':
    args = parse_args()
    run_parallel_pcam(args)
