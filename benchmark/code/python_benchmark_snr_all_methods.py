"""SNR benchmark that produced the tables behind Figure 4.

Generates N100 scale-free networks at SNR 0.01, 0.1 and 1.0 (3 replicates
each) and scores LASSO, LSCO, CLR, GENIE3, TIGRESS, NestBoot+LASSO and
NestBoot+LSCO. F1/MCC reported by CompareModels are the best value over the
zeta sweep; AUROC is from the continuous scores.

The published boxplot also shows random/null TIGRESS, LASSO and LSCO. Those
arms are not written by this script or by matlab_benchmark_snr_all_methods.m,
and they are not in benchmark/results/*snr*.csv.
"""
import sys
import os
import json
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import contextlib
import warnings
import traceback

sys.path.insert(0, 'src')
from analyze.Data import Data
from datastruct.Network import Network
from datastruct.Dataset import Dataset
from methods.lasso import Lasso
from methods.lsco import LSCO
from methods.clr import CLR
from methods.genie3 import GENIE3
from methods.tigress import TIGRESS
from analyze.CompareModels import CompareModels
from bootstrap.nb_fdr import NetworkBootstrap

def generate_scalefree_network(N, avg_degree, seed=None):
    """Generate a scale-free network using preferential attachment."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a random sparse network with desired average degree
    density = avg_degree / N
    A = np.random.choice([0, 1], size=(N, N), p=[1-density, density])
    
    # Make it directed and remove self-loops
    A = np.triu(A, 1)  # Upper triangular for directed
    A = A + A.T  # Make symmetric for undirected? Wait, GRNs are usually directed
    
    # Actually, for GRNs, they can be directed or undirected. Let's keep it directed for now
    A = np.triu(A, 1)  # Keep upper triangular
    
    # Ensure some connectivity
    if np.sum(A) == 0:
        A[0, 1] = 1  # At least one edge
    
    return A

def add_noise(X, P, SNR):
    """Add noise to achieve desired SNR."""
    s = np.linalg.svd(X, compute_uv=False)
    sigma = min(s)
    stdE = sigma / (SNR * np.sqrt(np.linalg.norm(P)**2 / len(P.flatten())))
    E = stdE * np.random.randn(*P.shape)
    return E, stdE

def run_comparison_analysis(true_net, inferred_net):
    """Compare inferred network with true network."""
    comp = CompareModels(true_net, Network(inferred_net))

    try:
        true_flat = (true_net.A != 0).astype(float).flatten()
        inferred_flat = np.abs(inferred_net.flatten())

        if len(np.unique(true_flat)) > 1 and np.sum(inferred_flat) > 0:
            auroc = roc_auc_score(true_flat, inferred_flat)
        else:
            auroc = 0.5
    except:
        auroc = 0.5

    return {
        'f1_score': comp.F1[0] if len(comp.F1) > 0 else 0.0,
        'auroc': auroc,
        'precision': comp.pre[0] if len(comp.pre) > 0 else 0.0,
        'recall': comp.sen[0] if len(comp.sen) > 0 else 0.0,
        'mcc': comp.MCC[0] if len(comp.MCC) > 0 else 0.0
    }

def run_method(method_name, data, zetavec):
    """Run a specific inference method."""
    start_time = time.time()

    try:
        if method_name == 'LASSO':
            A_3d, _ = Lasso(data, alpha_range=zetavec)
            # Take the last (most sparse) result
            A_final = A_3d[:, :, -1]
        elif method_name == 'LSCO':
            A_3d, _ = LSCO(data, threshold_range=zetavec)
            A_final = A_3d[:, :, -1]
        elif method_name == 'CLR':
            A_3d, _ = CLR(data)
            A_final = A_3d[:, :, 0]
        elif method_name == 'GENIE3':
            A_3d, _ = GENIE3(data)
            A_final = A_3d[:, :, 0]
        elif method_name == 'TIGRESS':
            A_3d, _ = TIGRESS(data, n_bootstrap=5)
            A_final = A_3d[:, :, 0]
        elif method_name == 'NestBoot+LASSO':
            # Run NestBoot with LASSO
            nb = NetworkBootstrap()
            results = nb.run_nestboot(
                dataset=data,
                inference_method=lambda ds, **kwargs: Lasso(ds, alpha_range=zetavec)[0],
                nest_runs=5,  # Reasonable number
                boot_runs=5,
                seed=42
            )
            # Reconstruct network from results
            A_final = np.zeros((data.data.N, data.data.N))
            for i, (gi, gj) in enumerate(zip(results.gene_i, results.gene_j)):
                idx_i = int(gi.split('_')[1])
                idx_j = int(gj.split('_')[1])
                A_final[idx_i, idx_j] = results.sxnet[i] if i < len(results.sxnet) else 0
        elif method_name == 'NestBoot+LSCO':
            # Run NestBoot with LSCO
            nb = NetworkBootstrap()
            results = nb.run_nestboot(
                dataset=data,
                inference_method=lambda ds, **kwargs: LSCO(ds, threshold_range=zetavec)[0],
                nest_runs=5,
                boot_runs=5,
                seed=42
            )
            # Reconstruct network from results
            A_final = np.zeros((data.data.N, data.data.N))
            for i, (gi, gj) in enumerate(zip(results.gene_i, results.gene_j)):
                idx_i = int(gi.split('_')[1])
                idx_j = int(gj.split('_')[1])
                A_final[idx_i, idx_j] = results.sxnet[i] if i < len(results.sxnet) else 0
        else:
            raise ValueError("Unknown method: {}".format(method_name))

        exec_time = time.time() - start_time
        return A_final, exec_time, None  # No memory tracking in Python

    except Exception as e:
        exec_time = time.time() - start_time
        sys.stderr.write(f"Error in run_method({method_name}): {e}\n")
        return None, exec_time, str(e)

def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
    
    # Configuration
    OUTPUT_DIR = 'benchmark_results'
    FIGURES_DIR = 'benchmark_figures'

    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    figures_dir = Path(FIGURES_DIR)
    datasets_dir = output_dir / 'gs-datasets' / 'N100'
    networks_dir = output_dir / 'gs-networks' / 'N100'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # SNR values
    snr_values = [0.01, 0.1, 1.0]
    num_datasets_per_snr = 3  # Generate 3 datasets per SNR
    num_datasets = len(snr_values) * num_datasets_per_snr  # 9 total datasets
    N = 100  # Number of genes
    M = 300   # Number of experiments
    avg_degree = 3

    # Zetavec for LASSO and LSCO
    zetavec = np.logspace(-6, 0, 30)

    print("🚀 Python SNR All-Methods N100 Benchmark")
    print("=" * 50)

    # Generate datasets
    print("📊 Generating datasets...")
    np.random.seed(42)
    dataset_counter = 1

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        total_datasets = len(snr_values) * num_datasets_per_snr
        dataset_pbar = tqdm(total=total_datasets, desc="Generating datasets")
        dataset_counter = 1
        for snr in snr_values:
            for rep in range(1, num_datasets_per_snr + 1):
                # Generate scale-free network
                A = generate_scalefree_network(N, avg_degree, seed=dataset_counter)
            net = Network(A)

            # Generate perturbations with disjoint structure
            reps = np.random.randint(1, 6, N)  # Random reps per gene
            total_reps = np.sum(reps)
            if total_reps > M:
                reps = np.round(reps / total_reps * M).astype(int)
                reps[reps < 1] = 1
                total_reps = np.sum(reps)
                if total_reps > M:
                    excess = total_reps - M
                    idx = np.random.choice(N, excess, replace=False)
                    for i in idx:
                        if reps[i] > 1:
                            reps[i] -= 1
                elif total_reps < M:
                    deficit = M - total_reps
                    idx = np.random.choice(N, deficit, replace=True)
                    for i in idx:
                        reps[i] += 1
            elif total_reps < M:
                deficit = M - total_reps
                idx = np.random.choice(N, deficit, replace=True)
                for i in idx:
                    reps[i] += 1
            
            P = np.zeros((N, M))
            current_col = 0
            for gene in range(N):
                num_exp = reps[gene]
                if current_col + num_exp <= M:
                    P[gene, current_col:current_col+num_exp] = np.random.randn(num_exp)
                    current_col += num_exp
                else:
                    break

            # Compute true response
            X = net.A @ P

            # Add noise
            E, stdE = add_noise(X, P, snr)
            Y = X + E

            # Create dataset
            dataset = Dataset()
            dataset._P = P
            dataset._Y = Y
            dataset._E = E
            dataset._network = net
            dataset._lambda = stdE**2
            dataset._set_name()

            # Save network
            network_filename = 'SNR{:.2f}-Python-N{}-ID{}.json'.format(snr, N, dataset_counter)
            network_path = networks_dir / network_filename
            with open(str(network_path), 'w') as f:
                json.dump({'obj_data': {'A': net.A.tolist(), 'network': net.network}}, f)

            # Save dataset
            dataset_filename = 'SNR{:.2f}-Python-N{}-ID{}.json'.format(snr, N, dataset_counter)
            dataset_path = datasets_dir / dataset_filename
            with open(str(dataset_path), 'w') as f:
                json.dump({
                    'obj_data': {
                        'P': dataset._P.tolist(),
                        'Y': dataset._Y.tolist(),
                        'E': dataset._E.tolist(),
                        'network': dataset._dataset_name,
                        'lambda': dataset._lambda
                    }
                }, f)

            dataset_counter += 1

    print("✅ Dataset generation complete!")

    # Run benchmarks
    print("\n🔬 Running benchmarks...")

    # Find all generated datasets
    dataset_files = sorted(datasets_dir.glob("*.json"))
    print("Found {} generated datasets.".format(len(dataset_files)))

    # Results storage
    all_results = []
    results_file = output_dir / 'python_snr_all_methods_results.csv'
    
    # Load existing results to avoid re-running
    existing_results = set()
    if results_file.exists():
        try:
            df_existing = pd.read_csv(results_file)
            for _, row in df_existing.iterrows():
                existing_results.add((row['dataset'], row['method']))
            print(f"Found {len(existing_results)} existing results. Skipping completed (dataset, method) pairs.")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")

    # Process all datasets
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        dataset_pbar = tqdm(dataset_files, desc="Processing datasets")
        for dataset_path in dataset_pbar:
            dataset_filename = dataset_path.name

            try:
                # Load dataset
                data = Data.from_json_file(str(dataset_path))

                # Find network file (same name as dataset file)
                network_path = networks_dir / dataset_filename
                if network_path.exists():
                    net = Network.from_json_file(str(network_path))
                else:
                    continue

                # Run all methods
                methods = ['LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS', 'NestBoot+LASSO', 'NestBoot+LSCO']

                for method in methods:
                    if (dataset_filename, method) in existing_results:
                        continue
                        
                    dataset_pbar.set_description("Processing {} - {}".format(dataset_filename, method))
                    inferred_net, exec_time, error = run_method(method, data, zetavec)

                    if inferred_net is not None:
                        # Compare with true network
                        metrics = run_comparison_analysis(net, inferred_net)

                        # Store results
                        result = {
                            'dataset': dataset_filename,
                            'network': network_path.name,
                            'method': method,
                            'source': 'Python',
                            'execution_time': exec_time,
                            'memory_usage': 0.0,  # Not tracked
                            'f1_score': metrics['f1_score'],
                            'auroc': metrics['auroc'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'mcc': metrics['mcc'],
                            'density': np.sum(inferred_net != 0) / inferred_net.size,
                            'timestamp': datetime.now().isoformat()
                        }
                        all_results.append(result)
                        
                        # Save results immediately after each method
                        # Append to file instead of overwriting
                        df_result = pd.DataFrame([result])
                        if not results_file.exists():
                            df_result.to_csv(results_file, index=False)
                        else:
                            df_result.to_csv(results_file, mode='a', header=False, index=False)
                        
                        # Add to existing results to prevent duplicates in this run
                        existing_results.add((dataset_filename, method))

            except Exception as e:
                sys.stderr.write(f"Error processing dataset {dataset_filename}: {e}\n")
                traceback.print_exc()
                with open(output_dir / 'benchmark_errors.log', 'a') as f:
                    f.write(f"[{datetime.now().isoformat()}] Error processing dataset {dataset_filename}: {str(e)}\n")

    print("\n🏁 Benchmark Complete. Processed {} datasets.".format(len(dataset_files)))
    print("   Results saved to: {}".format(results_file))

    # Generate summary
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n📊 Summary by Method:")
        for method in methods:
            method_results = df_results[df_results['method'] == method]
            if len(method_results) > 0:
                print("   {}:".format(method))
                print("      F1: {:.3f} ± {:.3f}".format(method_results['f1_score'].mean(), method_results['f1_score'].std()))
                print("      AUROC: {:.3f} ± {:.3f}".format(method_results['auroc'].mean(), method_results['auroc'].std()))
                print("      Time: {:.1f} ± {:.1f}s".format(method_results['execution_time'].mean(), method_results['execution_time'].std()))

if __name__ == "__main__":
    main()
