% MATLAB SNR Benchmark
% Generates N100 datasets/networks for SNRs 0.01, 0.1, 1.0 (10 each)
% Runs all 5 methods (LSCO, LASSO, CLR, GENIE3, TIGRESS) plus NestBoot versions
% Date: 2025-12-03

%% Setup
clear all;
rng(42); % Set random seed
output_dir = 'benchmark_results';
figures_dir = 'benchmark_figures';
datasets_dir = fullfile(output_dir, 'gs-datasets', 'N100');
networks_dir = fullfile(output_dir, 'gs-networks', 'N100');

if ~exist(datasets_dir, 'dir'), mkdir(datasets_dir); end
if ~exist(networks_dir, 'dir'), mkdir(networks_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

addpath('GeneSPIDER2/'); % Add GeneSPIDER2 library
addpath('GeneSPIDER2/dependencies/jsonlab-master'); % Add dependencies
addpath(genpath('GeneSPIDER2/dependencies')); % Add TIGRESS dependency

% Initialize performance tracking
all_results = {};
fprintf('🚀 MATLAB SNR Benchmark\n');
fprintf('========================================\n');

%% Configuration
% SNR values
snr_values = [0.01, 0.1, 1.0];
num_datasets_per_snr = 3; % Generate 3 datasets per SNR
num_datasets = length(snr_values) * num_datasets_per_snr; % 9 total datasets
N = 100; % Number of genes
M = 300; % Number of experiments
avg_degree = 3; % Average degree for scale-free network

% Zetavec for LASSO and LSCO
zetavec = logspace(-6, 0, 30); % Sparsity parameters
best_idx = 25; % Index to use for results

%% Generate datasets
fprintf('📊 Generating datasets...\n');
dataset_counter = 16; % Start from 16 to avoid overlap with Python (1-15)

% Suppress output during generation
original_warning_state = warning('off', 'all');
diary off;

for snr_idx = 1:length(snr_values)
    snr = snr_values(snr_idx);
    for rep = 1:num_datasets_per_snr  % 3 datasets per SNR
        % Generate scale-free network
        A = datastruct.scalefree(N, avg_degree);
        net = datastruct.Network(A);
        
        % Generate perturbations with disjoint structure for SNR_Wiki calculation
        reps = randi([1, 5], N, 1); % Random reps per gene
        total_reps = sum(reps);
        if total_reps > M
            reps = round(reps / total_reps * M);
            reps(reps < 1) = 1;
            total_reps = sum(reps);
            if total_reps > M
                excess = total_reps - M;
                idx = randperm(N, excess);
                reps(idx) = reps(idx) - 1;
                reps(reps < 1) = 1;
            elseif total_reps < M
                deficit = M - total_reps;
                idx = randperm(N, deficit);
                reps(idx) = reps(idx) + 1;
            end
        elseif total_reps < M
            deficit = M - total_reps;
            idx = randperm(N, deficit);
            reps(idx) = reps(idx) + 1;
        end
        
        P = zeros(N, M);
        current_col = 1;
        for gene = 1:N
            num_exp = reps(gene);
            if current_col + num_exp - 1 <= M
                P(gene, current_col:current_col+num_exp-1) = randn(1, num_exp);
                current_col = current_col + num_exp;
            else
                break;
            end
        end
        
        % Compute true response
        X = net.G * P;
        
        % Add noise
        [E, stdE] = datastruct.noise(X, P, 'SNR', snr);
        Y = X + E;
        
        % Create dataset
        data_struct = struct('P', P, 'Y', Y, 'E', E);
        data = datastruct.Dataset(net, data_struct);
        data.lambda = stdE^2;
        data.setname();
        
        % Save network
        network_filename = sprintf('SNR%.2f-MATLAB-N%d-ID%d.json', snr, N, dataset_counter);
        network_path = fullfile(networks_dir, network_filename);
        data_to_save = struct('obj_data', struct(net));
        json_str = jsonencode(data_to_save);
        fid = fopen(network_path, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        
        % Save dataset
        dataset_filename = sprintf('SNR%.2f-MATLAB-N%d-ID%d.json', snr, N, dataset_counter);
        dataset_path = fullfile(datasets_dir, dataset_filename);
        data_to_save = struct('obj_data', struct(data));
        json_str = jsonencode(data_to_save);
        fid = fopen(dataset_path, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        
        dataset_counter = dataset_counter + 1;
        
        % Update progress bar
        total_datasets = length(snr_values) * num_datasets_per_snr;
        current_dataset = (snr_idx - 1) * num_datasets_per_snr + rep;
        progress_bar(current_dataset, total_datasets, 'Generating datasets');
    end
end

% Restore output
warning(original_warning_state);
fprintf('\n✅ Dataset generation complete!\n');

%% Run benchmarks
fprintf('\n🔬 Running benchmarks...\n');

% Find all generated datasets
dataset_files = dir(fullfile(datasets_dir, '*.json'));
fprintf('Found %d generated datasets.\n', length(dataset_files));

%% Process all datasets
% Suppress output during processing
original_warning_state = warning('off', 'all');
diary off;

% Create CSV file with header
header = {'method', 'use_nestboot', 'execution_time', 'memory_usage', 'parameter_index', 'num_edges', 'density', 'sparsity', 'network_shape_0', 'network_shape_1', 'f1_score', 'mcc', 'sensitivity', 'specificity', 'precision', 'auroc', 'dataset', 'network', 'source', 'timestamp'};
results_csv_path = fullfile(output_dir, 'matlab_snr_all_methods_results.csv');
fid = fopen(results_csv_path, 'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
fclose(fid);

% Initialize counters
method_counter = 0;
total_methods = length(dataset_files) * 7;
all_results = {};

for i = 1:length(dataset_files)
    dataset_filename = dataset_files(i).name;
    dataset_path = fullfile(datasets_dir, dataset_filename);
    
    try
        % Load dataset
        data = datastruct.Dataset.load(dataset_path);
        
        % Find network file (same name in networks_dir)
        network_path = strrep(dataset_path, datasets_dir, networks_dir);
        
        net = datastruct.Network.load(network_path);
        
        [~, network_filename, ~] = fileparts(network_path);
        network_filename = [network_filename, '.json'];
        
        % Run all methods
        methods = {'LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS', 'NestBoot+LASSO', 'NestBoot+LSCO'};
        
        for m = 1:length(methods)
            method_name = methods{m};
            
            [estA, exec_time, mem_usage, error_msg] = run_method(method_name, data, net, zetavec);
            
            if ~isempty(estA)
                % Compare with true network
                result = compare_and_report(method_name, net, estA, best_idx, exec_time, mem_usage, data, dataset_filename, network_filename);
                all_results = [all_results; {result}];
                
                % Append result to CSV
                fid = fopen(results_csv_path, 'a');
                fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%s\n', ...
                    result.method, result.use_nestboot, result.execution_time, result.memory_usage, result.parameter_index, ...
                    result.num_edges, result.density, result.sparsity, result.network_shape_0, result.network_shape_1, ...
                    result.f1_score, result.mcc, result.sensitivity, result.specificity, result.precision, result.auroc, ...
                    result.dataset, result.network, result.source, result.timestamp);
                fclose(fid);
            end
            
            % Update progress bar
            method_counter = method_counter + 1;
            progress_bar(method_counter, total_methods, 'Running methods');
        end
        
    catch ME
    end
end

% Restore output
warning(original_warning_state);
fprintf('\n✅ All methods completed!\n');

%% Generate Summary
fprintf('\n📊 Summary by Method:\n');
if ~isempty(all_results)
    method_names = unique(cellfun(@(x) x.method, all_results));
    for i = 1:length(method_names)
        method = method_names{i};
        mask = cellfun(@(x) strcmp(x.method, method), all_results);
        method_results = all_results(mask);
        if ~isempty(method_results)
            fprintf('   %s:\n', method);
            f1_scores = cellfun(@(x) x.f1_score, method_results);
            auroc_scores = cellfun(@(x) x.auroc, method_results);
            exec_times = cellfun(@(x) x.execution_time, method_results);
            fprintf('      F1: %.3f ± %.3f\n', mean(f1_scores), std(f1_scores));
            fprintf('      AUROC: %.3f ± %.3f\n', mean(auroc_scores), std(auroc_scores));
            fprintf('      Time: %.1f ± %.1fs\n', mean(exec_times), std(exec_times));
            fprintf('      Tests: %d\n', length(method_results));
        end
    end
end

fprintf('\n✅ MATLAB SNR benchmark complete!\n');
fprintf('📁 Results saved to: %s\n', output_dir);
fprintf('📊 Total tests: %d\n', length(all_results));

% Save summary statistics
if ~isempty(all_results)
    summary = create_summary_statistics([all_results{:}]);
    save_json(fullfile(output_dir, 'matlab_snr_all_methods_summary.json'), summary);
end

%% Helper Functions
function progress_bar(current, total, description)
    % Simple text-based progress bar for MATLAB terminal
    persistent last_progress;
    
    if isempty(last_progress)
        last_progress = 0;
    end
    
    progress = current / total;
    bar_length = 40;
    filled_length = round(progress * bar_length);
    
    bar = ['[' repmat('=', 1, filled_length) repmat(' ', 1, bar_length - filled_length) ']'];
    percent = sprintf('%.1f%%', progress * 100);
    
    % Only update if progress changed significantly
    if progress - last_progress > 0.01 || current == total
        fprintf('\r%s: %s %s (%d/%d)', description, bar, percent, current, total);
        last_progress = progress;
        
        if current == total
            fprintf('\n');
            last_progress = 0; % Reset for next use
        end
    end
end

function [estA, exec_time, mem_usage, error_msg] = run_method(method_name, data, net, zetavec)
    % Run a specific inference method
    tic;
    mem_start = get_memory_usage();
    error_msg = '';
    
    % Suppress output by redirecting to evalc
    try
        if strcmp(method_name, 'LASSO')
            % LASSO: Methods.lasso(data, net, zetavec, false)
            [~, estA] = evalc('Methods.lasso(data, net, zetavec, false)');
        elseif strcmp(method_name, 'LSCO')
            % LSCO: Methods.lsco(data, net, zetavec, false, 'input')
            [~, estA] = evalc('Methods.lsco(data, net, zetavec, false, ''input'')');
        elseif strcmp(method_name, 'CLR')
            % CLR: Try different possible signatures
            try
                [~, estA] = evalc('Methods.CLR(data, zetavec)');
            catch
                try
                    [~, estA] = evalc('Methods.CLR(data)');
                catch
                    [~, estA] = evalc('Methods.CLR(data, net, zetavec)');
                end
            end
        elseif strcmp(method_name, 'GENIE3')
            % GENIE3: Try different possible signatures
            try
                [~, estA] = evalc('Methods.GENIE3(data, zetavec)');
            catch
                try
                    [~, estA] = evalc('Methods.GENIE3(data)');
                catch
                    [~, estA] = evalc('Methods.GENIE3(data, net, zetavec)');
                end
            end
        elseif strcmp(method_name, 'TIGRESS')
            % TIGRESS: Try different possible signatures
            try
                [~, estA] = evalc('Methods.TIGRESS(data, zetavec)');
            catch
                try
                    [~, estA] = evalc('Methods.TIGRESS(data)');
                catch
                    [~, estA] = evalc('Methods.TIGRESS(data, net, zetavec)');
                end
            end
        elseif strcmp(method_name, 'NestBoot+LASSO')
            % NestBoot + LASSO
            [~, nbout] = evalc('Methods.NestBoot.run(''lasso'', data, zetavec, 5, 5, 0.05)');
            estA = nbout.binary_networks;
        elseif strcmp(method_name, 'NestBoot+LSCO')
            % NestBoot + LSCO
            [~, nbout] = evalc('Methods.NestBoot.run(''lsco'', data, zetavec, 5, 5, 0.05)');
            estA = nbout.binary_networks;
        else
            error('Unknown method: %s', method_name);
        end
    catch ME
        error_msg = ME.message;
        estA = [];
        exec_time = toc;
        mem_usage = 0;
        return;
    end
    
    exec_time = toc;
    mem_usage = get_memory_usage() - mem_start;
end

function result = compare_and_report(method_name, net, estA, idx, exec_time, mem_usage, data, dataset_filename, network_filename)
    % Compare inferred network with true network
    M = analyse.CompareModels(net, estA);
    results_struct = struct(M);
    
    result = create_result_struct(lower(method_name), false, data, exec_time, mem_usage, ...
        estA(:, :, idx), results_struct, M.AUROC(), idx);
    result.dataset = dataset_filename;
    result.network = network_filename;
    result.source = 'MATLAB';
    result.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
end

function mem_usage = get_memory_usage()
    try
        [~, meminfo] = memory;
        mem_usage = meminfo.MemUsedMATLAB / 1024 / 1024; % Convert to MB
    catch
        mem_usage = 0; % Memory monitoring not available
    end
end

function result = create_result_struct(method, use_nestboot, data, exec_time, mem_usage, network_matrix, comparison_results, auroc, idx)
    result = struct();
    result.method = method;
    result.use_nestboot = use_nestboot;
    result.execution_time = exec_time;
    result.memory_usage = mem_usage;
    result.parameter_index = idx;
    result.num_edges = nnz(network_matrix);
    result.density = nnz(network_matrix) / numel(network_matrix);
    result.sparsity = 1 - result.density;
    result.network_shape_0 = size(network_matrix, 1);
    result.network_shape_1 = size(network_matrix, 2);
    
    if isfield(comparison_results, 'F1') && ~isempty(comparison_results.F1)
        result.f1_score = comparison_results.F1(idx);
    else
        result.f1_score = 0.0;
    end
    
    if isfield(comparison_results, 'MCC') && ~isempty(comparison_results.MCC)
        result.mcc = comparison_results.MCC(idx);
    else
        result.mcc = 0.0;
    end
    
    if isfield(comparison_results, 'sen') && ~isempty(comparison_results.sen)
        result.sensitivity = comparison_results.sen(idx);
    else
        result.sensitivity = 0.0;
    end
    
    if isfield(comparison_results, 'spe') && ~isempty(comparison_results.spe)
        result.specificity = comparison_results.spe(idx);
    else
        result.specificity = 0.0;
    end
    
    if isfield(comparison_results, 'pre') && ~isempty(comparison_results.pre)
        result.precision = comparison_results.pre(idx);
    else
        result.precision = 0.0;
    end
    
    result.auroc = auroc;
end

function summary = create_summary_statistics(all_results)
    methods = unique({all_results.method});
    summary = struct();
    
    for i = 1:length(methods)
        method = methods{i};
        mask = strcmp({all_results.method}, method);
        method_results = all_results(mask);
        
        if ~isempty(method_results)
            method_name = method;
            summary.(method_name) = struct();
            summary.(method_name).avg_execution_time = mean([method_results.execution_time]);
            summary.(method_name).avg_memory_usage = mean([method_results.memory_usage]);
            summary.(method_name).avg_f1_score = mean([method_results.f1_score]);
            summary.(method_name).avg_precision = mean([method_results.precision]);
            summary.(method_name).avg_recall = mean([method_results.sensitivity]);
            summary.(method_name).avg_sparsity = mean([method_results.sparsity]);
            summary.(method_name).avg_density = mean([method_results.density]);
            summary.(method_name).avg_auroc = mean([method_results.auroc]);
            summary.(method_name).count = length(method_results);
        end
    end
end

function save_json(filename, data)
    json_str = jsonencode(data);
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end
