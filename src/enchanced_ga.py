from mpi4py import MPI
import numpy as np
import pandas as pd
import time
from genetic_algorithms_functions import compute_adaptation, tournament_selection, pmx_crossover, apply_mutation, create_population

def execute_ga_process(instance_seed, processor_id):
    """Execute genetic algorithm with given seed for parallel processing."""
    # Configuration parameters
    pop_size = 5000
    max_generations = 100
    mutation_prob = 0.2
    tournament_count = 4
    stagnation_threshold = 5
    
    # Load city distance data
    dist_data = pd.read_csv('../data/city_distances.csv').values
    cities_count = dist_data.shape[0]
    
    # Initialize random state
    np.random.seed(instance_seed + processor_id)
    
    # Create starting population
    current_pop = create_population(pop_size, cities_count)
    
    # Track progress variables
    top_adaptation = -np.inf
    optimal_path = None
    stagnation_timer = 0
    
    # Evolution process
    exec_start = time.time()
    for gen in range(max_generations):
        # Calculate population fitness
        adapt_scores = np.array([compute_adaptation(ind, dist_data) for ind in current_pop])
        
        # Update best solution
        current_top = np.max(adapt_scores)
        if current_top > top_adaptation:
            top_adaptation = current_top
            top_index = np.argmax(adapt_scores)
            optimal_path = current_pop[top_index].copy()
            stagnation_timer = 0
        else:
            stagnation_timer += 1
        
        # Population reset condition
        if stagnation_timer >= stagnation_threshold:
            new_pop = create_population(pop_size - 1, cities_count)
            if optimal_path is not None:
                new_pop.append(optimal_path)
            current_pop = new_pop
            stagnation_timer = 0
            continue
        
        # Parent selection process
        chosen_parents = tournament_selection(current_pop, adapt_scores, 
                                           tournament_size=tournament_count)
        
        # Ensure even number for pairing
        if len(chosen_parents) & 1:
            chosen_parents.append(chosen_parents[-1])
        
        # Generate new offspring
        children = []
        for p in range(0, len(chosen_parents), 2):
            if p + 1 < len(chosen_parents):
                p1, p2 = chosen_parents[p][1:], chosen_parents[p+1][1:]
                offspring = pmx_crossover(p1, p2)
                children.append([0] + offspring)
        
        # Apply genetic modifications
        modified_children = [apply_mutation(ind.copy(), mutation_prob) 
                           for ind in children]
        
        # Update population
        replace_count = len(modified_children)
        replace_indices = np.argsort(adapt_scores)[:replace_count]
        for i, idx in enumerate(replace_indices):
            current_pop[idx] = modified_children[i]
    
    total_time = time.time() - exec_start
    
    return {
        'optimal_path': optimal_path,
        'adaptation_score': top_adaptation,
        'execution_time': total_time,
        'processor_num': processor_id
    }

def master_function():
    # MPI setup
    mpi_comm = MPI.COMM_WORLD
    current_rank = mpi_comm.Get_rank()
    process_count = mpi_comm.Get_size()
    
    # Execute parallel GA runs
    main_seed = 42
    ga_result = execute_ga_process(main_seed, current_rank)
    
    # Collect and process results
    collected_data = mpi_comm.gather(ga_result, root=0)
    
    # Root process analysis
    if current_rank == 0:
        print(f"Aggregated {len(collected_data)} parallel executions")
        
        # Determine superior solution
        max_score = -np.inf
        superior_result = None
        
        for data in collected_data:
            if data['adaptation_score'] > max_score:
                max_score = data['adaptation_score']
                superior_result = data
        
        print("\n******* OPTIMAL SOLUTION FOUND *******")
        print(f"Origin processor: {superior_result['processor_num']}")
        print(f"Path sequence: {superior_result['optimal_path']}")
        
        if max_score > -1e6:
            print(f"Route length: {-max_score}")
            print("VALID route configuration")
        else:
            print("ALERT: Invalid route detected")
        
        # Compute performance statistics
        total_time = sum(r['execution_time'] for r in collected_data)
        mean_duration = total_time / len(collected_data)
        max_duration = max(r['execution_time'] for r in collected_data)
        achieved_speedup = max_duration / mean_duration * process_count
        
        print("\n******* EXECUTION ANALYSIS *******")
        print(f"Parallel nodes used: {process_count}")
        print(f"Mean execution duration: {mean_duration:.2f}s")
        print(f"Potential speedup: {process_count:.2f}x")
        print(f"Realized speedup: {achieved_speedup:.2f}x")
        print(f"System efficiency: {achieved_speedup/process_count:.2%}")

if __name__ == "__main__":
    master_function()