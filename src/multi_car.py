import numpy as np
import pandas as pd
import time
import sys
from mpi4py import MPI

def evaluate_routes(vehicle_paths, dist_matrix, circular_route=True):
    """Assess quality of vehicle path configurations."""
    route_penalty = 0
    valid_config = True
    
    for path in vehicle_paths:
        if path[0] != 0:
            valid_config = False
            break
            
        path_length = 0
        for idx in range(len(path) - 1):
            current, next_node = path[idx], path[idx+1]
            segment_dist = dist_matrix[current, next_node]
            
            if segment_dist == 1e5:
                valid_config = False
                break
            path_length += segment_dist
        
        if circular_route and len(path) > 1:
            final_node = path[-1]
            return_dist = dist_matrix[final_node, 0]
            if return_dist == 1e5:
                valid_config = False
                break
            path_length += return_dist
            
        route_penalty += path_length
        
        if not valid_config:
            break
    
    return -1e6 if not valid_config else -route_penalty

def create_vehicle_population(pop_count, node_total, vehicle_count, min_stops=3):
    """Generate initial solutions for vehicle routing."""
    solutions = []
    
    for _ in range(pop_count):
        nodes = list(np.random.permutation(range(1, node_total)))
        
        base_allocation = len(nodes) // vehicle_count
        remaining = len(nodes) % vehicle_count
        
        if base_allocation < min_stops:
            usable_vehicles = len(nodes) // min_stops
            usable_vehicles = max(1, usable_vehicles)
            base_allocation = len(nodes) // usable_vehicles
            remaining = len(nodes) % usable_vehicles
            actual_vehicles = usable_vehicles
        else:
            actual_vehicles = vehicle_count
        
        allocations = []
        ptr = 0
        
        for v in range(actual_vehicles):
            allocation = base_allocation + 1 if v < remaining else base_allocation
            if ptr + allocation <= len(nodes):
                vehicle_path = [0] + nodes[ptr:ptr+allocation]
                ptr += allocation
                allocations.append(vehicle_path)
        
        if ptr < len(nodes):
            allocations[-1].extend(nodes[ptr:])
        
        solutions.append(allocations)
    
    return solutions

def resolve_node_conflicts(solution):
    """Rectify duplicate node assignments across vehicle paths."""
    all_stops = []
    for path in solution:
        all_stops.extend(path[1:])
    
    duplicates = []
    unique_stops = []
    encountered = set()
    
    for stop in all_stops:
        if stop in encountered:
            duplicates.append(stop)
        else:
            encountered.add(stop)
            unique_stops.append(stop)
    
    if not duplicates:
        return solution
    
    missing_stops = list(set(range(1, max(all_stops)+1) - encountered))
    
    replacement_map = {}
    for d, m in zip(duplicates, missing_stops[:len(duplicates)]):
        replacement_map[d] = m
    
    corrected = []
    for path in solution:
        new_path = [0]
        for stop in path[1:]:
            new_path.append(replacement_map.get(stop, stop))
        
        if len(new_path) > 1:
            corrected.append(new_path)
    
    return corrected

def sequential_crossover(p1, p2):
    """Produce offspring through sequential segment inheritance."""
    route_len = len(p1)
    if route_len != len(p2):
        return p1.copy()
    
    if route_len > 2:
        a, b = sorted([np.random.randint(0, route_len), np.random.randint(0, route_len)])
    else:
        a, b = 0, 0
    
    child = [None]*route_len
    child[a:b+1] = p1[a:b+1]
    
    remaining = [n for n in p2 if n not in child[a:b+1]]
    fill_pos = [i for i in range(route_len) if child[i] is None]
    
    for idx, pos in enumerate(fill_pos):
        child[pos] = remaining[idx] if idx < len(remaining) else p1[pos]
    
    return child

def combine_solutions(sol1, sol2, combine_method="path"):
    """Merge two solutions to create offspring."""
    if combine_method == "path" and len(sol1) > 1 and len(sol2) > 1:
        split_point = np.random.randint(1, min(len(sol1), len(sol2)))
        offspring_a = sol1[:split_point] + sol2[split_point:]
        offspring_b = sol2[:split_point] + sol1[split_point:]
        return resolve_node_conflicts(offspring_a), resolve_node_conflicts(offspring_b)
    else:
        merged_a = []
        merged_b = []
        max_paths = max(len(sol1), len(sol2))
        
        for i in range(max_paths):
            if i < len(sol1) and i < len(sol2):
                o1 = [0] + sequential_crossover(sol1[i][1:], sol2[i][1:])
                o2 = [0] + sequential_crossover(sol2[i][1:], sol1[i][1:])
                merged_a.append(o1)
                merged_b.append(o2)
            else:
                source = sol1 if i < len(sol1) else sol2
                merged_a.append(source[i].copy())
                merged_b.append(source[i].copy())
        
        return resolve_node_conflicts(merged_a), resolve_node_conflicts(merged_b)

def modify_solution(solution, mutation_prob=0.1, transfer_prob=0.2):
    """Introduce random changes to vehicle paths."""
    modified = [p.copy() for p in solution]
    
    # Intra-path modifications
    for i, path in enumerate(modified):
        if np.random.rand() < mutation_prob and len(path) > 3:
            a, b = np.random.choice(range(1, len(path)), 2, False)
            path[a], path[b] = path[b], path[a]
    
    # Inter-path modifications
    if len(modified) > 1 and np.random.rand() < transfer_prob:
        src, dest = np.random.choice(len(modified), 2, False)
        if len(modified[src]) > 2:
            move_idx = np.random.randint(1, len(modified[src]))
            moved_node = modified[src].pop(move_idx)
            insert_pos = np.random.randint(1, len(modified[dest])+1)
            modified[dest].insert(insert_pos, moved_node)
    
    return modified

def select_contestant(scores, pool_size):
    """Choose solution through competitive selection."""
    candidates = np.random.choice(len(scores), pool_size, False)
    contest_scores = [scores[i] for i in candidates]
    return candidates[np.argmax(contest_scores)]

def execute_vehicle_ga(dist_file, vehicles=3, pop_count=200, iterations=100, min_stops=2):
    """Optimize vehicle routing using evolutionary methods."""
    try:
        dist_data = pd.read_csv(dist_file).values
        print(f"Loaded distance data: {dist_data.shape[0]} locations")
    except Exception as e:
        print(f"Data load error: {e}")
        return None, None
    
    node_count = dist_data.shape[0]
    
    print(f"Creating initial solutions for {vehicles} vehicles...")
    population = create_vehicle_population(pop_count, node_count, vehicles, min_stops)
    
    top_score = -np.inf
    best_config = None
    mutation_prob = 0.2
    crossover_prob = 0.8
    selection_pool = 5
    
    start_timer = time.time()
    
    for iter in range(iterations):
        iter_start = time.time()
        
        quality_scores = [evaluate_routes(s, dist_data) for s in population]
        
        current_top_idx = np.argmax(quality_scores)
        current_top = quality_scores[current_top_idx]
        
        if current_top > top_score:
            top_score = current_top
            best_config = [p.copy() for p in population[current_top_idx]]
            print(f"Iter {iter}: Improved score {top_score}")
            print("Vehicle paths:")
            for idx, path in enumerate(best_config):
                path_dist = sum(dist_data[path[i], path[i+1]] for i in range(len(path)-1))
                print(f"  Vehicle {idx+1}: {path} ({path_dist} units)")
        
        if iter % 10 == 0:
            print(f"Iteration {iter}: Current best = {current_top}")
        
        new_pop = [population[current_top_idx]]
        
        while len(new_pop) < pop_count:
            p1_idx = select_contestant(quality_scores, selection_pool)
            p2_idx = select_contestant(quality_scores, selection_pool)
            
            if np.random.rand() < crossover_prob:
                combine_type = "path" if np.random.rand() < 0.5 else "node"
                o1, o2 = combine_solutions(population[p1_idx], population[p2_idx], combine_type)
            else:
                o1, o2 = [p.copy() for p in population[p1_idx]], [p.copy() for p in population[p2_idx]]
            
            new_pop.extend([modify_solution(o1, mutation_prob), modify_solution(o2, mutation_prob)])
        
        population = new_pop[:pop_count]
        
        if iter % 10 == 0:
            print(f"  Iteration duration: {time.time()-iter_start:.1f}s")
    
    total_duration = time.time() - start_timer
    print(f"\nOptimization completed in {total_duration:.1f} seconds")
    
    if best_config:
        total_dist = -top_score
        print("\n******* OPTIMAL VEHICLE PATHS *******")
        print(f"Total distance: {total_dist}")
        for idx, path in enumerate(best_config):
            path_dist = sum(dist_data[path[i], path[i+1]] for i in range(len(path)-1))
            if len(path) > 1:
                path_dist += dist_data[path[-1], 0]
            print(f"Vehicle {idx+1}: {path} → {path_dist} units")
        
        stops_per_vehicle = [len(p)-1 for p in best_config]
        print(f"\nWork distribution:")
        print(f"  Min stops: {min(stops_per_vehicle)}")
        print(f"  Max stops: {max(stops_per_vehicle)}")
        print(f"  Avg stops: {np.mean(stops_per_vehicle):.1f}")
    else:
        print("\nNo valid configuration found")
    
    return best_config, top_score

def master_process():
    """Coordinate parallel optimization processes."""
    if len(sys.argv) > 1:
        dist_file = sys.argv[1]
    else:
        dist_file = 'city_distances_extended.csv'
        if not pd.io.common.file_exists(dist_file):
            dist_file = '../data/city_distances_extended.csv'
            if not pd.io.common.file_exists(dist_file):
                dist_file = 'city_distances.csv'
                if not pd.io.common.file_exists(dist_file):
                    dist_file = '../data/city_distances.csv'
    
    vehicle_count = 3
    if len(sys.argv) > 2:
        try:
            vehicle_count = int(sys.argv[2])
        except:
            print(f"Invalid vehicle count. Using {vehicle_count}")
    
    mpi_comm = MPI.COMM_WORLD
    proc_id = mpi_comm.Get_rank()
    proc_count = mpi_comm.Get_size()
    
    if proc_id == 0:
        print(f"Initiating multi-vehicle optimization with {proc_count} workers")
        print(f"Distance data source: {dist_file}")
        print(f"Vehicle fleet size: {vehicle_count}")
    
    pop_size = 200 + proc_id * 50
    max_iters = 100 - proc_id * 10
    min_stops = 2 + proc_id % 3
    
    solution, score = execute_vehicle_ga(
        dist_file,
        vehicles=vehicle_count,
        pop_count=pop_size,
        iterations=max_iters,
        min_stops=min_stops
    )
    
    all_results = mpi_comm.gather((solution, score, proc_id), root=0)
    
    if proc_id == 0:
        top_score = -np.inf
        best_solution = None
        best_worker = -1
        
        for result in all_results:
            if result[1] > top_score:
                top_score = result[1]
                best_solution = result[0]
                best_worker = result[2]
        
        print("\n******* GLOBAL OPTIMUM FOUND *******")
        print(f"Origin worker: {best_worker}")
        print(f"Optimization score: {top_score}")
        
        try:
            dist_matrix = pd.read_csv(dist_file).values
            total_dist = -top_score
            print(f"Combined route length: {total_dist}")
            for vidx, path in enumerate(best_solution):
                path_dist = sum(dist_matrix[path[i], path[i+1]] for i in range(len(path)-1))
                if len(path) > 1:
                    path_dist += dist_matrix[path[-1], 0]
                print(f"Vehicle {vidx+1}: {path} → {path_dist}")
        except Exception as e:
            print(f"Final evaluation error: {e}")

if __name__ == "__main__":
    master_process()