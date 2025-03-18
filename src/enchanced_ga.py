import numpy as np
import pandas as pd
import time
from mpi4py import MPI
from genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)

def run_enhanced_parallel_ga(distance_matrix, num_nodes, pop_size=50, num_generations=100, 
                            tournament_size=3, number_tournaments=4, migration_interval=20,
                            migration_size=5, mutation_rate=0.05):
    """
    Run an enhanced parallel genetic algorithm using the Island Model.
    
    Parameters:
        - distance_matrix: Matrix of distances between nodes
        - num_nodes: Number of nodes in the city (excluding depot)
        - pop_size: Size of the population per process
        - num_generations: Number of generations to run
        - tournament_size: Number of individuals to compete in tournament selection
        - number_tournaments: Number of tournaments to run
        - migration_interval: How often to migrate individuals between islands
        - migration_size: Number of individuals to migrate
        - mutation_rate: Probability of mutation for each individual
        
    Returns:
        - best_route: The best route found
        - best_distance: The total distance of the best route
        - history: List of best distances for each generation
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each process maintains its own population (island model)
    population = generate_unique_population(pop_size, num_nodes)
    
    # Each process tracks its own best solution
    best_route = None
    best_distance = float('inf')
    history = []
    
    # Use different random seeds for diversity
    np.random.seed(rank + int(time.time()))
    
    # Apply adaptive mutation rate based on rank
    adaptive_mutation_rate = mutation_rate * (1 + (rank / size))
    
    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitness_scores = np.array([calculate_fitness(route, distance_matrix) for route in population])
        
        # Track the best solution in this generation
        max_fitness_idx = np.argmax(fitness_scores)
        current_best_route = population[max_fitness_idx]
        current_best_distance = -fitness_scores[max_fitness_idx]
        
        # Update overall best if current is better
        if current_best_distance < best_distance:
            best_route = current_best_route.copy()
            best_distance = current_best_distance
        
        # Store best distance for history
        history.append(best_distance)
        
        # Migrate individuals between islands periodically
        if generation % migration_interval == 0 and size > 1:
            # Select top individuals to migrate
            migrants_idx = np.argsort(fitness_scores)[-migration_size:]
            migrants = [population[i] for i in migrants_idx]
            
            # Send to next process in a ring topology
            dest = (rank + 1) % size
            source = (rank - 1) % size
            
            # Send migrants to destination
            comm.send(migrants, dest=dest, tag=20)
            
            # Receive migrants from source
            received_migrants = comm.recv(source=source, tag=20)
            
            # Replace worst individuals with migrants
            worst_idx = np.argsort(fitness_scores)[:migration_size]
            for i, idx in enumerate(worst_idx):
                population[idx] = received_migrants[i]
        
        # Create new population
        new_population = []
        
        # Keep the best individual (elitism)
        new_population.append(current_best_route)
        
        # Fill the rest of the population with offspring
        while len(new_population) < pop_size:
            # Select parents using tournament selection
            parents = select_in_tournament(population, fitness_scores, 
                                         number_tournaments, tournament_size)
            
            # Apply crossover
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    # Create offspring
                    offspring1 = order_crossover(parents[i], parents[i+1])
                    offspring2 = order_crossover(parents[i+1], parents[i])
                    
                    # Apply mutation (with adaptive rate)
                    offspring1 = mutate(offspring1, adaptive_mutation_rate)
                    offspring2 = mutate(offspring2, adaptive_mutation_rate)
                    
                    # Add to new population
                    new_population.append(offspring1)
                    if len(new_population) < pop_size:
                        new_population.append(offspring2)
        
        # Replace old population with new population
        population = new_population
        
        # Gather best distances from all processes
        all_best_distances = comm.gather(best_distance, root=0)
        
        # Master process coordinates
        if rank == 0 and generation % 10 == 0:
            global_best = min(all_best_distances)
            print(f"Generation {generation}: Global best distance = {global_best}")
    
    # Final gathering of best routes from all processes
    all_best_routes = comm.gather((best_route, best_distance), root=0)
    
    if rank == 0:
        # Find the global best route
        global_best_route, global_best_distance = min(all_best_routes, key=lambda x: x[1])
        
        print(f"Final best distance: {global_best_distance}")
        print(f"Best route: 0 -> {' -> '.join(map(str, global_best_route))} -> 0")
        
        return global_best_route, global_best_distance, history
    else:
        return None, None, None

def run_multi_car_enhanced_ga(distance_matrix, num_nodes, num_cars=3, pop_size=50, 
                             num_generations=100, tournament_size=3, number_tournaments=4, 
                             migration_interval=20, migration_size=5, mutation_rate=0.05):
    """
    Extension: Run genetic algorithm for multiple cars (fleet).
    
    This is only a sketch implementation to show how the multi-car approach would work.
    For actual implementation, further development would be needed.
    
    Parameters:
        Similar to run_enhanced_parallel_ga, plus:
        - num_cars: Number of cars in the fleet
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each process will handle optimization for one car configuration
    car_idx = rank % num_cars
    
    # Generate initial random distribution of nodes to cars
    # This would need to be implemented:
    # - All nodes must be visited exactly once by any vehicle
    # - Each vehicle starts and ends at depot (node 0)
    
    # For now, just sketch the approach:
    # 1. Divide nodes approximately equally among cars
    # 2. Each process optimizes route for its assigned car
    # 3. Periodically exchange best routes between processes
    # 4. At the end, combine all routes for final solution
    
    print(f"Process {rank} handling car {car_idx}")
    
    # The rest would follow a similar pattern to run_enhanced_parallel_ga
    # but with modifications to handle multiple cars

if __name__ == "__main__":
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check available processors
    if rank == 0:
        print(f"Running with {size} processes")
    
    # Load distance matrix (everyone needs it)
    distance_df = pd.read_csv('city_distances.csv', index_col=0)
    distance_matrix = distance_df.values
    
    # Get number of nodes (excluding depot)
    num_nodes = len(distance_matrix) - 1
    
    # Parameters
    pop_size = 100
    num_generations = 200
    tournament_size = 4
    number_tournaments = 6
    migration_interval = 20
    migration_size = 5
    mutation_rate = 0.05
    
    # Start timing
    start_time = time.time()
    
    # Run the enhanced parallel genetic algorithm
    best_route, best_distance, history = run_enhanced_parallel_ga(
        distance_matrix, num_nodes, pop_size, num_generations, tournament_size,
        number_tournaments, migration_interval, migration_size, mutation_rate
    )
    
    # End timing (only on master)
    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nExecution time: {elapsed_time:.2f} seconds")

        try:
            extended_df = pd.read_csv('./data/city_distances_extended.csv', index_col=0)
            extended_matrix = extended_df.values
            extended_nodes = len(extended_matrix) - 1
            
            print("\nRunning with extended city dataset...")
            start_time = time.time()
            
            extended_pop_size = 200
            extended_generations = 300
            
           
            ext_best_route, ext_best_distance, ext_history = run_enhanced_parallel_ga(
                extended_matrix, extended_nodes, extended_pop_size, extended_generations,
                tournament_size, number_tournaments, migration_interval, migration_size, mutation_rate
            )
            
            end_time = time.time()
            extended_time = end_time - start_time
            print(f"Extended city execution time: {extended_time:.2f} seconds")
            
        except FileNotFoundError:
            print("Extended city dataset not found. Skipping extended test.")