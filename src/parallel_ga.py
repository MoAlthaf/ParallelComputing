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

def run_parallel_ga(distance_matrix, num_nodes, pop_size=50, num_generations=100, 
                    tournament_size=3, number_tournaments=4, mutation_rate=0.05):
    """
    Run the genetic algorithm in parallel using MPI.
    
    Parameters:
        - distance_matrix: Matrix of distances between nodes
        - num_nodes: Number of nodes in the city (excluding depot)
        - pop_size: Size of the population
        - num_generations: Number of generations to run
        - tournament_size: Number of individuals to compete in tournament selection
        - number_tournaments: Number of tournaments to run
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
    
    # Master process (rank 0) coordinates
    if rank == 0:
        # Generate initial population
        population = generate_unique_population(pop_size, num_nodes)
        
        # Track the best solution and history
        best_route = None
        best_distance = float('inf')
        history = []
        
        for generation in range(num_generations):
            # Distribute population among workers
            chunks = np.array_split(population, size - 1)
            
            # Send chunks to worker processes
            for i, chunk in enumerate(chunks):
                if i < size - 1:  # Make sure we don't exceed number of workers
                    comm.send(chunk.tolist(), dest=i+1, tag=10)
            
            # Collect fitness scores from workers
            fitness_scores = np.empty(pop_size)
            start_idx = 0
            
            for i in range(size - 1):
                if i < len(chunks):
                    chunk_size = len(chunks[i])
                    chunk_scores = comm.recv(source=i+1, tag=11)
                    fitness_scores[start_idx:start_idx+chunk_size] = chunk_scores
                    start_idx += chunk_size
            
            # Track the best solution in this generation
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_route = population[max_fitness_idx]
            current_best_distance = -fitness_scores[max_fitness_idx]  # Convert back to distance
            
            # Update overall best if current is better
            if current_best_distance < best_distance:
                best_route = current_best_route.copy()
                best_distance = current_best_distance
            
            # Store best distance for history
            history.append(best_distance)
            
            # Create new population
            new_population = []
            
            # Keep the best individual (elitism)
            new_population.append(current_best_route)
            
            # Fill the rest of the population with offspring
            while len(new_population) < pop_size:
                # Selection, crossover and mutation can be also parallelized
                # but we'll keep it sequential for simplicity
                parents = select_in_tournament(population, fitness_scores, 
                                              number_tournaments, tournament_size)
                
                for i in range(0, len(parents), 2):
                    if i+1 < len(parents):
                        offspring1 = order_crossover(parents[i], parents[i+1])
                        offspring2 = order_crossover(parents[i+1], parents[i])
                        
                        offspring1 = mutate(offspring1, mutation_rate)
                        offspring2 = mutate(offspring2, mutation_rate)
                        
                        new_population.append(offspring1)
                        if len(new_population) < pop_size:
                            new_population.append(offspring2)
            
            # Replace old population with new population
            population = new_population
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best distance = {best_distance}")
        
        # Print final result
        print(f"Final best distance: {best_distance}")
        print(f"Best route: 0 -> {' -> '.join(map(str, best_route))} -> 0")
        
        # Signal workers to stop
        for i in range(1, size):
            comm.send(None, dest=i, tag=10)
        
        return best_route, best_distance, history
    
    # Worker processes (rank > 0)
    else:
        while True:
            # Receive chunk of population from master
            chunk = comm.recv(source=0, tag=10)
            
            # Check if it's termination signal
            if chunk is None:
                break
            
            # Calculate fitness for each individual in the chunk
            chunk_scores = np.array([calculate_fitness(route, distance_matrix) for route in chunk])
            
            # Send results back to master
            comm.send(chunk_scores, dest=0, tag=11)
        
        return None, None, None

if __name__ == "__main__":
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Load distance matrix (everyone needs it)
    distance_df = pd.read_csv('./data/city_distances.csv', index_col=0)
    distance_matrix = distance_df.values
    
    # Get number of nodes (excluding depot)
    num_nodes = len(distance_matrix) - 1
    
    # Parameters
    pop_size = 100
    num_generations = 200
    tournament_size = 4
    number_tournaments = 6
    mutation_rate = 0.05
    
    # Start timing
    start_time = time.time()
    
    # Run the parallel genetic algorithm
    best_route, best_distance, history = run_parallel_ga(
        distance_matrix, num_nodes, pop_size, num_generations, 
        tournament_size, number_tournaments, mutation_rate
    )
    
    # End timing (only on master)
    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nExecution time: {elapsed_time:.2f} seconds")