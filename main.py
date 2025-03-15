import time
from src.square_program import run_benchmark
from src.connection_pool import run_simulation

def main():
    """Main entry point for the assignment."""
    print("DSAI 3202 - Parallel and Distributed Computing - Assignment 1")
    print("=" * 60)
    
    # Part 3: Square Program
    print("\nPart 3: Square Program")
    print("-" * 30)
    
    # Run benchmarks with 10^6 and 10^7 numbers
    run_benchmark(10**6)
    run_benchmark(10**7)
    
    # Part 4: Process Synchronization with Semaphores
    print("\nPart 4: Process Synchronization with Semaphores")
    print("-" * 50)
    
    # Run simulation with different numbers of processes and pool sizes
    run_simulation(num_processes=10, pool_size=3)
    run_simulation(num_processes=5, pool_size=2)

if __name__ == "__main__":
    main()