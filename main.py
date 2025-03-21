import os
import math
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random
from src.square_program import square
from src.connection_pool import ConnectionPool, access_database

# Configuration parameters
DATA_SIZE = 10**6 * 10
INPUT_RANGE = (1, 100)
WORKER_COUNT = os.cpu_count() or 4

# Data generation
input_values = [random.randrange(*INPUT_RANGE) for _ in range(DATA_SIZE)]

def linear_execution():
    start = perf_counter()
    output = [square(x) for x in input_values]
    duration = perf_counter() - start
    print(f"Linear execution duration: {duration:.3f}s")

def parallel_map_processing():
    start = perf_counter()
    with mp.Pool(processes=WORKER_COUNT) as workers:
        result = workers.map(square, input_values)
    duration = perf_counter() - start
    print(f"Parallel mapping duration: {duration:.3f}s")

def future_based_processing():
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        result = list(executor.map(square, input_values))
    duration = perf_counter() - start
    print(f"Futures-based processing time: {duration:.3f}s")

def execute():
    print("Performance benchmarking:")
    for func in [linear_execution, parallel_map_processing, future_based_processing]:
        func()
    
    print("\nDatabase connection stress test")
    connection_manager = ConnectionPool(capacity=4)
    workers = [mp.Process(target=access_database, args=(connection_manager,)) 
              for _ in range(8)]
    
    for w in workers:
        w.start()
    for w in workers:
        w.join()

if __name__ == "__main__":
    execute()