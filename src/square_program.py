import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def square(x):
    """Compute the square of a number."""
    return x * x

def sequential_approach(numbers):
    """Process numbers sequentially."""
    start_time = time.time()
    results = [square(num) for num in numbers]
    end_time = time.time()
    return results, end_time - start_time

def process_per_number_approach(numbers):
    """Create a separate process for each number."""
    start_time = time.time()
    
    manager = mp.Manager()
    result_dict = manager.dict()
    
    def worker(idx, num):
        result_dict[idx] = square(num)
    
    processes = []
    for i, num in enumerate(numbers):
        p = mp.Process(target=worker, args=(i, num))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # Convert the result dictionary to a list maintaining original order
    results = [result_dict[i] for i in range(len(numbers))]
    end_time = time.time()
    
    return results, end_time - start_time

def pool_map_approach(numbers, processes=None):
    """Use multiprocessing pool with map()."""
    start_time = time.time()
    
    with mp.Pool(processes=processes) as pool:
        results = pool.map(square, numbers)
    
    end_time = time.time()
    return results, end_time - start_time

def pool_apply_approach(numbers, processes=None):
    """Use multiprocessing pool with apply()."""
    start_time = time.time()
    
    with mp.Pool(processes=processes) as pool:
        results = [pool.apply(square, args=(num,)) for num in numbers]
    
    end_time = time.time()
    return results, end_time - start_time

def pool_apply_async_approach(numbers, processes=None):
    """Use multiprocessing pool with apply_async()."""
    start_time = time.time()
    
    with mp.Pool(processes=processes) as pool:
        results_async = [pool.apply_async(square, args=(num,)) for num in numbers]
        results = [r.get() for r in results_async]
    
    end_time = time.time()
    return results, end_time - start_time

def futures_approach(numbers, max_workers=None):
    """Use concurrent.futures ProcessPoolExecutor."""
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(square, numbers))
    
    end_time = time.time()
    return results, end_time - start_time

def run_benchmark(size=10**6, max_workers=None):
    """Run all benchmark approaches on a dataset of given size."""
    print(f"\nBenchmarking with {size} numbers:")
    
    # Generate random numbers
    numbers = [random.randint(1, 1000) for _ in range(size)]
    
    # Cannot run process-per-number for very large lists (10^7) due to system limitations
    approaches = [
        ("Sequential", lambda: sequential_approach(numbers)),
        ("Pool Map", lambda: pool_map_approach(numbers, max_workers)),
        ("Pool Apply", lambda: pool_apply_approach(numbers, max_workers)),
        ("Pool Apply Async", lambda: pool_apply_async_approach(numbers, max_workers)),
        ("ProcessPoolExecutor", lambda: futures_approach(numbers, max_workers))
    ]
    
    # Add process-per-number only for smaller lists
    if size <= 10**6:
        approaches.insert(1, ("Process per Number", lambda: process_per_number_approach(numbers)))
    
    results = {}
    for name, func in approaches:
        try:
            print(f"Running {name}...", end="", flush=True)
            _, duration = func()
            results[name] = duration
            print(f" completed in {duration:.4f} seconds")
        except Exception as e:
            print(f" ERROR: {str(e)}")
            results[name] = None
    
    return results

if __name__ == "__main__":
    # Determine optimal number of workers (usually cpu_count)
    cpu_count = mp.cpu_count()
    print(f"Number of CPU cores: {cpu_count}")
    max_workers = cpu_count
    
    # Run benchmarks
    results_10_6 = run_benchmark(10**6, max_workers)
    results_10_7 = run_benchmark(10**7, max_workers)
    
    # Print summary
    print("\nSummary:")
    print("Approach                | 10^6 numbers      | 10^7 numbers")
    print("-" * 60)
    
    for name in results_10_6.keys():
        time_10_6 = f"{results_10_6[name]:.4f}s" if results_10_6[name] is not None else "N/A"
        time_10_7 = f"{results_10_7[name]:.4f}s" if name in results_10_7 and results_10_7[name] is not None else "N/A"
        print(f"{name:<22} | {time_10_6:<16} | {time_10_7}")