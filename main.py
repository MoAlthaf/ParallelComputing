import time
import threading
import multiprocessing

#Sequential Case

def sequential_sum(n):
    return sum(range(1, n + 1))

# Measure execution time for sequential sum
def sequential_case():
    start_time = time.time()
    n = 10000000  # Example large number
    total_sum = sequential_sum(n)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Sequential Sum: {total_sum}")
    print(f"Execution Time: {execution_time} seconds")
    return execution_time

# Threading Case

# Function to calculate sum of a range in a thread
def threaded_sum(start, end, result):
    result[0] = sum(range(start, end + 1))

# Function to parallelize the sum using threads
def parallel_threaded_sum(n, num_threads):
    threads = []
    result = [0] * num_threads  # Store results for each thread
    range_size = n // num_threads

    for i in range(num_threads):
        start = i * range_size + 1
        end = (i + 1) * range_size if i != num_threads - 1 else n
        thread = threading.Thread(target=threaded_sum, args=(start, end, result[i:i+1]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return sum(result)

# Measure execution time for threading
def threading_case():
    start_time = time.time()
    n = 10000000
    num_threads = 4  # Example number of threads
    total_sum = parallel_threaded_sum(n, num_threads)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Threaded Sum: {total_sum}")
    print(f"Execution Time: {execution_time} seconds")
    return execution_time

# Multiprocessing Case

# Function to calculate sum of a range in a process
def process_sum(start, end, result, idx):
    result[idx] = sum(range(start, end + 1))

# Function to parallelize the sum using multiprocessing
def parallel_multiprocessing_sum(n, num_processes):
    processes = []
    result = multiprocessing.Array('i', num_processes) 
    range_size = n // num_processes

    for i in range(num_processes):
        start = i * range_size + 1
        end = (i + 1) * range_size if i != num_processes - 1 else n
        process = multiprocessing.Process(target=process_sum, args=(start, end, result, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    return sum(result)

# Measure execution time for multiprocessing
def multiprocessing_case():
    start_time = time.time()
    n = 10000000
    num_processes = 4  
    total_sum = parallel_multiprocessing_sum(n, num_processes)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Multiprocessing Sum: {total_sum}")
    print(f"Execution Time: {execution_time} seconds")
    return execution_time

# Performance Analysis 

def calculate_speedup(sequential_time, parallel_time):
    return sequential_time / parallel_time

def calculate_efficiency(speedup, num_processors):
    return speedup / num_processors

def amdhal_law_speedup(p, n):
    return 1 / (1 - p + p / n)

def gustafson_law_speedup(p, n):
    return n - (1 - p) * n

# Main Execution 

def main():
    # Sequential Case
    sequential_time = sequential_case()

    # Threading Case
    threading_time = threading_case()

    # Multiprocessing Case
    multiprocessing_time = multiprocessing_case()

    # Performance Analysis
    num_processors = 4  
    speedup_threading = calculate_speedup(sequential_time, threading_time)
    efficiency_threading = calculate_efficiency(speedup_threading, num_processors)

    speedup_multiprocessing = calculate_speedup(sequential_time, multiprocessing_time)
    efficiency_multiprocessing = calculate_efficiency(speedup_multiprocessing, num_processors)

    # Amdhal’s Law and Gustafson’s Law
    p = 0.9  
    amdhal_speedup_threading = amdhal_law_speedup(p, num_processors)
    amdhal_speedup_multiprocessing = amdhal_law_speedup(p, num_processors)

    gustafson_speedup_threading = gustafson_law_speedup(p, num_processors)
    gustafson_speedup_multiprocessing = gustafson_law_speedup(p, num_processors)

    print("\nPerformance Analysis:")
    print(f"Speedup (Threading): {speedup_threading}")
    print(f"Efficiency (Threading): {efficiency_threading}")
    print(f"Amdhal’s Speedup (Threading): {amdhal_speedup_threading}")
    print(f"Gustafson’s Speedup (Threading): {gustafson_speedup_threading}")

    print(f"Speedup (Multiprocessing): {speedup_multiprocessing}")
    print(f"Efficiency (Multiprocessing): {efficiency_multiprocessing}")
    print(f"Amdhal’s Speedup (Multiprocessing): {amdhal_speedup_multiprocessing}")
    print(f"Gustafson’s Speedup (Multiprocessing): {gustafson_speedup_multiprocessing}")
    print("Threading is ideal for tasks that are I/O-bound,For simple parallel tasks that are I/O-bound or light in CPU computation, threading can outperform multiprocessing")
    print("Multiprocessing creates separate processes with their own memory space, which allows them to bypass the GIL.")
    print("Challenges")
    print("Python’s GIL restricts true parallel execution in threads for CPU-bound tasks, which limits the performance of threading in these cases.")
    

if __name__ == "__main__":
    main()
