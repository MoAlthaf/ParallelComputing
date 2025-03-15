import time
import random
import multiprocessing as mp
from multiprocessing import Semaphore

class ConnectionPool:
    def __init__(self, pool_size):
        """
        Initialize the connection pool with a given size.
        
        Args:
            pool_size: The number of connections in the pool
        """
        self.pool_size = pool_size
        self.semaphore = mp.Semaphore(pool_size)
        self.connections = [f"Connection-{i}" for i in range(pool_size)]
        self.connection_lock = mp.Lock()
        self.available = [True] * pool_size
    
    def get_connection(self):
        """
        Get a connection from the pool.
        This will block if no connections are available.
        
        Returns:
            A connection identifier
        """
        # Wait for an available connection
        self.semaphore.acquire()
        
        # Find an available connection
        with self.connection_lock:
            for i, is_available in enumerate(self.available):
                if is_available:
                    self.available[i] = False
                    return self.connections[i]
    
    def release_connection(self, connection):
        """
        Release a connection back to the pool.
        
        Args:
            connection: The connection to release
        """
        # Find the connection index
        with self.connection_lock:
            for i, conn in enumerate(self.connections):
                if conn == connection:
                    self.available[i] = True
                    break
        
        # Signal that a connection is available
        self.semaphore.release()

def access_database(process_id, pool):
    """
    Simulate a database operation using a connection from the pool.
    
    Args:
        process_id: The ID of the process
        pool: The connection pool
    """
    print(f"Process {process_id}: Waiting for connection...")
    
    # Get a connection from the pool
    connection = pool.get_connection()
    
    print(f"Process {process_id}: Acquired {connection}")
    
    # Simulate some work
    work_time = random.uniform(0.5, 2.0)
    time.sleep(work_time)
    
    # Release the connection
    print(f"Process {process_id}: Releasing {connection} after {work_time:.2f}s")
    pool.release_connection(connection)

def run_simulation(num_processes=10, pool_size=3):
    """
    Run a simulation with multiple processes accessing a connection pool.
    
    Args:
        num_processes: The number of processes to create
        pool_size: The size of the connection pool
    """
    print(f"\nRunning simulation with {num_processes} processes and {pool_size} connections")
    
    # Create a shared connection pool
    pool = ConnectionPool(pool_size)
    
    # Create and start processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=access_database, args=(i, pool))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All processes completed")

if __name__ == "__main__":
    # Run the simulation
    run_simulation(num_processes=10, pool_size=3)