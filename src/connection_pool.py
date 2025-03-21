import multiprocessing
import time
import random

class ConnectionPool:
    def __init__(self, size):
        self.size = size
        self.semaphore = multiprocessing.Semaphore(size)
        self.connections = [f"Connection-{i}" for i in range(size)]

    def get_connection(self):
        self.semaphore.acquire()
        return self.connections.pop()

    def release_connection(self, conn):
        self.connections.append(conn)
        self.semaphore.release()

def access_database(pool):
    conn = pool.get_connection()
    print(f"Process {multiprocessing.current_process().name} acquired {conn}")
    time.sleep(random.uniform(1, 3))  # Simulate database operation
    print(f"Process {multiprocessing.current_process().name} released {conn}")
    pool.release_connection(conn)
      
