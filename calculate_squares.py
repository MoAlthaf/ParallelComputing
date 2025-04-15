from mpi4py import MPI
import numpy as np
import time

def square(start, end):
    return np.square(np.arange(start, end))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = int(2e8)  

    chunk_size = N // size
    start = rank * chunk_size + 1
    end = start + chunk_size if rank != size - 1 else N + 1

    comm.Barrier()
    start_time = MPI.Wtime()

    partial_squares = square(start, end)


    gathered = comm.gather(partial_squares, root=0)

    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        all_squares = np.concatenate(gathered)
        print(f"Total elements: {len(all_squares)}")
        print(f"Last square: {all_squares[-1]}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
