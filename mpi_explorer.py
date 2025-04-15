from mpi4py import MPI
from src.maze import create_maze
from src.explorer import Explorer
from time import perf_counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

maze_type = "static"
width = 50
height = 50


maze = create_maze(width, height, maze_type)
explorer = Explorer(maze, visualize=False)

start = perf_counter()
time_taken, moves = explorer.solve()
end = perf_counter()

stats = {
    'rank': rank,
    'time': time_taken,
    'moves': len(moves),
    'backtracks': explorer.backtrack_count,
    'speed': len(moves) / time_taken if time_taken > 0 else 0
}

# Gathering results at root process
all_stats = comm.gather(stats, root=0)

if rank == 0:
    print("\n=== MPI Explorer Results ===")
    for s in all_stats:
        print(f"Explorer {s['rank']}: {s['moves']} moves, {s['time']:.3f}s, "
              f"{s['backtracks']} backtracks, {s['speed']:.2f} moves/sec")

    best = min(all_stats, key=lambda x: (x['moves'], x['time'], x['backtracks']))
    print("Best Explorer (MPI):")
    print(f"ID={best['rank']} | Moves={best['moves']} | Time={best['time']:.3f}s | "
          f"Backtracks={best['backtracks']} | Speed={best['speed']:.2f} moves/sec")
