# parallel_explorers.py

import time
from src.maze import create_maze
from src.explorer import Explorer
import multiprocessing

def run_explorer_instance(index, maze_type="static", width=50, height=50):
    maze = create_maze(width, height, maze_type)
    explorer = Explorer(maze, visualize=False)

    #start_time = time.time()
    time_taken, moves = explorer.solve()
    #end_time = time.time()


    return {
        "id": index,
        "time_taken": time_taken,
        "moves": len(moves),
        "backtracks": explorer.backtrack_count,
        "avg_speed": len(moves) / time_taken
    }



def main():
    num_explorers = 4  

    print(f"Running {num_explorers} explorers in parallel...\n")
    with multiprocessing.Pool(processes=num_explorers) as pool:
        results = pool.starmap(run_explorer_instance, [(i, "static", 50, 50) for i in range(num_explorers)])

    
    print("=== Explorer Results ===")
    for r in results:
        print(f"Explorer {r['id']}: {r['moves']} moves, {r['time_taken']:.2f}s, "
              f"{r['backtracks']} backtracks, {r['avg_speed']:.2f} moves/sec")

   
    best = min(results, key=lambda x: (x['moves'], x['time_taken'], x['backtracks']))
    print("Best Explorer:")
    print(f"ID={best['id']} | Moves={best['moves']} | Time={best['time_taken']:.2f}s | "
          f"Backtracks={best['backtracks']} | Speed={best['avg_speed']:.2f} moves/sec")

if __name__ == "__main__":
    main()
