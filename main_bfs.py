

import argparse
import time
from src.maze import create_maze
from explorer_bfs import BFSExplorer

def main():
    parser = argparse.ArgumentParser(description="Maze Runner BFS Solver")
    parser.add_argument("--type", choices=["random", "static"], default="static",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    
    args = parser.parse_args()

    # Create the maze
    maze = create_maze(args.width, args.height, args.type)

    # Run BFS solver
    explorer = BFSExplorer(maze)

    print("\nSolving maze using BFS...")
    start_time = time.time()
    explorer.solve()
    end_time = time.time()

    print("\n=== BFS Maze Exploration Statistics ===")
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print(f"Total moves made: {explorer.move_count}")
    print(f"Path length: {len(explorer.path)}")
    print("========================================\n")

if __name__ == "__main__":
    main()
