from collections import deque

class BFSExplorer:
    def __init__(self, maze):
        self.maze = maze
        self.start = maze.start_pos
        self.end = maze.end_pos
        self.path = []
        self.visited = set()
        self.move_count = 0

    def solve(self):
        queue = deque()
        queue.append((self.start, [self.start]))
        self.visited.add(self.start)

        while queue:
            current, path = queue.popleft()
            x, y = current

            if current == self.end:
                self.path = path
                self.move_count = len(path)
                return

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.maze.width and
                    0 <= ny < self.maze.height and
                    self.maze.grid[ny][nx] == 0 and
                    (nx, ny) not in self.visited):

                    self.visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
