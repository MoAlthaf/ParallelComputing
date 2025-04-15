

### Question 1
**Explain how the automated maze explorer works.**

---

The automated maze explorer in `explorer.py` uses the **right-hand rule algorithm** to solve mazes. Below is a breakdown of how it works:

---

#### Algorithm Used
- The explorer starts at the maze's entry point, facing **right** by default.
- It tries to turn **right first**, then goes **straight**, then **left**, and finally turns around if no path is found.

---

#### Loop Detection
- The explorer keeps a record of the **last 3 visited cells** using a `deque`.
- If the last 3 positions are identical (`A → A → A`), it assumes a **loop** and initiates **backtracking**.

---

#### Backtracking Strategy
- When stuck, it looks back through its history to find the **last visited cell with multiple possible paths**.
- It follows that path in reverse using a **backtrack stack**.
- Each backtrack is counted (`backtrack_count`).

---

#### Performance Statistics
At the end of the exploration, the explorer prints the following:
- **Total time taken**: Duration of solving the maze
- **Total moves**: Steps taken (including forward and backtracks)
- **Number of backtrack operations**
- **Average moves per second**

---

#### Verification:
- Read the code and ran the program.
