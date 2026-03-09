"""
Part 2 - Repeated Forward A*
=============================
Implements Repeated Forward A* for agent navigation in unknown gridworlds.

Features:
- Full A* implementation with Manhattan distance heuristic
- Tie-breaking: prefer larger g-values (as specified)
- Lazy initialization using counter/search stamps
- Explicit closed list (no full-grid iteration)
- Step-by-step visualization with matplotlib
- Loads gridworlds from Part 0 JSON or generates fresh ones

Requirements:
    pip install matplotlib numpy

Run:
   cd ~/Desktop/AI/Project1
python3 repeated_forward_astar.py
"""

import json
import os
import random
import heapq  # priority queue for A* open list
import numpy as np # grid arrays
import matplotlib.pyplot as plt # visualization
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE  = 51 # 51×51 grid
NUM_WORLDS = 30
BLOCK_PROB = 0.30
DIRS4      = [(-1,0),(1,0),(0,-1),(0,1)] # North, South, West, East
JSON_PATH  = os.path.join("gridworlds", "gridworlds_30.json")

INF = float('inf') # represents "no path found yet"

# ═══════════════════════════════════════════════════════════════════════════════
# MAZE GENERATION (same as Part 0)
# Creates a 51×51 grid using DFS
# 30% cells blocked, 70% unblocked
# Same seed = same maze every time
# ═══════════════════════════════════════════════════════════════════════════════
def generate_maze(seed):
    rng  = random.Random(seed)
    rows = cols = GRID_SIZE
    grid    = np.ones((rows, cols), dtype=np.int8)
    visited = np.zeros((rows, cols), dtype=bool)

    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def unvisited_neighbors(r, c):
        dirs = DIRS4[:]
        rng.shuffle(dirs)
        return [(r+dr, c+dc) for dr,dc in dirs
                if in_bounds(r+dr, c+dc) and not visited[r+dr][c+dc]]

    all_cells = [(r,c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)
    unvisited_set = set(all_cells)

    sr, sc = rng.randint(0,rows-1), rng.randint(0,cols-1)
    visited[sr][sc] = True
    grid[sr][sc]    = 0
    unvisited_set.discard((sr,sc))
    stack = [(sr,sc)]

    while unvisited_set:
        if not stack:
            nr, nc = next(iter(unvisited_set))
            visited[nr][nc] = True
            unvisited_set.discard((nr,nc))
            grid[nr][nc] = 1 if rng.random() < BLOCK_PROB else 0
            if grid[nr][nc] == 0: stack.append((nr,nc))
            continue
        r, c = stack[-1]
        neighbors = unvisited_neighbors(r, c)
        if not neighbors:
            stack.pop()
        else:
            nr, nc = neighbors[0]
            visited[nr][nc] = True
            unvisited_set.discard((nr,nc))
            if rng.random() < BLOCK_PROB:
                grid[nr][nc] = 1
            else:
                grid[nr][nc] = 0
                stack.append((nr,nc))
    return grid

def place_agent_target(grid, rng):
    free = list(zip(*np.where(grid == 0)))
    if len(free) < 2: return (0,0),(GRID_SIZE-1,GRID_SIZE-1)
    ai = rng.randint(0, len(free)-1)
    ti = rng.randint(0, len(free)-2)
    if ti >= ai: ti += 1
    return tuple(map(int,free[ai])), tuple(map(int,free[ti]))

def build_world(world_id, seed):
    grid = generate_maze(seed)
    rng  = random.Random(seed ^ 0xDEADBEEF)
    agent, target = place_agent_target(grid, rng)
    return {"id": world_id, "seed": seed, "grid": grid,
            "agent": agent, "target": target}

def load_or_generate_worlds():
    if os.path.exists(JSON_PATH):
        print(f"Loading worlds from {JSON_PATH} ...")
        with open(JSON_PATH) as f:
            data = json.load(f)
        worlds = []
        for d in data:
            d["grid"]   = np.array(d["grid"], dtype=np.int8)
            d["agent"]  = tuple(d["agent"])
            d["target"] = tuple(d["target"])
            worlds.append(d)
        print(f"  Loaded {len(worlds)} worlds.")
        return worlds
    else:
        print("No saved worlds found. Generating 30 new worlds...")
        worlds = []
        for i in range(NUM_WORLDS):
            seed = (i+1) * 0x9E3779B9 & 0xFFFFFFFF
            worlds.append(build_world(i+1, seed))
        print("  Done.")
        return worlds


# ═══════════════════════════════════════════════════════════════════════════════
# REPEATED FORWARD A* IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def manhattan(r1, c1, r2, c2):
    return abs(r1-r2) + abs(c1-c2)

class RepeatedForwardAStar:
    """
    Repeated Forward A* as described in the project spec.

    Key implementation details:
    - Searches from current agent position → target (forward)
    - Manhattan distance heuristic (consistent in 4-dir gridworld)
    - Tie-breaking: among equal f-values, prefer LARGER g-values
    - Lazy initialization: g/search values initialized on first encounter
    - Explicit closed list (linked list / set, no full-grid scan)
    - Agent observes 4 neighbors at each step → updates known_blocked map
    - Freespace assumption: unknown cells treated as unblocked
    """

    def __init__(self, true_grid, agent, target):
        self.true_grid   = true_grid          # the real world (agent can't see all)
        self.target      = target
        self.agent       = agent
        self.rows = self.cols = GRID_SIZE

        # Agent's knowledge: only cells it has observed
        # None = unknown (assumed unblocked), 1 = known blocked, 0 = known unblocked
        self.known = np.full((self.rows, self.cols), -1, dtype=np.int8)

        # A* state arrays (lazy init via counter stamps)
        self.g      = np.full((self.rows, self.cols), INF) # g-values: all start as infinity
        self.h      = np.zeros((self.rows, self.cols)) # h-values: Manhattan distance, pre-computed once
        self.search = np.zeros((self.rows, self.cols), dtype=np.int32) # counter stamps for lazy initialization
        self.parent = {}   # (r,c) → (pr,pc) # tree pointers — used to reconstruct path

        # Pre-compute h-values (Manhattan to target) — fixed, never changes
        tr, tc = target
        for r in range(self.rows):
            for c in range(self.cols):
                self.h[r][c] = manhattan(r, c, tr, tc)

        self.counter = 0   # which search number we're on (1st, 2nd, 3rd...)

        # Statistics
        self.total_moves    = 0
        self.num_searches   = 0
        self.total_expanded = 0

        # For visualization: store each A* search result
        self.history = []   # list of step dicts

        # Observe initial position neighbors
        self._observe(agent)

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _observe(self, pos):
        """Agent observes its 4 neighbors — updates known map."""
        r, c = pos
        self.known[r][c] = 0  # current cell = unblocked (I'm standing here)
        for dr, dc in DIRS4:
            nr, nc = r+dr, c+dc
            if self._in_bounds(nr, nc):
                self.known[nr][nc] = self.true_grid[nr][nc] # peek at reality

    def _is_blocked(self, r, c):
        """Is cell (r,c) known to be blocked? Unknown = assumed unblocked."""
        return self.known[r][c] == 1

    def _compute_path(self):
        """
        Run one A* search from self.agent to self.target.
        Returns path as list of (r,c) from agent to target,
        or None if no path exists.

        Tie-breaking: prefer larger g-value (closer to goal).
        Uses lazy initialization with counter stamps.
        """
        self.counter += 1
        self.num_searches += 1
        counter = self.counter

        start  = self.agent
        goal   = self.target
        sr, sc = start
        gr, gc = goal

        # Initialize start state
        self.search[sr][sc] = counter
        self.g[sr][sc]      = 0

        # Initialize goal state (needed for termination check)
        if self.search[gr][gc] != counter:
            self.search[gr][gc] = counter
            self.g[gr][gc]      = INF

        # Open list: (f, -g, r, c)  — min-heap
        # Negative g for tie-breaking: larger g = smaller -g = higher priority
        open_heap = []
        f_start = self.h[sr][sc]
        heapq.heappush(open_heap, (f_start, -0, sr, sc))

        # Closed list (explicit set)
        closed = set()

        expanded_cells = []   # for visualization

        while open_heap:
            # Pop cell with lowest f (best estimate)
            f, neg_g, r, c = heapq.heappop(open_heap)

            # Skip if already expanded (closed)
            if (r, c) in closed:
                continue

            g_cur = self.g[r][c]

            # Termination: stop if best f in open ≥ g(goal)
            if f > self.g[gr][gc]:
                break

            # Expand this cell
            closed.add((r, c))
            expanded_cells.append((r, c))
            self.total_expanded += 1

            # Explore neighbors
            for dr, dc in DIRS4:
                nr, nc = r+dr, c+dc
                if not self._in_bounds(nr, nc):
                    continue
                if self._is_blocked(nr, nc):
                    continue

                # Lazy initialization
                if self.search[nr][nc] != counter:
                    self.search[nr][nc] = counter
                    self.g[nr][nc]      = INF

                new_g = g_cur + 1   # all action costs = 1
                if new_g < self.g[nr][nc]:
                    self.g[nr][nc]      = new_g
                    self.parent[(nr,nc)] = (r, c)
                    f_new = new_g + self.h[nr][nc]
                    heapq.heappush(open_heap, (f_new, -new_g, nr, nc))

        # Check if goal was reached
        if self.g[gr][gc] == INF:
            return None, expanded_cells

        # Reconstruct path by following parent pointers
        path = []
        cur  = goal
        while cur != start:
            path.append(cur)
            cur = self.parent.get(cur)
            if cur is None:
                return None, expanded_cells
        path.append(start)
        path.reverse()
        return path, expanded_cells

    def run(self):
        """
        Main loop of Repeated Forward A*.
        Returns: (success, trajectory, history)
        """
        trajectory = [self.agent]

        while self.agent != self.target:
            # Search for shortest presumed-unblocked path
            path, expanded = self._compute_path()

            if path is None:
                # No path exists — target unreachable
                self.history.append({
                    "type":      "search_failed",
                    "agent":     self.agent,
                    "known":     self.known.copy(),
                    "expanded":  expanded,
                    "path":      None,
                    "trajectory": list(trajectory),
                })
                return False, trajectory, self.history

            # Record this search in history
            self.history.append({
                "type":      "search",
                "agent":     self.agent,
                "known":     self.known.copy(),
                "expanded":  list(expanded),
                "path":      list(path),
                "trajectory": list(trajectory),
                "search_num": self.num_searches,
            })

            # Follow the path until blocked or target reached
            for i in range(1, len(path)):
                next_cell = path[i]
                nr, nc    = next_cell

                # Observe before moving
                self._observe(next_cell)

                # Check if this step is blocked (shouldn't be, but check ahead)
                if self._is_blocked(nr, nc):
                    break   # replan

                # Move agent
                self.agent = next_cell
                self.total_moves += 1
                trajectory.append(self.agent)

                # Record movement step
                self.history.append({
                    "type":      "move",
                    "agent":     self.agent,
                    "known":     self.known.copy(),
                    "expanded":  [],
                    "path":      path[i:],
                    "trajectory": list(trajectory),
                })

                if self.agent == self.target:
                    return True, trajectory, self.history

                # Check if remaining path is now blocked
                path_blocked = False
                for j in range(i+1, len(path)):
                    pr, pc = path[j]
                    if self._is_blocked(pr, pc):
                        path_blocked = True
                        break
                if path_blocked:
                    break   # replan

        return True, trajectory, self.history


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Color map: 0=free(white), 1=blocked(dark)
CMAP_GRID = ListedColormap(["#e8eaf6", "#1a1a2e"])

class AStarVisualizer:
    """
    Interactive step-by-step visualizer for Repeated Forward A*.
    Shows:
      - True gridworld (left)
      - Agent's known map with A* overlays (right)
      - Expanded cells (grey), current path (blue), trajectory (green)
    """

    def __init__(self, world, result):
        self.world     = world
        self.success   = result["success"]
        self.trajectory= result["trajectory"]
        self.history   = result["history"]
        self.stats     = result["stats"]
        self.step      = 0

        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 7), facecolor="#0d0d18")
        self.fig.canvas.manager.set_window_title(
            f"Repeated Forward A* — World #{self.world['id']}")

        # Left: true grid
        self.ax_true = self.fig.add_axes([0.02, 0.12, 0.42, 0.82])
        # Right: agent knowledge
        self.ax_know = self.fig.add_axes([0.52, 0.12, 0.42, 0.82])

        for ax in [self.ax_true, self.ax_know]:
            ax.set_facecolor("#0d0d18")
            ax.set_xticks([]); ax.set_yticks([])

        # Buttons
        ax_prev  = self.fig.add_axes([0.02,  0.02, 0.08, 0.06])
        ax_next  = self.fig.add_axes([0.11,  0.02, 0.08, 0.06])
        ax_start = self.fig.add_axes([0.22,  0.02, 0.10, 0.06])
        ax_end   = self.fig.add_axes([0.33,  0.02, 0.10, 0.06])
        ax_auto  = self.fig.add_axes([0.52,  0.02, 0.12, 0.06])

        def mk_btn(ax, label):
            b = Button(ax, label, color="#1a1a2e", hovercolor="#2a2a40")
            b.label.set_color("#e0e0f0")
            b.label.set_fontfamily("monospace")
            b.label.set_fontsize(9)
            return b

        self.btn_prev  = mk_btn(ax_prev,  "◀ PREV")
        self.btn_next  = mk_btn(ax_next,  "NEXT ▶")
        self.btn_start = mk_btn(ax_start, "|◀ START")
        self.btn_end   = mk_btn(ax_end,   "END ▶|")
        self.btn_auto  = mk_btn(ax_auto,  "▶ AUTO PLAY")

        self.btn_prev.on_clicked(lambda e: self._step(-1))
        self.btn_next.on_clicked(lambda e: self._step(+1))
        self.btn_start.on_clicked(lambda e: self._goto(0))
        self.btn_end.on_clicked(lambda e: self._goto(len(self.history)-1))
        self.btn_auto.on_clicked(self._autoplay)

        self._draw()
        plt.show()

    def _step(self, delta):
        self.step = max(0, min(len(self.history)-1, self.step + delta))
        self._draw()

    def _goto(self, idx):
        self.step = idx
        self._draw()

    def _autoplay(self, event):
        import time
        for i in range(self.step, len(self.history)):
            self.step = i
            self._draw()
            plt.pause(0.15)

    def _draw(self):
        h   = self.history[self.step]
        w   = self.world
        tr, tc = w["target"]

        # ── Left: True Grid ──────────────────────────────────────────────────
        self.ax_true.cla()
        self.ax_true.imshow(w["grid"], cmap=CMAP_GRID, vmin=0, vmax=1,
                            origin="upper", interpolation="nearest")

        # Full trajectory so far
        traj = h["trajectory"]
        if len(traj) > 1:
            rows_t = [p[0] for p in traj]
            cols_t = [p[1] for p in traj]
            self.ax_true.plot(cols_t, rows_t, "-", color="#00e5b0",
                              linewidth=1.5, alpha=0.7, zorder=3)

        # Agent
        ar, ac = h["agent"]
        self.ax_true.plot(ac, ar, "o", color="#00e5b0", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        self.ax_true.text(ac, ar, "A", color="white", fontsize=6,
                          ha="center", va="center", fontweight="bold", zorder=6)
        # Target
        self.ax_true.plot(tc, tr, "o", color="#ff5370", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        self.ax_true.text(tc, tr, "T", color="white", fontsize=6,
                          ha="center", va="center", fontweight="bold", zorder=6)

        self.ax_true.set_title("TRUE GRIDWORLD", color="#7c6af7",
                               fontfamily="monospace", fontsize=10, pad=6)
        self.ax_true.set_xticks([]); self.ax_true.set_yticks([])

        # ── Right: Agent Knowledge Map ───────────────────────────────────────
        self.ax_know.cla()

        # Build display grid:
        # -1 (unknown) → show as white (freespace assumption)
        # 0 (known free) → white
        # 1 (known blocked) → dark
        known = h["known"]
        display = np.where(known == 1, 1, 0).astype(np.float32)

        # Shade unknown cells slightly differently
        unknown_overlay = (known == -1).astype(np.float32) * 0.15

        self.ax_know.imshow(display, cmap=CMAP_GRID, vmin=0, vmax=1,
                            origin="upper", interpolation="nearest", alpha=1.0)

        # Overlay unknown cells in light grey
        unknown_rgb = np.zeros((*display.shape, 4))
        unknown_rgb[known == -1] = [0.7, 0.7, 0.8, 0.3]
        self.ax_know.imshow(unknown_rgb, origin="upper",
                            interpolation="nearest", zorder=1)

        # Expanded cells (closed list) — grey overlay
        if h["expanded"]:
            exp_overlay = np.zeros((*display.shape, 4))
            for er, ec in h["expanded"]:
                exp_overlay[er, ec] = [0.5, 0.5, 0.6, 0.45]
            self.ax_know.imshow(exp_overlay, origin="upper",
                                interpolation="nearest", zorder=2)

        # Planned path — blue line
        if h["path"] and len(h["path"]) > 1:
            pr = [p[0] for p in h["path"]]
            pc = [p[1] for p in h["path"]]
            self.ax_know.plot(pc, pr, "-", color="#448aff",
                              linewidth=2.5, zorder=4, alpha=0.9)
            # Path dots
            self.ax_know.plot(pc, pr, ".", color="#82b1ff",
                              markersize=3, zorder=4)

        # Trajectory so far — green
        if len(traj) > 1:
            rows_t = [p[0] for p in traj]
            cols_t = [p[1] for p in traj]
            self.ax_know.plot(cols_t, rows_t, "-", color="#00e5b0",
                              linewidth=1.5, alpha=0.8, zorder=3)

        # Agent
        self.ax_know.plot(ac, ar, "o", color="#00e5b0", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        self.ax_know.text(ac, ar, "A", color="white", fontsize=6,
                          ha="center", va="center", fontweight="bold", zorder=7)
        # Target
        self.ax_know.plot(tc, tr, "o", color="#ff5370", markersize=10,
                          markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        self.ax_know.text(tc, tr, "T", color="white", fontsize=6,
                          ha="center", va="center", fontweight="bold", zorder=7)

        # Title with step info
        step_type = h["type"]
        snum = h.get("search_num", "")
        label = {
            "search":       f"A* SEARCH #{snum} — {len(h['expanded'])} cells expanded",
            "move":         f"AGENT MOVING — step {self.step}",
            "search_failed":"❌ NO PATH FOUND",
        }.get(step_type, step_type)

        self.ax_know.set_title(f"AGENT KNOWLEDGE MAP  |  {label}",
                               color="#00e5b0", fontfamily="monospace",
                               fontsize=9, pad=6)
        self.ax_know.set_xticks([]); self.ax_know.set_yticks([])

        # Legend
        legend_items = [
            mpatches.Patch(color="#e8eaf6",           label="Free (known/assumed)"),
            mpatches.Patch(color="#1a1a2e",           label="Blocked (known)"),
            mpatches.Patch(color="#b0b0c0", alpha=0.5,label="Expanded (closed list)"),
            mpatches.Patch(color="#448aff",           label="Planned path"),
            mpatches.Patch(color="#00e5b0",           label="Trajectory"),
        ]
        self.ax_know.legend(handles=legend_items, loc="lower left",
                            fontsize=7, facecolor="#1a1a2e",
                            edgecolor="#333", labelcolor="white",
                            framealpha=0.9)

        # Stats footer
        status = "✅ REACHED TARGET" if self.success else "❌ TARGET UNREACHABLE"
        footer = (f"Step {self.step+1}/{len(self.history)}  |  "
                  f"Moves: {self.stats['total_moves']}  |  "
                  f"Searches: {self.stats['num_searches']}  |  "
                  f"Cells expanded: {self.stats['total_expanded']}  |  "
                  f"{status}")
        self.fig.texts.clear()
        self.fig.text(0.5, 0.005, footer, ha="center", color="#6668a0",
                      fontfamily="monospace", fontsize=8)

        self.fig.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD SELECTOR UI
# ═══════════════════════════════════════════════════════════════════════════════

def select_and_run(worlds):
    """Show a simple world-picker, then run A* on selected world."""

    print("\n" + "="*50)
    print("  REPEATED FORWARD A* — Part 2")
    print("="*50)
    print(f"  {len(worlds)} worlds loaded.")
    print("\n  Enter world number (1–30) to run A* on it.")
    print("  Or press Enter for world 1.\n")

    while True:
        try:
            choice = input("  World number: ").strip()
            if choice == "":
                idx = 0
            else:
                idx = int(choice) - 1
            if 0 <= idx < len(worlds):
                break
            print(f"  Please enter a number between 1 and {len(worlds)}.")
        except ValueError:
            print("  Invalid input.")

    world = worlds[idx]
    print(f"\n  Running Repeated Forward A* on World #{world['id']} ...")
    print(f"  Agent: {world['agent']}  →  Target: {world['target']}")
    print(f"  Manhattan distance: "
          f"{abs(world['agent'][0]-world['target'][0]) + abs(world['agent'][1]-world['target'][1])}")

    # Run the algorithm
    solver  = RepeatedForwardAStar(world["grid"], world["agent"], world["target"])
    success, trajectory, history = solver.run()

    stats = {
        "total_moves":    solver.total_moves,
        "num_searches":   solver.num_searches,
        "total_expanded": solver.total_expanded,
    }

    result = {
        "success":    success,
        "trajectory": trajectory,
        "history":    history,
        "stats":      stats,
    }

    print(f"\n  Result: {'✅ TARGET REACHED' if success else '❌ TARGET UNREACHABLE'}")
    print(f"  Total moves:       {stats['total_moves']}")
    print(f"  Number of searches:{stats['num_searches']}")
    print(f"  Total expanded:    {stats['total_expanded']}")
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"\n  Launching visualizer ...")
    print(f"  Use ◀ PREV / NEXT ▶ buttons to step through the search process.")
    print(f"  Use ▶ AUTO PLAY to watch it animate.\n")

    AStarVisualizer(world, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    worlds = load_or_generate_worlds()

    while True:
        select_and_run(worlds)
        again = input("\n  Run on another world? [y/N]: ").strip().lower()
        if again != "y":
            break

    print("\n  Done. Goodbye!")
