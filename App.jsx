import { useState, useEffect, useCallback, useRef } from "react";

const GRID_SIZE = 51;
const NUM_WORLDS = 30;
const BLOCK_PROB = 0.3;

function generateMaze(seed) {
  // Seeded pseudo-random number generator
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  const rows = GRID_SIZE, cols = GRID_SIZE;
  const grid = Array.from({ length: rows }, () => new Array(cols).fill(null)); // null = unvisited
  const DIRS = [[-2,0],[2,0],[0,-2],[0,2]];

  const inBounds = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols;

  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  // Initialize all as blocked
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      grid[r][c] = 1; // 1 = blocked

  const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));

  const startR = Math.floor(rand() * rows);
  const startC = Math.floor(rand() * cols);

  const stack = [[startR, startC]];
  visited[startR][startC] = true;
  grid[startR][startC] = 0; // unblocked

  while (stack.length > 0) {
    const [r, c] = stack[stack.length - 1];
    const dirs = shuffle([...DIRS]);
    let found = false;

    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      const mr = r + dr / 2, mc = c + dc / 2;
      if (inBounds(nr, nc) && !visited[nr][nc]) {
        visited[nr][nc] = true;
        // carve wall between
        grid[Math.floor(mr)][Math.floor(mc)] = 0;
        if (rand() < BLOCK_PROB) {
          grid[nr][nc] = 1; // blocked
        } else {
          grid[nr][nc] = 0; // unblocked
          stack.push([nr, nc]);
        }
        found = true;
        break;
      }
    }

    if (!found) stack.pop();
  }

  // Handle unvisited nodes (isolated components)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!visited[r][c]) {
        visited[r][c] = true;
        grid[r][c] = rand() < BLOCK_PROB ? 1 : 0;
      }
    }
  }

  return grid;
}

function placeAgentAndTarget(grid, rand) {
  const unblocked = [];
  for (let r = 0; r < GRID_SIZE; r++)
    for (let c = 0; c < GRID_SIZE; c++)
      if (grid[r][c] === 0) unblocked.push([r, c]);

  if (unblocked.length < 2) return { agent: [0,0], target: [GRID_SIZE-1, GRID_SIZE-1] };

  const ai = Math.floor(rand() * unblocked.length);
  let ti = Math.floor(rand() * (unblocked.length - 1));
  if (ti >= ai) ti++;

  return { agent: unblocked[ai], target: unblocked[ti] };
}

// Generate all 30 worlds once
function generateAllWorlds() {
  return Array.from({ length: NUM_WORLDS }, (_, i) => {
    let s = (i + 1) * 123456789;
    const rand = () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
    const grid = generateMaze(i * 987654321 + 42);
    const { agent, target } = placeAgentAndTarget(grid, rand);
    const blocked = grid.flat().filter(v => v === 1).length;
    const total = GRID_SIZE * GRID_SIZE;
    return { id: i + 1, grid, agent, target, blockRate: ((blocked / total) * 100).toFixed(1) };
  });
}

const WORLDS = generateAllWorlds();

export default function GridWorldVisualizer() {
  const [selected, setSelected] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [showStats, setShowStats] = useState(false);
  const canvasRef = useRef(null);

  const world = WORLDS[selected];

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { grid, agent, target } = world;
    const cellSize = Math.max(2, Math.floor((canvas.width / GRID_SIZE)));

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const x = c * cellSize, y = r * cellSize;
        if (grid[r][c] === 1) {
          ctx.fillStyle = "#1a1a2e";
        } else {
          ctx.fillStyle = "#e8e8f0";
        }
        ctx.fillRect(x, y, cellSize, cellSize);
      }
    }

    // Grid lines (only if cellSize large enough)
    if (cellSize >= 6) {
      ctx.strokeStyle = "rgba(150,150,180,0.15)";
      ctx.lineWidth = 0.5;
      for (let r = 0; r <= GRID_SIZE; r++) {
        ctx.beginPath(); ctx.moveTo(0, r * cellSize); ctx.lineTo(GRID_SIZE * cellSize, r * cellSize); ctx.stroke();
      }
      for (let c = 0; c <= GRID_SIZE; c++) {
        ctx.beginPath(); ctx.moveTo(c * cellSize, 0); ctx.lineTo(c * cellSize, GRID_SIZE * cellSize); ctx.stroke();
      }
    }

    // Agent
    const [ar, ac] = agent;
    ctx.fillStyle = "#00d4aa";
    ctx.beginPath();
    ctx.arc(ac * cellSize + cellSize / 2, ar * cellSize + cellSize / 2, cellSize * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = cellSize > 4 ? 1.5 : 0.5;
    ctx.stroke();

    // Target
    const [tr, tc] = target;
    ctx.fillStyle = "#ff6b6b";
    ctx.beginPath();
    ctx.arc(tc * cellSize + cellSize / 2, tr * cellSize + cellSize / 2, cellSize * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = cellSize > 4 ? 1.5 : 0.5;
    ctx.stroke();

    // Star on target
    if (cellSize >= 6) {
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${Math.max(8, cellSize * 0.6)}px serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("★", tc * cellSize + cellSize / 2, tr * cellSize + cellSize / 2);
      ctx.fillStyle = "#1a1a2e";
      ctx.font = `bold ${Math.max(6, cellSize * 0.5)}px monospace`;
      ctx.fillText("A", ac * cellSize + cellSize / 2, ar * cellSize + cellSize / 2);
    }
  }, [world]);

  useEffect(() => { draw(); }, [draw, zoom]);

  const canvasSize = Math.min(520, Math.floor(zoom * 520));
  const cellSize = Math.floor(canvasSize / GRID_SIZE);
  const actualSize = cellSize * GRID_SIZE;

  const unblocked = world.grid.flat().filter(v => v === 0).length;
  const manhattan = Math.abs(world.agent[0] - world.target[0]) + Math.abs(world.agent[1] - world.target[1]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)",
      fontFamily: "'Courier New', monospace",
      color: "#e0e0f0",
      padding: "24px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "20px"
    }}>
      {/* Header */}
      <div style={{ textAlign: "center" }}>
        <div style={{
          fontSize: "11px", letterSpacing: "6px", color: "#00d4aa",
          textTransform: "uppercase", marginBottom: "6px"
        }}>
          ◈ GRIDWORLD NAVIGATION LAB ◈
        </div>
        <h1 style={{
          margin: 0, fontSize: "28px", fontWeight: 900,
          background: "linear-gradient(90deg, #00d4aa, #7b68ee, #ff6b6b)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "2px"
        }}>
          51 × 51 MAZE ENVIRONMENTS
        </h1>
        <div style={{ fontSize: "11px", color: "#8888aa", marginTop: "4px" }}>
          30 DFS-generated gridworlds · 30% block probability
        </div>
      </div>

      {/* World Selector Grid */}
      <div style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(123,104,238,0.3)",
        borderRadius: "12px",
        padding: "16px",
        width: "100%",
        maxWidth: "580px"
      }}>
        <div style={{ fontSize: "10px", letterSpacing: "3px", color: "#7b68ee", marginBottom: "10px" }}>
          SELECT ENVIRONMENT
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
          {WORLDS.map((w, i) => (
            <button key={i} onClick={() => setSelected(i)} style={{
              width: "36px", height: "36px", border: "none", borderRadius: "6px",
              cursor: "pointer", fontSize: "11px", fontFamily: "monospace", fontWeight: 700,
              background: selected === i
                ? "linear-gradient(135deg, #00d4aa, #7b68ee)"
                : "rgba(255,255,255,0.06)",
              color: selected === i ? "#fff" : "#8888bb",
              transition: "all 0.15s",
              boxShadow: selected === i ? "0 0 12px rgba(0,212,170,0.4)" : "none",
              transform: selected === i ? "scale(1.1)" : "scale(1)"
            }}>
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: "flex", gap: "20px", alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center" }}>

        {/* Canvas */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(0,212,170,0.25)",
          borderRadius: "12px",
          padding: "16px",
          boxShadow: "0 0 40px rgba(0,212,170,0.08)"
        }}>
          <div style={{ fontSize: "10px", letterSpacing: "3px", color: "#00d4aa", marginBottom: "10px" }}>
            WORLD #{world.id} · {GRID_SIZE}×{GRID_SIZE}
          </div>
          <canvas
            ref={canvasRef}
            width={actualSize}
            height={actualSize}
            style={{ display: "block", borderRadius: "6px", imageRendering: "pixelated" }}
          />
          {/* Zoom */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "12px" }}>
            <span style={{ fontSize: "10px", color: "#8888aa" }}>ZOOM</span>
            <input type="range" min="0.5" max="2" step="0.1" value={zoom}
              onChange={e => setZoom(parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: "#00d4aa" }} />
            <span style={{ fontSize: "10px", color: "#00d4aa", width: "32px" }}>{zoom.toFixed(1)}×</span>
          </div>
          {/* Legend */}
          <div style={{ display: "flex", gap: "16px", marginTop: "10px", fontSize: "10px", color: "#8888aa" }}>
            <span>
              <span style={{ display: "inline-block", width: "10px", height: "10px", background: "#1a1a2e", border: "1px solid #444", borderRadius: "2px", marginRight: "4px", verticalAlign: "middle" }} />
              BLOCKED
            </span>
            <span>
              <span style={{ display: "inline-block", width: "10px", height: "10px", background: "#e8e8f0", borderRadius: "2px", marginRight: "4px", verticalAlign: "middle" }} />
              FREE
            </span>
            <span>
              <span style={{ display: "inline-block", width: "10px", height: "10px", background: "#00d4aa", borderRadius: "50%", marginRight: "4px", verticalAlign: "middle" }} />
              AGENT (A)
            </span>
            <span>
              <span style={{ display: "inline-block", width: "10px", height: "10px", background: "#ff6b6b", borderRadius: "50%", marginRight: "4px", verticalAlign: "middle" }} />
              TARGET (★)
            </span>
          </div>
        </div>

        {/* Stats Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px", minWidth: "200px" }}>
          {/* World Stats */}
          <div style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(123,104,238,0.3)",
            borderRadius: "12px",
            padding: "16px"
          }}>
            <div style={{ fontSize: "10px", letterSpacing: "3px", color: "#7b68ee", marginBottom: "12px" }}>
              STATISTICS
            </div>
            {[
              ["Grid Size", `${GRID_SIZE} × ${GRID_SIZE}`],
              ["Total Cells", (GRID_SIZE * GRID_SIZE).toLocaleString()],
              ["Free Cells", unblocked.toLocaleString()],
              ["Blocked Cells", (GRID_SIZE * GRID_SIZE - unblocked).toLocaleString()],
              ["Block Rate", `${world.blockRate}%`],
              ["Agent Pos", `(${world.agent[0]}, ${world.agent[1]})`],
              ["Target Pos", `(${world.target[0]}, ${world.target[1]})`],
              ["Manhattan Dist", manhattan],
            ].map(([label, val]) => (
              <div key={label} style={{
                display: "flex", justifyContent: "space-between",
                borderBottom: "1px solid rgba(255,255,255,0.05)",
                padding: "5px 0", fontSize: "12px"
              }}>
                <span style={{ color: "#8888aa" }}>{label}</span>
                <span style={{ color: "#e0e0f0", fontWeight: 700 }}>{val}</span>
              </div>
            ))}
          </div>

          {/* All worlds summary */}
          <div style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(0,212,170,0.2)",
            borderRadius: "12px",
            padding: "16px"
          }}>
            <div style={{ fontSize: "10px", letterSpacing: "3px", color: "#00d4aa", marginBottom: "12px" }}>
              ALL 30 WORLDS
            </div>
            {[
              ["Avg Block Rate", (WORLDS.reduce((a,w) => a + parseFloat(w.blockRate), 0) / NUM_WORLDS).toFixed(1) + "%"],
              ["Min Block Rate", Math.min(...WORLDS.map(w => parseFloat(w.blockRate))).toFixed(1) + "%"],
              ["Max Block Rate", Math.max(...WORLDS.map(w => parseFloat(w.blockRate))).toFixed(1) + "%"],
            ].map(([label, val]) => (
              <div key={label} style={{
                display: "flex", justifyContent: "space-between",
                borderBottom: "1px solid rgba(255,255,255,0.05)",
                padding: "5px 0", fontSize: "12px"
              }}>
                <span style={{ color: "#8888aa" }}>{label}</span>
                <span style={{ color: "#e0e0f0", fontWeight: 700 }}>{val}</span>
              </div>
            ))}

            <button onClick={() => setShowStats(s => !s)} style={{
              marginTop: "12px", width: "100%", padding: "8px",
              background: "linear-gradient(135deg, rgba(0,212,170,0.2), rgba(123,104,238,0.2))",
              border: "1px solid rgba(0,212,170,0.4)",
              borderRadius: "6px", color: "#00d4aa",
              cursor: "pointer", fontSize: "10px", letterSpacing: "2px",
              fontFamily: "monospace"
            }}>
              {showStats ? "HIDE" : "SHOW"} ALL BLOCK RATES
            </button>

            {showStats && (
              <div style={{
                marginTop: "10px", maxHeight: "200px", overflowY: "auto",
                fontSize: "11px"
              }}>
                {WORLDS.map((w, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between",
                    padding: "3px 0",
                    color: i === selected ? "#00d4aa" : "#8888aa"
                  }}>
                    <span>World {w.id}</span>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <div style={{
                        width: `${parseFloat(w.blockRate)}px`, height: "4px",
                        background: i === selected ? "#00d4aa" : "#7b68ee",
                        borderRadius: "2px"
                      }} />
                      <span>{w.blockRate}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Navigation buttons */}
          <div style={{ display: "flex", gap: "8px" }}>
            <button onClick={() => setSelected(s => Math.max(0, s - 1))} style={{
              flex: 1, padding: "10px", border: "1px solid rgba(123,104,238,0.4)",
              borderRadius: "8px", background: "rgba(123,104,238,0.1)",
              color: "#7b68ee", cursor: "pointer", fontSize: "16px",
              fontFamily: "monospace"
            }}>◀</button>
            <button onClick={() => setSelected(s => Math.min(NUM_WORLDS - 1, s + 1))} style={{
              flex: 1, padding: "10px", border: "1px solid rgba(0,212,170,0.4)",
              borderRadius: "8px", background: "rgba(0,212,170,0.1)",
              color: "#00d4aa", cursor: "pointer", fontSize: "16px",
              fontFamily: "monospace"
            }}>▶</button>
          </div>
        </div>
      </div>

      <div style={{ fontSize: "10px", color: "#55556a", letterSpacing: "1px", textAlign: "center" }}>
        DFS maze generation · 30% block probability · random tie-breaking · 51×51 cells
      </div>
    </div>
  );
}
