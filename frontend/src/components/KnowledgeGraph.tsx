import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getVideoGraph } from "../api/client";
import type { GraphData, GraphEdge, GraphNode } from "../types";

interface KnowledgeGraphProps {
  videoId: string;
  onClose: () => void;
}

interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface CanvasSize {
  width: number;
  height: number;
}

interface RelationshipItem extends GraphEdge {
  id: string;
  otherNodeId: string;
}

const DEFAULT_CANVAS: CanvasSize = { width: 640, height: 380 };
const MIN_CANVAS_HEIGHT = 320;
const MAX_CANVAS_HEIGHT = 520;
const TYPE_COLORS: Record<string, string> = {
  person: "#ef4444",
  vehicle: "#3b82f6",
  location: "#22c55e",
  object: "#a855f7",
};

export default function KnowledgeGraph({ videoId, onClose }: KnowledgeGraphProps) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>(DEFAULT_CANVAS);
  const [isDragging, setIsDragging] = useState(false);

  const canvasHostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const animRef = useRef<number>(0);
  const dragRef = useRef<{ nodeId: string; offsetX: number; offsetY: number } | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setSelectedNodeId(null);
    setHoveredNodeId(null);

    getVideoGraph(videoId)
      .then((data) => {
        if (!isMounted) return;
        setGraph(data);
        nodesRef.current = initializeNodes(data.nodes, DEFAULT_CANVAS);
      })
      .catch((error) => {
        console.error("Failed to load knowledge graph:", error);
        if (!isMounted) return;
        setGraph({ nodes: [], edges: [] });
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [videoId]);

  useEffect(() => {
    if (!graph || graph.nodes.length === 0) return;
    if (nodesRef.current.length !== graph.nodes.length) {
      nodesRef.current = initializeNodes(graph.nodes, canvasSize);
    }
  }, [canvasSize, graph]);

  useEffect(() => {
    const host = canvasHostRef.current;
    if (!host) return;

    const updateSize = (widthHint?: number) => {
      const width = Math.max(360, Math.floor(widthHint ?? host.clientWidth ?? DEFAULT_CANVAS.width));
      const height = clamp(Math.round(width * 0.6), MIN_CANVAS_HEIGHT, MAX_CANVAS_HEIGHT);

      setCanvasSize((prev) => {
        if (prev.width === width && prev.height === height) {
          return prev;
        }

        if (nodesRef.current.length > 0 && prev.width > 0 && prev.height > 0) {
          const scaleX = width / prev.width;
          const scaleY = height / prev.height;
          nodesRef.current = nodesRef.current.map((node) => ({
            ...node,
            x: clamp(node.x * scaleX, 28, width - 28),
            y: clamp(node.y * scaleY, 28, height - 28),
          }));
        }

        return { width, height };
      });
    };

    updateSize();

    if (typeof ResizeObserver === "undefined") {
      const handleResize = () => updateSize();
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      updateSize(entry?.contentRect.width);
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!graph || graph.nodes.length === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvasSize.width * dpr);
    canvas.height = Math.floor(canvasSize.height * dpr);
    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const tick = () => {
      const width = canvasSize.width;
      const height = canvasSize.height;
      const nodes = nodesRef.current;
      const edges = graph.edges;
      const activeNodeId = selectedNodeId ?? hoveredNodeId;
      const connectedNodeIds = new Set<string>();

      if (activeNodeId) {
        for (const edge of edges) {
          if (edge.source === activeNodeId) connectedNodeIds.add(edge.target);
          if (edge.target === activeNodeId) connectedNodeIds.add(edge.source);
        }
      }

      if (nodes.length > 1) {
        for (let i = 0; i < nodes.length; i += 1) {
          for (let j = i + 1; j < nodes.length; j += 1) {
            const a = nodes[i];
            const b = nodes[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist = Math.max(Math.hypot(dx, dy), 1);
            const force = 2200 / (dist * dist);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            a.vx -= fx;
            a.vy -= fy;
            b.vx += fx;
            b.vy += fy;
          }
        }
      }

      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      for (const edge of edges) {
        const src = nodeMap.get(edge.source);
        const tgt = nodeMap.get(edge.target);
        if (!src || !tgt) continue;
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const desired = Math.max(90, Math.min(width, height) * 0.18);
        const force = (dist - desired) * 0.01;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        src.vx += fx;
        src.vy += fy;
        tgt.vx -= fx;
        tgt.vy -= fy;
      }

      for (const node of nodes) {
        node.vx += (width / 2 - node.x) * 0.0012;
        node.vy += (height / 2 - node.y) * 0.0012;
      }

      for (const node of nodes) {
        if (dragRef.current?.nodeId === node.id) continue;
        node.vx *= 0.84;
        node.vy *= 0.84;
        node.x = clamp(node.x + node.vx, 28, width - 28);
        node.y = clamp(node.y + node.vy, 28, height - 28);
      }

      ctx.clearRect(0, 0, width, height);
      drawBackground(ctx, width, height);

      for (const group of groupEdges(edges)) {
        const spread = Math.max(group.length - 1, 1);
        group.forEach((edge, index) => {
          const src = nodeMap.get(edge.source);
          const tgt = nodeMap.get(edge.target);
          if (!src || !tgt) return;

          const isActive =
            activeNodeId !== null &&
            (edge.source === activeNodeId || edge.target === activeNodeId);
          const curveOffset = (index - spread / 2) * 16;
          drawEdge(ctx, src, tgt, edge, curveOffset, isActive);
        });
      }

      const orderedNodes = [...nodes].sort((a, b) => {
        const aPriority =
          (selectedNodeId === a.id ? 3 : 0) +
          (hoveredNodeId === a.id ? 2 : 0) +
          (connectedNodeIds.has(a.id) ? 1 : 0);
        const bPriority =
          (selectedNodeId === b.id ? 3 : 0) +
          (hoveredNodeId === b.id ? 2 : 0) +
          (connectedNodeIds.has(b.id) ? 1 : 0);
        return aPriority - bPriority;
      });

      for (const node of orderedNodes) {
        drawNode(ctx, node, hoveredNodeId === node.id, selectedNodeId === node.id, connectedNodeIds.has(node.id));
      }

      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [canvasSize, graph, hoveredNodeId, selectedNodeId]);

  const activeNode = useMemo(
    () => graph?.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [graph, selectedNodeId]
  );

  const relationshipItems = useMemo<RelationshipItem[]>(() => {
    if (!graph) return [];

    const items = graph.edges
      .map((edge, index) => ({
        ...edge,
        id: `${edge.source}-${edge.target}-${edge.timestamp}-${index}`,
        otherNodeId:
          selectedNodeId === edge.source ? edge.target : edge.source,
      }))
      .filter((edge) => {
        if (!selectedNodeId) return true;
        return edge.source === selectedNodeId || edge.target === selectedNodeId;
      })
      .sort((a, b) => a.timestamp - b.timestamp);

    return items;
  }, [graph, selectedNodeId]);

  const nodeLookup = useMemo(
    () => new Map((graph?.nodes ?? []).map((node) => [node.id, node])),
    [graph]
  );

  const getCanvasPoint = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const scaleX = canvasSize.width / rect.width;
      const scaleY = canvasSize.height / rect.height;
      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
      };
    },
    [canvasSize]
  );

  const findNodeAtPoint = useCallback((x: number, y: number) => {
    for (const node of [...nodesRef.current].reverse()) {
      const radius = nodeRadius(node.count);
      const dx = x - node.x;
      const dy = y - node.y;
      if (dx * dx + dy * dy <= radius * radius) {
        return node;
      }
    }
    return null;
  }, []);

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const point = getCanvasPoint(event.clientX, event.clientY);
      if (!point) return;

      if (dragRef.current) {
        const dragged = nodesRef.current.find((node) => node.id === dragRef.current?.nodeId);
        if (dragged) {
          dragged.x = clamp(point.x - dragRef.current.offsetX, 28, canvasSize.width - 28);
          dragged.y = clamp(point.y - dragRef.current.offsetY, 28, canvasSize.height - 28);
          dragged.vx = 0;
          dragged.vy = 0;
          setHoveredNodeId(dragged.id);
        }
        return;
      }

      const hit = findNodeAtPoint(point.x, point.y);
      setHoveredNodeId((prev) => (prev === hit?.id ? prev : hit?.id ?? null));
    },
    [canvasSize, findNodeAtPoint, getCanvasPoint]
  );

  const handleMouseDown = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const point = getCanvasPoint(event.clientX, event.clientY);
      if (!point) return;

      const hit = findNodeAtPoint(point.x, point.y);
      if (!hit) {
        setSelectedNodeId(null);
        setHoveredNodeId(null);
        return;
      }

      setSelectedNodeId(hit.id);
      setHoveredNodeId(hit.id);
      dragRef.current = {
        nodeId: hit.id,
        offsetX: point.x - hit.x,
        offsetY: point.y - hit.y,
      };
      setIsDragging(true);
    },
    [findNodeAtPoint, getCanvasPoint]
  );

  const releaseDrag = useCallback(() => {
    dragRef.current = null;
    setIsDragging(false);
  }, []);

  if (loading) return <div style={styles.loading}>Loading graph...</div>;
  if (!graph || graph.nodes.length === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <h3 style={styles.title}>Knowledge Graph</h3>
          <button onClick={onClose} style={styles.closeBtn}>×</button>
        </div>
        <p style={styles.empty}>No entities detected in this video.</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div>
          <h3 style={styles.title}>Knowledge Graph</h3>
          <p style={styles.subtitle}>
            {graph.nodes.length} entities and {graph.edges.length} relationships
          </p>
        </div>

        <div style={styles.legend}>
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <span key={type} style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: color }} />
              {type}
            </span>
          ))}
        </div>

        <button onClick={onClose} style={styles.closeBtn}>×</button>
      </div>

      <div style={styles.content}>
        <div ref={canvasHostRef} style={styles.canvasPanel}>
          <canvas
            ref={canvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            style={{
              ...styles.canvas,
              cursor: isDragging ? "grabbing" : hoveredNodeId ? "pointer" : "grab",
            }}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={releaseDrag}
            onMouseLeave={() => {
              setHoveredNodeId(null);
              releaseDrag();
            }}
          />

          <div style={styles.canvasHint}>
            {activeNode
              ? `Selected ${humanizeLabel(activeNode.label)}. Drag to rearrange, or click another entity to follow its relationships.`
              : "Click a node to inspect relationships. Drag nodes to untangle the graph."}
          </div>
        </div>

        <aside style={styles.sidebar}>
          <div style={styles.sidebarCard}>
            <div style={styles.sidebarHeadingRow}>
              <span style={styles.sidebarLabel}>Focus</span>
              {activeNode && (
                <button onClick={() => setSelectedNodeId(null)} style={styles.secondaryBtn}>
                  Clear
                </button>
              )}
            </div>

            {activeNode ? (
              <>
                <h4 style={styles.nodeTitle}>{humanizeLabel(activeNode.label)}</h4>
                <p style={styles.nodeMeta}>
                  {activeNode.type} • seen {activeNode.count} times • {formatTime(activeNode.first_seen)} to {formatTime(activeNode.last_seen)}
                </p>
              </>
            ) : (
              <p style={styles.nodeMeta}>
                Select an entity to highlight its incoming and outgoing relationships.
              </p>
            )}
          </div>

          <div style={styles.sidebarCard}>
            <div style={styles.sidebarHeadingRow}>
              <span style={styles.sidebarLabel}>
                {activeNode ? "Connected relationships" : "Relationship preview"}
              </span>
              <span style={styles.countBadge}>{relationshipItems.length}</span>
            </div>

            {relationshipItems.length === 0 ? (
              <p style={styles.emptySidebar}>
                No explicit relationships were detected for this graph yet.
              </p>
            ) : (
              <div style={styles.relationshipList}>
                {relationshipItems.slice(0, 8).map((item) => {
                  const relatedNode = nodeLookup.get(item.otherNodeId);
                  const sourceNode = nodeLookup.get(item.source);
                  const targetNode = nodeLookup.get(item.target);
                  const relationshipLabel = activeNode
                    ? humanizeLabel(relatedNode?.label ?? item.otherNodeId)
                    : `${humanizeLabel(sourceNode?.label ?? item.source)} -> ${humanizeLabel(
                        targetNode?.label ?? item.target
                      )}`;
                  return (
                    <button
                      key={item.id}
                      style={styles.relationshipItem}
                      onClick={() => setSelectedNodeId(item.otherNodeId)}
                    >
                      <span style={styles.relationshipTime}>{formatTime(item.timestamp)}</span>
                      <span style={styles.relationshipBody}>
                        <span style={styles.relationshipAction}>{humanizeLabel(item.action || "related_to")}</span>
                        <span style={styles.relationshipTarget}>
                          {relationshipLabel}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function initializeNodes(nodes: GraphNode[], canvasSize: CanvasSize): SimNode[] {
  if (nodes.length === 0) return [];

  const centerX = canvasSize.width / 2;
  const centerY = canvasSize.height / 2;
  const orbit = Math.max(90, Math.min(canvasSize.width, canvasSize.height) * 0.28);

  return nodes.map((node, index) => {
    const angle = (2 * Math.PI * index) / nodes.length;
    return {
      ...node,
      x: centerX + orbit * Math.cos(angle),
      y: centerY + orbit * Math.sin(angle),
      vx: 0,
      vy: 0,
    };
  });
}

function drawBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#f8fafc");
  gradient.addColorStop(1, "#eef2ff");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.lineWidth = 1;
  for (let x = 32; x < width; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function drawEdge(
  ctx: CanvasRenderingContext2D,
  src: SimNode,
  tgt: SimNode,
  edge: GraphEdge,
  offset: number,
  isActive: boolean
) {
  const dx = tgt.x - src.x;
  const dy = tgt.y - src.y;
  const dist = Math.max(Math.hypot(dx, dy), 1);
  const nx = -dy / dist;
  const ny = dx / dist;
  const cx = (src.x + tgt.x) / 2 + nx * offset;
  const cy = (src.y + tgt.y) / 2 + ny * offset;

  ctx.beginPath();
  ctx.moveTo(src.x, src.y);
  ctx.quadraticCurveTo(cx, cy, tgt.x, tgt.y);
  ctx.strokeStyle = isActive ? "#334155" : "rgba(148, 163, 184, 0.55)";
  ctx.lineWidth = isActive ? 2.4 : 1.2;
  ctx.stroke();

  const arrowAngle = Math.atan2(tgt.y - cy, tgt.x - cx);
  const arrowSize = isActive ? 8 : 6;
  ctx.beginPath();
  ctx.moveTo(tgt.x, tgt.y);
  ctx.lineTo(
    tgt.x - arrowSize * Math.cos(arrowAngle - Math.PI / 6),
    tgt.y - arrowSize * Math.sin(arrowAngle - Math.PI / 6)
  );
  ctx.lineTo(
    tgt.x - arrowSize * Math.cos(arrowAngle + Math.PI / 6),
    tgt.y - arrowSize * Math.sin(arrowAngle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fillStyle = isActive ? "#334155" : "rgba(148, 163, 184, 0.8)";
  ctx.fill();

  if (isActive && edge.action) {
    const labelX = (src.x + 2 * cx + tgt.x) / 4;
    const labelY = (src.y + 2 * cy + tgt.y) / 4;
    ctx.font = "600 11px sans-serif";
    ctx.textAlign = "center";
    ctx.strokeStyle = "rgba(248, 250, 252, 0.95)";
    ctx.lineWidth = 4;
    ctx.strokeText(humanizeLabel(edge.action), labelX, labelY - 6);
    ctx.fillStyle = "#475569";
    ctx.fillText(humanizeLabel(edge.action), labelX, labelY - 6);
  }
}

function drawNode(
  ctx: CanvasRenderingContext2D,
  node: SimNode,
  isHovered: boolean,
  isSelected: boolean,
  isConnected: boolean
) {
  const radius = nodeRadius(node.count);
  const color = TYPE_COLORS[node.type] || "#94a3b8";

  ctx.beginPath();
  ctx.arc(node.x, node.y, radius + (isSelected ? 6 : isHovered ? 4 : 0), 0, Math.PI * 2);
  ctx.fillStyle = isSelected
    ? "rgba(59, 130, 246, 0.14)"
    : isConnected
    ? "rgba(15, 23, 42, 0.08)"
    : "rgba(255, 255, 255, 0.7)";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = isHovered || isSelected ? color : `${color}cc`;
  ctx.fill();

  ctx.strokeStyle = isSelected ? "#0f172a" : isHovered ? color : "#ffffff";
  ctx.lineWidth = isSelected ? 3 : 2;
  ctx.stroke();

  ctx.font = `${isSelected ? "700" : "600"} 12px sans-serif`;
  ctx.textAlign = "center";
  ctx.strokeStyle = "rgba(248, 250, 252, 0.95)";
  ctx.lineWidth = 5;
  ctx.strokeText(humanizeLabel(node.label), node.x, node.y + radius + 16);
  ctx.fillStyle = "#0f172a";
  ctx.fillText(humanizeLabel(node.label), node.x, node.y + radius + 16);
}

function groupEdges(edges: GraphEdge[]): GraphEdge[][] {
  const groups = new Map<string, GraphEdge[]>();
  for (const edge of edges) {
    const key = [edge.source, edge.target].sort().join("::");
    groups.set(key, [...(groups.get(key) ?? []), edge]);
  }
  return [...groups.values()];
}

function nodeRadius(count: number): number {
  return Math.min(10 + Math.sqrt(Math.max(count, 1)) * 3, 22);
}

function humanizeLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function formatTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }

  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: "#ffffff",
    borderRadius: "16px",
    border: "1px solid #e2e8f0",
    padding: "18px",
    marginBottom: "16px",
    boxShadow: "0 10px 28px rgba(15, 23, 42, 0.08)",
  },
  header: {
    display: "flex",
    alignItems: "flex-start",
    gap: "16px",
    marginBottom: "16px",
    flexWrap: "wrap",
  },
  title: {
    margin: 0,
    fontSize: "18px",
    fontWeight: 700,
    color: "#0f172a",
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: "13px",
    color: "#64748b",
  },
  legend: {
    display: "flex",
    gap: "12px",
    flexWrap: "wrap",
    flex: 1,
    alignItems: "center",
    paddingTop: "2px",
  },
  legendItem: {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "12px",
    color: "#475569",
    textTransform: "capitalize",
  },
  legendDot: {
    width: "10px",
    height: "10px",
    borderRadius: "999px",
    display: "inline-block",
  },
  closeBtn: {
    background: "none",
    border: "none",
    fontSize: "22px",
    color: "#94a3b8",
    cursor: "pointer",
    padding: 0,
    lineHeight: 1,
  },
  content: {
    display: "flex",
    gap: "16px",
    flexWrap: "wrap",
  },
  canvasPanel: {
    flex: "2 1 520px",
    minWidth: "320px",
  },
  canvas: {
    width: "100%",
    height: "auto",
    display: "block",
    borderRadius: "14px",
    border: "1px solid #dbe3f0",
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.6)",
    background: "#f8fafc",
  },
  canvasHint: {
    marginTop: "10px",
    fontSize: "12px",
    color: "#64748b",
  },
  sidebar: {
    flex: "1 1 260px",
    minWidth: "240px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  sidebarCard: {
    borderRadius: "14px",
    border: "1px solid #e2e8f0",
    background: "#f8fafc",
    padding: "14px",
  },
  sidebarHeadingRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "10px",
    marginBottom: "8px",
  },
  sidebarLabel: {
    fontSize: "12px",
    fontWeight: 700,
    color: "#475569",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  },
  secondaryBtn: {
    border: "1px solid #cbd5e1",
    borderRadius: "999px",
    background: "#ffffff",
    color: "#334155",
    fontSize: "12px",
    padding: "4px 10px",
    cursor: "pointer",
  },
  nodeTitle: {
    margin: "0 0 6px",
    fontSize: "18px",
    color: "#0f172a",
  },
  nodeMeta: {
    margin: 0,
    fontSize: "13px",
    color: "#64748b",
    lineHeight: 1.5,
  },
  countBadge: {
    fontSize: "12px",
    fontWeight: 700,
    color: "#1d4ed8",
    background: "#dbeafe",
    borderRadius: "999px",
    padding: "4px 8px",
  },
  relationshipList: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  relationshipItem: {
    display: "flex",
    gap: "10px",
    alignItems: "flex-start",
    border: "1px solid #dbe3f0",
    borderRadius: "12px",
    background: "#ffffff",
    cursor: "pointer",
    padding: "10px 12px",
    textAlign: "left",
  },
  relationshipTime: {
    minWidth: "52px",
    fontSize: "12px",
    fontFamily: "monospace",
    color: "#1d4ed8",
    paddingTop: "2px",
  },
  relationshipBody: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
    minWidth: 0,
  },
  relationshipAction: {
    fontSize: "12px",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  relationshipTarget: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#0f172a",
  },
  emptySidebar: {
    margin: 0,
    fontSize: "13px",
    color: "#64748b",
    lineHeight: 1.5,
  },
  loading: {
    padding: "40px",
    textAlign: "center",
    color: "#64748b",
  },
  empty: {
    textAlign: "center",
    color: "#94a3b8",
    padding: "40px",
  },
};
