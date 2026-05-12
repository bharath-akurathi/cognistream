import type { SearchResult } from "../types";

interface ResultsPanelProps {
  results: SearchResult[];
  query: string;
  isLoading: boolean;
  error: string | null;
  activeIndex: number | null;
  onResultClick: (index: number) => void;
  onFindSimilar?: (segmentId: string) => void;
  onExportClip?: (result: SearchResult) => void;
}

export default function ResultsPanel({
  results,
  query,
  isLoading,
  error,
  activeIndex,
  onResultClick,
}: ResultsPanelProps) {
  if (isLoading) {
    return (
      <div style={styles.status}>
        <div style={styles.spinner} />
        <span>Searching for "{query}"...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ ...styles.status, color: "#dc2626" }}>
        <span>{error}</span>
      </div>
    );
  }

  if (query && results.length === 0) {
    return (
      <div style={styles.status}>
        <span>No results found for "{query}"</span>
      </div>
    );
  }

  if (results.length === 0) return null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>Results</span>
        <span style={styles.headerCount}>{results.length} segments</span>
      </div>

      <div style={styles.list}>
        {results.map((result, idx) => (
          <ResultCard
            key={result.segment_id}
            result={result}
            rank={idx + 1}
            isActive={idx === activeIndex}
            onClick={() => onResultClick(idx)}
          />
        ))}
      </div>
    </div>
  );
}

/* Single result card */

function ResultCard({
  result,
  rank,
  isActive,
  onClick,
}: {
  result: SearchResult;
  rank: number;
  isActive: boolean;
  onClick: () => void;
}) {
  // Split the text into visual caption and speech portions
  const { caption, speech } = splitText(result.text);

  return (
    <button
      onClick={onClick}
      style={{
        ...styles.card,
        borderColor: isActive ? "#3b82f6" : "#e2e8f0",
        background: isActive ? "#eff6ff" : "#fff",
      }}
    >
      <div style={styles.cardLeft}>
        <span style={styles.rank}>#{rank}</span>
        <span style={styles.timestamp}>{formatTime(result.start_time)}</span>
        <span style={styles.duration}>
          {formatDuration(result.end_time - result.start_time)}
        </span>
      </div>

      <div style={styles.cardCenter}>
        {caption && <p style={styles.captionText}>{caption}</p>}
        {speech && (
          <p style={styles.speechText}>
            <span style={styles.speechLabel}>Speech: </span>
            {speech}
          </p>
        )}
        {result.event_type && <span style={styles.eventBadge}>{result.event_type}</span>}
      </div>

      <div style={styles.cardRight}>
        <span
          style={{
            ...styles.sourceBadge,
            background: sourceBadgeColor(result.source_type),
          }}
        >
          {result.source_type}
        </span>
        <span style={styles.score}>{Math.round(result.score * 100)}%</span>
      </div>
    </button>
  );
}

function splitText(text: string): { caption: string; speech: string } {
  const speechMatch = text.match(/\[Speech:\s*(.*?)\]$/s);
  if (speechMatch) {
    return {
      caption: text.slice(0, speechMatch.index).trim(),
      speech: speechMatch[1].trim(),
    };
  }
  return { caption: text, speech: "" };
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 1) return "<1s";
  return `${seconds.toFixed(1)}s`;
}

function sourceBadgeColor(source: string): string {
  switch (source) {
    case "visual":
      return "#ede9fe";
    case "audio":
      return "#d1fae5";
    case "event":
      return "#fee2e2";
    default:
      return "#dbeafe";
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "2px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "8px 0",
  },
  headerTitle: {
    fontSize: "14px",
    fontWeight: 600,
    color: "#1e293b",
  },
  headerCount: {
    fontSize: "12px",
    color: "#64748b",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    maxHeight: "500px",
    overflowY: "auto",
    paddingRight: "4px",
  },
  card: {
    display: "flex",
    gap: "16px",
    padding: "14px 16px",
    border: "2px solid #e2e8f0",
    borderRadius: "10px",
    cursor: "pointer",
    textAlign: "left",
    width: "100%",
    transition: "border-color 0.15s, background 0.15s",
    fontFamily: "inherit",
    fontSize: "inherit",
  },
  cardLeft: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "4px",
    minWidth: "48px",
  },
  rank: {
    fontSize: "11px",
    fontWeight: 700,
    color: "#94a3b8",
  },
  timestamp: {
    fontSize: "14px",
    fontWeight: 600,
    fontFamily: "monospace",
    color: "#1e293b",
  },
  duration: {
    fontSize: "11px",
    color: "#94a3b8",
  },
  cardCenter: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    minWidth: 0,
  },
  captionText: {
    fontSize: "14px",
    color: "#1e293b",
    lineHeight: 1.5,
    margin: 0,
    display: "-webkit-box",
    WebkitLineClamp: 3,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  },
  speechText: {
    fontSize: "13px",
    color: "#475569",
    fontStyle: "italic",
    lineHeight: 1.4,
    margin: 0,
    display: "-webkit-box",
    WebkitLineClamp: 2,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  },
  speechLabel: {
    fontStyle: "normal",
    fontWeight: 600,
    color: "#64748b",
  },
  eventBadge: {
    alignSelf: "flex-start",
    fontSize: "11px",
    fontWeight: 600,
    padding: "2px 8px",
    borderRadius: "4px",
    background: "#fef3c7",
    color: "#92400e",
  },
  cardRight: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-end",
    gap: "6px",
    minWidth: "52px",
  },
  sourceBadge: {
    fontSize: "11px",
    fontWeight: 600,
    padding: "2px 8px",
    borderRadius: "4px",
    color: "#334155",
  },
  score: {
    fontSize: "13px",
    fontWeight: 700,
    fontFamily: "monospace",
    color: "#3b82f6",
  },
  status: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    padding: "32px",
    color: "#64748b",
    fontSize: "14px",
  },
  spinner: {
    width: "18px",
    height: "18px",
    border: "2px solid #e2e8f0",
    borderTopColor: "#3b82f6",
    borderRadius: "50%",
    animation: "spin 0.6s linear infinite",
  },
};
