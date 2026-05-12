import { useCallback, useEffect, useRef, useState } from "react";
import type { SearchResult } from "../types";

interface VideoPlayerProps {
  videoUrl: string | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  onTimeUpdate: () => void;
  onLoadedMetadata: () => void;
  togglePlay: () => void;
  onSeek: (time: number) => void;
  activeResult: SearchResult | null;
}

export default function VideoPlayer({
  videoUrl,
  videoRef,
  currentTime,
  duration,
  isPlaying,
  onTimeUpdate,
  onLoadedMetadata,
  togglePlay,
  onSeek,
  activeResult,
}: VideoPlayerProps) {
  const seekTrackRef = useRef<HTMLDivElement>(null);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [isScrubbing, setIsScrubbing] = useState(false);

  const getTimeFromClientX = useCallback(
    (clientX: number) => {
      if (duration <= 0) return 0;
      const track = seekTrackRef.current;
      if (!track) return 0;
      const rect = track.getBoundingClientRect();
      if (rect.width <= 0) return 0;
      const percent = clamp((clientX - rect.left) / rect.width, 0, 1);
      return percent * duration;
    },
    [duration]
  );

  const previewSeekAt = useCallback(
    (clientX: number, commit: boolean) => {
      const nextTime = getTimeFromClientX(clientX);
      setHoverTime(nextTime);
      if (commit) {
        onSeek(nextTime);
      }
    },
    [getTimeFromClientX, onSeek]
  );

  useEffect(() => {
    if (!isScrubbing) return;

    const handlePointerMove = (event: PointerEvent) => {
      previewSeekAt(event.clientX, true);
    };

    const handlePointerUp = (event: PointerEvent) => {
      previewSeekAt(event.clientX, true);
      setIsScrubbing(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [isScrubbing, previewSeekAt]);

  if (!videoUrl) {
    return (
      <div style={styles.placeholder}>
        <svg style={styles.placeholderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <polygon points="5,3 19,12 5,21" />
        </svg>
        <p style={styles.placeholderText}>Select a video to begin</p>
      </div>
    );
  }

  const displayedTime = isScrubbing && hoverTime !== null ? hoverTime : currentTime;
  const progressPercent = duration > 0 ? (displayedTime / duration) * 100 : 0;
  const hoverPercent = duration > 0 && hoverTime !== null ? (hoverTime / duration) * 100 : null;
  const activeStartPercent =
    activeResult && duration > 0 ? (activeResult.start_time / duration) * 100 : null;
  const activeWidthPercent =
    activeResult && duration > 0
      ? (Math.max(activeResult.end_time - activeResult.start_time, 0.4) / duration) * 100
      : null;

  return (
    <div style={styles.container}>
      <video
        ref={videoRef}
        src={videoUrl}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
        style={styles.video}
        onClick={togglePlay}
      />

      <div style={styles.transport}>
        <div style={styles.transportHeader}>
          <button onClick={togglePlay} style={styles.playBtn} aria-label={isPlaying ? "Pause video" : "Play video"}>
            {isPlaying ? "\u23F8" : "\u25B6"}
          </button>

          <div style={styles.transportMeta}>
            <span style={styles.timecode}>
              {formatTime(displayedTime)} / {formatTime(duration)}
            </span>
            {activeResult && (
              <span style={styles.resultBadge}>
                Focus segment {formatTime(activeResult.start_time)} - {formatTime(activeResult.end_time)}
              </span>
            )}
          </div>
        </div>

        <div
          ref={seekTrackRef}
          style={styles.seekArea}
          role="slider"
          tabIndex={duration > 0 ? 0 : -1}
          aria-label="Video seek bar"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(displayedTime)}
          aria-valuetext={formatTime(displayedTime)}
          onPointerDown={(event) => {
            if (duration <= 0) return;
            setIsScrubbing(true);
            previewSeekAt(event.clientX, true);
          }}
          onPointerMove={(event) => {
            if (duration <= 0) return;
            previewSeekAt(event.clientX, isScrubbing);
          }}
          onPointerLeave={() => {
            if (!isScrubbing) {
              setHoverTime(null);
            }
          }}
          onKeyDown={(event) => {
            if (duration <= 0) return;

            if (event.key === "ArrowLeft") {
              event.preventDefault();
              onSeek(Math.max(currentTime - 5, 0));
              return;
            }

            if (event.key === "ArrowRight") {
              event.preventDefault();
              onSeek(Math.min(currentTime + 5, duration));
              return;
            }

            if (event.key === "Home") {
              event.preventDefault();
              onSeek(0);
              return;
            }

            if (event.key === "End") {
              event.preventDefault();
              onSeek(duration);
            }
          }}
        >
          <div style={styles.seekTrack}>
            {activeResult && activeStartPercent !== null && activeWidthPercent !== null && (
              <div
                style={{
                  ...styles.activeMarker,
                  left: `${activeStartPercent}%`,
                  width: `${activeWidthPercent}%`,
                }}
              />
            )}

            <div
              style={{
                ...styles.progressFill,
                width: `${progressPercent}%`,
              }}
            />

            {hoverPercent !== null && (
              <div
                style={{
                  ...styles.previewLine,
                  left: `${hoverPercent}%`,
                }}
              />
            )}

            <div
              style={{
                ...styles.scrubber,
                left: `${progressPercent}%`,
              }}
            />
          </div>

          {hoverPercent !== null && (
            <div
              style={{
                ...styles.previewBubble,
                left: `${hoverPercent}%`,
              }}
            >
              {formatTime(hoverTime ?? 0)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
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
    borderRadius: "12px",
    overflow: "hidden",
    background: "#020617",
    border: "1px solid #cbd5e1",
    boxShadow: "0 16px 36px rgba(15, 23, 42, 0.16)",
  },
  video: {
    width: "100%",
    display: "block",
    cursor: "pointer",
    maxHeight: "450px",
    objectFit: "contain",
    background: "#000",
  },
  transport: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    padding: "14px 16px 16px",
    background: "linear-gradient(180deg, #0f172a 0%, #020617 100%)",
  },
  transportHeader: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  playBtn: {
    width: "40px",
    height: "40px",
    borderRadius: "999px",
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(59, 130, 246, 0.18)",
    color: "#f8fafc",
    fontSize: "18px",
    cursor: "pointer",
    lineHeight: 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
  },
  transportMeta: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    flexWrap: "wrap",
    minWidth: 0,
  },
  timecode: {
    color: "#e2e8f0",
    fontSize: "13px",
    fontFamily: "monospace",
    whiteSpace: "nowrap",
  },
  resultBadge: {
    color: "#f8fafc",
    background: "rgba(245, 158, 11, 0.18)",
    border: "1px solid rgba(251, 191, 36, 0.4)",
    fontSize: "12px",
    borderRadius: "999px",
    padding: "5px 10px",
    whiteSpace: "nowrap",
  },
  seekArea: {
    position: "relative",
    paddingTop: "14px",
    paddingBottom: "8px",
    touchAction: "none",
    outline: "none",
  },
  seekTrack: {
    position: "relative",
    height: "10px",
    background: "rgba(51, 65, 85, 0.95)",
    borderRadius: "999px",
    overflow: "hidden",
    border: "1px solid rgba(148, 163, 184, 0.18)",
  },
  progressFill: {
    position: "absolute",
    top: 0,
    left: 0,
    height: "100%",
    background: "linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%)",
    borderRadius: "999px",
  },
  activeMarker: {
    position: "absolute",
    top: 0,
    height: "100%",
    background: "rgba(251, 191, 36, 0.38)",
    borderRadius: "999px",
    minWidth: "6px",
  },
  previewLine: {
    position: "absolute",
    top: "-4px",
    bottom: "-4px",
    width: "2px",
    background: "rgba(248, 250, 252, 0.82)",
    transform: "translateX(-50%)",
  },
  scrubber: {
    position: "absolute",
    top: "50%",
    width: "18px",
    height: "18px",
    borderRadius: "999px",
    background: "#f8fafc",
    border: "3px solid #2563eb",
    boxShadow: "0 4px 14px rgba(2, 6, 23, 0.3)",
    transform: "translate(-50%, -50%)",
  },
  previewBubble: {
    position: "absolute",
    top: 0,
    transform: "translate(-50%, -100%)",
    background: "#f8fafc",
    color: "#0f172a",
    fontSize: "12px",
    fontWeight: 700,
    fontFamily: "monospace",
    padding: "4px 8px",
    borderRadius: "8px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.18)",
    pointerEvents: "none",
  },
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "350px",
    background: "#f1f5f9",
    borderRadius: "12px",
    border: "2px dashed #cbd5e1",
    gap: "16px",
  },
  placeholderIcon: {
    width: "48px",
    height: "48px",
    color: "#94a3b8",
  },
  placeholderText: {
    color: "#64748b",
    fontSize: "15px",
  },
};
