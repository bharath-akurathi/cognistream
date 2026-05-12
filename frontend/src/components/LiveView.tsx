import { useCallback, useEffect, useRef, useState } from "react";
import {
  startLiveFeed,
  stopLiveFeed,
  getLiveFeedStatus,
  connectLiveWebSocket,
  uploadBrowserChunk,
  stopBrowserFeed,
  type LiveFeedInfo,
  type LiveWsEvent,
} from "../api/client";

interface LiveViewProps {
  onBack: () => void;
}

interface LiveLogEntry {
  id: number;
  time: string;
  type: string;
  message: string;
}

type SourceMode = "url" | "browser-camera" | "screen";

let logCounter = 0;

const SOURCE_PRESETS = [
  { label: "Webcam", value: "0", icon: "\uD83D\uDCF7", desc: "Laptop/USB camera via OpenCV" },
  { label: "Phone (IP Webcam)", value: "http://192.168.1.X:8080/video", icon: "\uD83D\uDCF1", desc: "Android IP Webcam app" },
  { label: "Phone (DroidCam)", value: "http://192.168.1.X:4747/video", icon: "\uD83D\uDCF1", desc: "DroidCam app" },
  { label: "RTSP Camera", value: "rtsp://admin:password@192.168.1.X:554/stream", icon: "\uD83C\uDFA5", desc: "IP security camera" },
  { label: "HTTP Stream", value: "http://", icon: "\uD83C\uDF10", desc: "MJPEG or HTTP video stream" },
  { label: "Video File", value: "", icon: "\uD83D\uDCC1", desc: "Local file path as fake live stream" },
];

export default function LiveView({ onBack }: LiveViewProps) {
  const [url, setUrl] = useState("");
  const [feedId, setFeedId] = useState("live-" + Date.now().toString(36).slice(-4));
  const [chunkSec, setChunkSec] = useState(15);
  const [sourceMode, setSourceMode] = useState<SourceMode>("url");
  const [feeds, setFeeds] = useState<LiveFeedInfo[]>([]);
  const [activeFeed, setActiveFeed] = useState<string | null>(null);
  const [log, setLog] = useState<LiveLogEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<
    { text: string; start_time: number; score: number }[]
  >([]);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const [browserRecording, setBrowserRecording] = useState(false);
  const wsRef = useRef<{ send: (msg: Record<string, unknown>) => void; close: () => void } | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement>(null);
  const chunkCountRef = useRef(0);
  const chunkStartRef = useRef(0);

  // Poll feed status every 5s
  useEffect(() => {
    const poll = () => {
      getLiveFeedStatus().then(setFeeds).catch(() => {});
    };
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  const addLog = useCallback((type: string, message: string) => {
    setLog((prev) => {
      const entry: LiveLogEntry = {
        id: ++logCounter,
        time: new Date().toLocaleTimeString(),
        type,
        message,
      };
      const next = [...prev, entry];
      return next.length > 200 ? next.slice(-100) : next;
    });
  }, []);

  // ── Server-side feed (RTSP/webcam/file) ─────────────────
  const handleStartServerFeed = async () => {
    if (!url.trim() || !feedId.trim()) {
      setError("URL and Feed ID are required.");
      return;
    }
    setError("");
    setStarting(true);
    try {
      await startLiveFeed(url.trim(), feedId.trim(), chunkSec);
      addLog("info", `Started feed "${feedId}" from ${url}`);
      connectToFeed(feedId.trim());
      const updated = await getLiveFeedStatus();
      setFeeds(updated);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to start feed";
      setError(msg);
    } finally {
      setStarting(false);
    }
  };

  // ── Browser camera / screen capture ─────────────────────
  const handleStartBrowserCapture = async () => {
    if (!feedId.trim()) {
      setError("Feed ID is required.");
      return;
    }
    setError("");
    setStarting(true);

    try {
      let stream: MediaStream;
      if (sourceMode === "screen") {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: true,
        });
      } else {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
          audio: true,
        });
      }

      streamRef.current = stream;

      // Show preview
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.play().catch(() => {});
      }

      // MediaRecorder produces chunks that are NOT independently playable
      // when using timeslice — only the first chunk has the webm header.
      // Fix: stop + restart the recorder for each chunk so every blob is
      // a complete, self-contained webm file that OpenCV/FFmpeg can read.
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
        ? "video/webm;codecs=vp9,opus"
        : MediaRecorder.isTypeSupported("video/webm;codecs=vp8,opus")
        ? "video/webm;codecs=vp8,opus"
        : "video/webm";

      chunkCountRef.current = 0;
      chunkStartRef.current = 0;

      const startNewRecorder = () => {
        if (!streamRef.current || streamRef.current.getTracks().every(t => t.readyState === "ended")) {
          return; // stream was stopped
        }

        const rec = new MediaRecorder(streamRef.current, {
          mimeType,
          videoBitsPerSecond: 1_000_000,
        });
        mediaRecorderRef.current = rec;

        const chunks: Blob[] = [];
        rec.ondataavailable = (event) => {
          if (event.data.size > 0) chunks.push(event.data);
        };

        rec.onstop = async () => {
          if (chunks.length === 0) return;

          // Combine all data chunks into one complete webm blob
          const blob = new Blob(chunks, { type: mimeType });
          if (blob.size < 500) return;

          const idx = chunkCountRef.current++;
          const startSec = chunkStartRef.current;
          chunkStartRef.current += chunkSec;

          addLog("chunk", `Uploading chunk ${idx} (${(blob.size / 1024).toFixed(0)} KB)...`);

          // Retry upload up to 2 times on failure
          let uploaded = false;
          for (let attempt = 0; attempt < 3 && !uploaded; attempt++) {
            try {
              const result = await uploadBrowserChunk(feedId.trim(), idx, startSec, blob);
              addLog("chunk", `Chunk ${idx}: ${result.segments_stored} segments indexed`);
              uploaded = true;
            } catch (err) {
              if (attempt < 2) {
                addLog("info", `Chunk ${idx} retry ${attempt + 1}/2...`);
                await new Promise((r) => setTimeout(r, 1000));
              } else {
                addLog("error", `Chunk ${idx} failed after 3 attempts: ${err instanceof Error ? err.message : "unknown"}`);
              }
            }
          }

          // Start next chunk (if not stopped)
          if (streamRef.current && !streamRef.current.getTracks().every(t => t.readyState === "ended")) {
            startNewRecorder();
          }
        };

        rec.start(); // record everything until we call stop()

        // Stop after chunkSec seconds → triggers onstop → uploads → starts next
        setTimeout(() => {
          if (rec.state === "recording") {
            rec.stop();
          }
        }, chunkSec * 1000);
      };

      startNewRecorder();
      setBrowserRecording(true);
      setActiveFeed(feedId.trim());
      addLog("info", `Browser ${sourceMode === "screen" ? "screen" : "camera"} capture started`);

      // Connect WebSocket for real-time events
      connectToFeed(feedId.trim());

      // Handle stream ending (user clicks "Stop sharing" in browser)
      stream.getVideoTracks()[0].onended = () => {
        handleStopBrowserCapture();
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to access camera/screen";
      setError(msg);
    } finally {
      setStarting(false);
    }
  };

  const handleStopBrowserCapture = async () => {
    // Stop recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;

    // Stop all tracks
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (videoPreviewRef.current) {
      videoPreviewRef.current.srcObject = null;
    }

    setBrowserRecording(false);
    addLog("info", "Browser capture stopped, finalizing...");

    try {
      const result = await stopBrowserFeed(feedId.trim());
      addLog("info", `Finalized: ${result.total_chunks} chunks processed`);
    } catch {
      addLog("error", "Failed to finalize browser feed");
    }
  };

  const handleStop = async (vid: string) => {
    try {
      await stopLiveFeed(vid);
      addLog("info", `Stopped feed "${vid}"`);
      if (activeFeed === vid) {
        wsRef.current?.close();
        wsRef.current = null;
        setActiveFeed(null);
      }
      const updated = await getLiveFeedStatus();
      setFeeds(updated);
    } catch {
      addLog("error", `Failed to stop feed "${vid}"`);
    }
  };

  const connectToFeed = (vid: string) => {
    wsRef.current?.close();
    setActiveFeed(vid);
    setLog([]);
    setSearchResults([]);
    addLog("info", `Connecting to live feed "${vid}"...`);

    const conn = connectLiveWebSocket(
      vid,
      (event: LiveWsEvent) => {
        switch (event.event_type) {
          case "status_change":
            addLog("status", `State: ${event.data.state as string}`);
            break;
          case "chunk_ready":
            addLog(
              "chunk",
              `Chunk ${event.data.chunk_index as number}: ${event.data.segments_stored as number} segments`
            );
            break;
          case "event_detected":
            addLog("event", `${event.data.event_type as string}: ${event.data.description as string}`);
            break;
          case "search_results": {
            const results = event.data.results as { text: string; start_time: number; score: number }[];
            setSearchResults(results);
            addLog("search", `${results.length} results for "${event.data.query as string}"`);
            break;
          }
          case "error":
            addLog("error", event.data.message as string);
            break;
          default:
            addLog("info", `${event.event_type}: ${JSON.stringify(event.data)}`);
        }
      },
      () => addLog("info", "WebSocket disconnected")
    );
    wsRef.current = conn;
  };

  const handleLiveSearch = () => {
    if (!searchQuery.trim() || !wsRef.current) return;
    wsRef.current.send({ action: "search", query: searchQuery.trim() });
    addLog("search", `Searching: "${searchQuery}"`);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const stateColor = (state: string) => {
    switch (state) {
      case "running": return "#22c55e";
      case "connecting": case "reconnecting": return "#eab308";
      case "error": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const logColor = (type: string) => {
    switch (type) {
      case "chunk": return "#3b82f6";
      case "event": return "#f59e0b";
      case "error": return "#ef4444";
      case "search": return "#a855f7";
      case "status": return "#22c55e";
      default: return "#9ca3af";
    }
  };

  const isBrowserMode = sourceMode === "browser-camera" || sourceMode === "screen";

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <button onClick={onBack} style={btnStyle("#374151")}>Back</button>
        <h2 style={{ margin: 0, fontSize: 20 }}>Live Video Feeds</h2>
        <span style={{ marginLeft: "auto", padding: "4px 12px", background: "#1e293b", borderRadius: 12, fontSize: 13, color: "#94a3b8" }}>
          {feeds.filter((f) => f.state === "running").length} active
        </span>
      </div>

      {/* Source mode tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {([
          { key: "url" as SourceMode, label: "Stream URL / Webcam" },
          { key: "browser-camera" as SourceMode, label: "Phone / Browser Camera" },
          { key: "screen" as SourceMode, label: "Screen Share" },
        ]).map((tab) => (
          <button
            key={tab.key}
            onClick={() => { setSourceMode(tab.key); setError(""); }}
            style={{
              padding: "8px 16px",
              background: sourceMode === tab.key ? "#3b82f6" : "#1e293b",
              color: sourceMode === tab.key ? "#fff" : "#94a3b8",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontWeight: sourceMode === tab.key ? 600 : 400,
              fontSize: 13,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── URL mode: quick presets + URL input ── */}
      {sourceMode === "url" && (
        <div style={cardStyle}>
          <h3 style={sectionTitle}>Quick Connect</h3>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            {SOURCE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setUrl(p.value)}
                title={p.desc}
                style={{
                  padding: "6px 14px",
                  background: url === p.value ? "#1e3a5f" : "#0f172a",
                  border: url === p.value ? "1px solid #3b82f6" : "1px solid #334155",
                  borderRadius: 6,
                  color: "#e2e8f0",
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                {p.icon} {p.label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="rtsp://camera:554/stream or 0 for webcam or C:/path/to/video.mp4"
              style={inputStyle("2 1 250px")}
            />
            <input
              type="text"
              value={feedId}
              onChange={(e) => setFeedId(e.target.value)}
              placeholder="Feed ID"
              style={inputStyle("1 1 120px")}
            />
            <input
              type="number"
              value={chunkSec}
              onChange={(e) => setChunkSec(Number(e.target.value))}
              min={5} max={120}
              title="Chunk seconds"
              style={{ ...inputStyle("0 0 70px"), textAlign: "center" as const }}
            />
            <button
              onClick={handleStartServerFeed}
              disabled={starting}
              style={btnStyle(starting ? "#334155" : "#3b82f6")}
            >
              {starting ? "Starting..." : "Start Feed"}
            </button>
          </div>
          <p style={{ color: "#64748b", fontSize: 12, marginTop: 10, lineHeight: 1.6 }}>
            <strong>Webcam:</strong> Enter <code>0</code> (or <code>1</code>, <code>2</code> for additional cameras)<br />
            <strong>Phone camera:</strong> Install <em>IP Webcam</em> (Android) or <em>DroidCam</em>, start server, enter the HTTP URL<br />
            <strong>CCTV/IP cam:</strong> Enter the RTSP URL from your camera's settings<br />
            <strong>Test with file:</strong> Enter a local video path (e.g. <code>D:/videos/sample.mp4</code>)
          </p>
          {error && <p style={{ color: "#ef4444", fontSize: 13, marginTop: 8 }}>{error}</p>}
        </div>
      )}

      {/* ── Browser camera / screen mode ── */}
      {isBrowserMode && (
        <div style={cardStyle}>
          <h3 style={sectionTitle}>
            {sourceMode === "screen" ? "Screen Capture" : "Browser Camera"}
          </h3>
          <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 12px" }}>
            {sourceMode === "screen"
              ? "Share your screen or a window. Audio will be captured if available."
              : "Use your phone's browser camera or laptop webcam directly. No app needed — works on any device with a camera."}
          </p>

          {/* Camera preview */}
          <div style={{ position: "relative", marginBottom: 12 }}>
            <video
              ref={videoPreviewRef}
              muted
              playsInline
              style={{
                width: "100%",
                maxHeight: 300,
                background: "#000",
                borderRadius: 8,
                display: browserRecording ? "block" : "none",
              }}
            />
            {browserRecording && (
              <span style={{
                position: "absolute", top: 10, right: 10,
                padding: "4px 10px", background: "#dc2626", color: "#fff",
                borderRadius: 4, fontSize: 12, fontWeight: 700,
              }}>
                REC
              </span>
            )}
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              type="text"
              value={feedId}
              onChange={(e) => setFeedId(e.target.value)}
              placeholder="Feed ID"
              style={inputStyle("1 1 140px")}
            />
            <input
              type="number"
              value={chunkSec}
              onChange={(e) => setChunkSec(Number(e.target.value))}
              min={5} max={60}
              title="Chunk seconds"
              style={{ ...inputStyle("0 0 70px"), textAlign: "center" as const }}
            />
            {!browserRecording ? (
              <button
                onClick={handleStartBrowserCapture}
                disabled={starting}
                style={btnStyle(starting ? "#334155" : "#22c55e")}
              >
                {starting ? "Starting..." : sourceMode === "screen" ? "Share Screen" : "Start Camera"}
              </button>
            ) : (
              <button onClick={handleStopBrowserCapture} style={btnStyle("#dc2626")}>
                Stop Recording
              </button>
            )}
          </div>
          {error && <p style={{ color: "#ef4444", fontSize: 13, marginTop: 8 }}>{error}</p>}
        </div>
      )}

      {/* Active feeds list */}
      {feeds.length > 0 && (
        <div style={{ ...cardStyle, marginTop: 16 }}>
          <h3 style={sectionTitle}>Active Feeds</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {feeds.map((f) => (
              <div
                key={f.video_id}
                onClick={() => connectToFeed(f.video_id)}
                style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "10px 14px",
                  background: activeFeed === f.video_id ? "#1e3a5f" : "#0f172a",
                  borderRadius: 6,
                  border: activeFeed === f.video_id ? "1px solid #3b82f6" : "1px solid transparent",
                  cursor: "pointer",
                }}
              >
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: stateColor(f.state), flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: "#e2e8f0" }}>{f.video_id}</div>
                  <div style={{ fontSize: 12, color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {f.url} | {f.chunks_processed} chunks | {f.total_segments} segments | {f.fps.toFixed(0)} fps
                  </div>
                </div>
                <span style={{ fontSize: 12, color: stateColor(f.state), fontWeight: 600, textTransform: "uppercase" }}>
                  {f.state}
                </span>
                {f.state === "running" && (
                  <button
                    onClick={(e) => { e.stopPropagation(); handleStop(f.video_id); }}
                    style={{ ...btnStyle("#dc2626"), padding: "4px 12px", fontSize: 12 }}
                  >
                    Stop
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live log + search panel */}
      {activeFeed && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 16, marginTop: 16 }}>
          {/* Event log */}
          <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, maxHeight: 500, overflowY: "auto", fontFamily: "monospace", fontSize: 13 }}>
            <h3 style={{ margin: "0 0 10px", fontSize: 14, color: "#94a3b8", position: "sticky", top: 0, background: "#0f172a", padding: "4px 0" }}>
              Live Events — {activeFeed}
            </h3>
            {log.map((entry) => (
              <div key={entry.id} style={{ marginBottom: 4, lineHeight: 1.5, display: "flex", gap: 8 }}>
                <span style={{ color: "#475569", flexShrink: 0 }}>{entry.time}</span>
                <span style={{ color: logColor(entry.type), fontWeight: 600, flexShrink: 0, width: 56, textAlign: "right" }}>[{entry.type}]</span>
                <span style={{ color: "#cbd5e1" }}>{entry.message}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>

          {/* Live search panel */}
          <div style={{ background: "#1e293b", borderRadius: 8, padding: 16, display: "flex", flexDirection: "column", maxHeight: 500 }}>
            <h3 style={{ margin: "0 0 10px", fontSize: 14, color: "#94a3b8" }}>Live Search</h3>
            <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLiveSearch()}
                placeholder="Search indexed segments..."
                style={{ ...inputStyle("1"), fontSize: 13, padding: "8px 10px" }}
              />
              <button onClick={handleLiveSearch} style={btnStyle("#7c3aed")}>Search</button>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {searchResults.length === 0 ? (
                <p style={{ color: "#475569", fontSize: 13, textAlign: "center" }}>
                  Search results from live indexed segments will appear here
                </p>
              ) : (
                searchResults.map((r, i) => (
                  <div key={i} style={{ padding: "8px 10px", background: "#0f172a", borderRadius: 6, marginBottom: 6 }}>
                    <div style={{ fontSize: 12, color: "#3b82f6", marginBottom: 4 }}>
                      {r.start_time.toFixed(1)}s | score: {r.score.toFixed(3)}
                    </div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.4 }}>
                      {r.text.length > 150 ? r.text.substring(0, 150) + "..." : r.text}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Shared styles ────────────────────────────────────────────

const cardStyle: React.CSSProperties = {
  background: "#1e293b",
  borderRadius: 8,
  padding: 20,
};

const sectionTitle: React.CSSProperties = {
  margin: "0 0 12px",
  fontSize: 15,
  color: "#94a3b8",
};

function inputStyle(flex: string): React.CSSProperties {
  return {
    flex,
    padding: "8px 12px",
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: 6,
    color: "#e2e8f0",
    fontSize: 14,
  };
}

function btnStyle(bg: string): React.CSSProperties {
  return {
    padding: "8px 20px",
    background: bg,
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 14,
  };
}
