import { useCallback, useMemo, useState } from "react";
import SearchBar from "./components/SearchBar";
import VideoPlayer from "./components/VideoPlayer";
import TimelineMarkers from "./components/TimelineMarkers";
import ResultsPanel from "./components/ResultsPanel";
import VideoList from "./components/VideoList";
import VideoUpload from "./components/VideoUpload";
import KnowledgeGraph from "./components/KnowledgeGraph";
import EventTimeline from "./components/EventTimeline";
import LiveView from "./components/LiveView";
import StatsPanel from "./components/StatsPanel";
import { useSearch } from "./hooks/useSearch";
import { useVideo } from "./hooks/useVideo";
import { getVideoStreamUrl, processVideo, findSimilar, exportClip } from "./api/client";
import type { VideoMeta, SearchResult } from "./types";

type View = "list" | "search" | "global-search" | "live";

export default function App() {
  // Navigation state
  const [view, setView] = useState<View>("list");
  const [selectedVideo, setSelectedVideo] = useState<VideoMeta | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [showGraph, setShowGraph] = useState(false);

  // Search hook - pass video_id when in scoped search view
  const searchVideoId = view === "search" ? selectedVideo?.video_id : undefined;
  const { results, isLoading, error, query, search, clear, setResults } = useSearch(
    searchVideoId
  );
  const {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    seekTo,
    onTimeUpdate,
    onLoadedMetadata,
    togglePlay,
  } = useVideo();

  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  // Video URL: use selected video when in search view
  const videoUrl = useMemo(() => {
    if (selectedVideo && view === "search") {
      return getVideoStreamUrl(selectedVideo.video_id);
    }
    return null;
  }, [selectedVideo, view]);

  const activeResult = activeIndex !== null ? results[activeIndex] ?? null : null;

  const handleResultClick = useCallback(
    (index: number) => {
      setActiveIndex(index);
      const result = results[index];
      if (result) seekTo(result.start_time, { autoplay: true });
    },
    [results, seekTo]
  );

  const handleSearch = useCallback(
    (q: string) => {
      setActiveIndex(null);
      search(q);
    },
    [search]
  );

  const handleClear = useCallback(() => {
    setActiveIndex(null);
    clear();
  }, [clear]);

  const handleSelectVideo = useCallback((video: VideoMeta) => {
    setSelectedVideo(video);
    setActiveIndex(null);
    setShowGraph(false);
    clear();
    setView("search");
  }, [clear]);

  const handleBackToList = useCallback(() => {
    setView("list");
    setSelectedVideo(null);
    setActiveIndex(null);
    setShowGraph(false);
    clear();
  }, [clear]);

  const handleGlobalSearch = useCallback(() => {
    setSelectedVideo(null);
    setActiveIndex(null);
    clear();
    setView("global-search");
  }, [clear]);

  const handleOpenLive = useCallback(() => {
    setSelectedVideo(null);
    setActiveIndex(null);
    clear();
    setView("live");
  }, [clear]);

  // Find similar handler
  const handleFindSimilar = useCallback(async (segmentId: string) => {
    try {
      const similar = await findSimilar(segmentId, 10, searchVideoId);
      setResults(similar);
      setActiveIndex(null);
    } catch (err) {
      console.error("Find similar failed:", err);
    }
  }, [searchVideoId, setResults]);

  // Clip export handler
  const handleExportClip = useCallback(async (result: SearchResult) => {
    try {
      const blob = await exportClip(result.video_id, result.start_time, result.end_time);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `clip_${result.start_time.toFixed(1)}-${result.end_time.toFixed(1)}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Clip export failed:", err);
    }
  }, []);

  // Track refresh trigger for VideoList
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadComplete = useCallback((videoId: string, shouldProcess: boolean) => {
    setShowUpload(false);
    setRefreshTrigger((prev) => prev + 1);
    if (shouldProcess) {
      processVideo(videoId).catch((err) => {
        console.error("Failed to start processing:", err);
      });
    }
  }, []);

  // When clicking a global search result, navigate to that video
  const handleGlobalResultClick = useCallback(
    (index: number) => {
      setActiveIndex(index);
      const result = results[index];
      if (result) {
        setSelectedVideo({
          video_id: result.video_id,
          filename: "",
          duration_sec: 0,
          status: "PROCESSED",
          created_at: "",
        });
        setView("search");
      }
    },
    [results]
  );

  const isSearchView = view === "search" || view === "global-search";
  const showBackBtn = isSearchView || view === "live";

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          {showBackBtn && (
            <button onClick={handleBackToList} style={styles.backBtn}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back
            </button>
          )}
          <div>
            <h1 style={styles.title}>
              <span style={styles.titleAccent}>Cogni</span>Stream
            </h1>
            <p style={styles.subtitle}>
              {view === "search" && selectedVideo
                ? selectedVideo.filename || "Video Search"
                : view === "global-search"
                ? "Search All Videos"
                : view === "live"
                ? "Live Video Feeds"
                : "Multimodal Video Retrieval"}
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {view === "list" && (
            <button onClick={handleOpenLive} style={styles.liveBtn}>
              <span style={styles.liveDot} />
              Live
            </button>
          )}
          {view === "search" && selectedVideo?.status === "PROCESSED" && (
            <button
              onClick={() => setShowGraph(!showGraph)}
              style={styles.graphBtn}
            >
              {showGraph ? "Hide Graph" : "Knowledge Graph"}
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      {view === "live" ? (
        <LiveView onBack={handleBackToList} />
      ) : view === "list" ? (
        <>
        <StatsPanel />
        <VideoList
          onSelectVideo={handleSelectVideo}
          onUploadClick={() => setShowUpload(true)}
          onGlobalSearch={handleGlobalSearch}
          refreshTrigger={refreshTrigger}
        />
        </>
      ) : view === "global-search" ? (
        <>
          <section style={styles.searchSection}>
            <SearchBar onSearch={handleSearch} onClear={handleClear} isLoading={isLoading} />
          </section>
          <div style={styles.resultsFullWidth}>
            <ResultsPanel
              results={results}
              query={query}
              isLoading={isLoading}
              error={error}
              activeIndex={activeIndex}
              onResultClick={handleGlobalResultClick}
              onFindSimilar={handleFindSimilar}
              onExportClip={handleExportClip}
            />
          </div>
        </>
      ) : (
        <>
          {/* Knowledge Graph */}
          {showGraph && selectedVideo && (
            <KnowledgeGraph
              videoId={selectedVideo.video_id}
              onClose={() => setShowGraph(false)}
            />
          )}

          {/* Search */}
          <section style={styles.searchSection}>
            <SearchBar onSearch={handleSearch} onClear={handleClear} isLoading={isLoading} />
          </section>

          {/* Player + Results */}
          <main style={styles.main}>
            {/* Left column: video + timeline */}
            <div style={styles.playerColumn}>
              <VideoPlayer
                videoUrl={videoUrl}
                videoRef={videoRef}
                currentTime={currentTime}
                duration={duration}
                isPlaying={isPlaying}
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={onLoadedMetadata}
                togglePlay={togglePlay}
                onSeek={seekTo}
                activeResult={activeResult}
              />
              <TimelineMarkers
                results={results}
                duration={duration}
                currentTime={currentTime}
                activeIndex={activeIndex}
                onMarkerClick={handleResultClick}
              />
              {/* Event timeline + annotations */}
              {selectedVideo && duration > 0 && (
                <EventTimeline
                  videoId={selectedVideo.video_id}
                  duration={duration}
                  currentTime={currentTime}
                  onSeek={(time) => seekTo(time, { autoplay: true })}
                />
              )}
            </div>

            {/* Right column: results */}
            <div style={styles.resultsColumn}>
              <ResultsPanel
                results={results}
                query={query}
                isLoading={isLoading}
                error={error}
                activeIndex={activeIndex}
                onResultClick={handleResultClick}
                onFindSimilar={handleFindSimilar}
                onExportClip={handleExportClip}
              />
            </div>
          </main>
        </>
      )}

      {/* Upload modal */}
      {showUpload && (
        <VideoUpload
          onComplete={handleUploadComplete}
          onCancel={() => setShowUpload(false)}
        />
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  app: {
    maxWidth: "1280px",
    margin: "0 auto",
    padding: "24px clamp(12px, 3vw, 32px)",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    color: "#1e293b",
    minHeight: "100vh",
  },
  header: {
    marginBottom: "24px",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    flexWrap: "wrap" as const,
    gap: "8px",
  },
  headerLeft: {
    display: "flex",
    alignItems: "flex-start",
    gap: "16px",
  },
  backBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 14px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: 500,
    color: "#475569",
    cursor: "pointer",
    marginTop: "4px",
  },
  graphBtn: {
    padding: "8px 14px",
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: 500,
    color: "#6366f1",
    cursor: "pointer",
    marginTop: "4px",
  },
  title: {
    fontSize: "28px",
    fontWeight: 800,
    margin: 0,
    letterSpacing: "-0.02em",
  },
  titleAccent: {
    color: "#3b82f6",
  },
  subtitle: {
    fontSize: "14px",
    color: "#64748b",
    margin: "4px 0 0 0",
  },
  searchSection: {
    marginBottom: "24px",
  },
  main: {
    display: "flex",
    gap: "28px",
    alignItems: "flex-start",
    flexWrap: "wrap" as const,
  },
  playerColumn: {
    flex: "1 1 400px",
    minWidth: 0,
  },
  resultsColumn: {
    flex: "1 1 300px",
    minWidth: 0,
  },
  resultsFullWidth: {
    maxWidth: "800px",
  },
  liveBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 14px",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: 600,
    color: "#dc2626",
    cursor: "pointer",
    marginTop: "4px",
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#dc2626",
    animation: "pulse 2s infinite",
  },
};
