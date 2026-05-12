import type {
  SearchRequest, SearchResponse, VideoMeta, VideoListResponse,
  GraphData, VideoEvent, Annotation, SearchResult,
} from "../types";

const API_BASE_URL = "/api";
const DEFAULT_TIMEOUT = 30000;

async function request<T>(
  path: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<T> {
  const { timeout = DEFAULT_TIMEOUT, headers, ...init } = options;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...headers,
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(await getErrorMessage(response));
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function requestBlob(
  path: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<Blob> {
  const { timeout = DEFAULT_TIMEOUT, ...init } = options;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(await getErrorMessage(response));
    }

    return await response.blob();
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function getErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    const data = await response.json().catch(() => null) as
      | { detail?: string; message?: string }
      | null;
    if (data?.detail) return data.detail;
    if (data?.message) return data.message;
  }

  const text = await response.text().catch(() => "");
  return text || `${response.status} ${response.statusText}`.trim();
}

function uploadFormData<T>(
  path: string,
  formData: FormData,
  options: {
    timeout?: number;
    onProgress?: (percent: number) => void;
  } = {}
): Promise<T> {
  const { timeout = DEFAULT_TIMEOUT, onProgress } = options;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${path}`);
    xhr.timeout = timeout;
    xhr.responseType = "json";
    xhr.setRequestHeader("Accept", "application/json");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as T);
        return;
      }

      const fallback = xhr.statusText || `HTTP ${xhr.status}`;
      const response = xhr.response as { detail?: string; message?: string } | null;
      reject(new Error(response?.detail || response?.message || fallback));
    };

    xhr.onerror = () => reject(new Error("Network error"));
    xhr.ontimeout = () => reject(new Error("Request timed out"));
    xhr.send(formData);
  });
}

// ── Search ──────────────────────────────────────────────────

export async function searchVideos(
  request: SearchRequest
): Promise<SearchResponse> {
  return requestJson("/search", request);
}

async function requestJson<T>(
  path: string,
  body?: unknown,
  options: Omit<RequestInit, "body"> & { timeout?: number } = {}
): Promise<T> {
  return request<T>(path, {
    method: body === undefined ? "GET" : "POST",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

// ── Video CRUD ──────────────────────────────────────────────

export async function getVideoMeta(videoId: string): Promise<VideoMeta> {
  return request<VideoMeta>(`/video/${videoId}`);
}

export function getVideoStreamUrl(videoId: string): string {
  return `/api/video/${videoId}/stream`;
}

export function getFrameUrl(videoId: string, frameName: string): string {
  return `/api/video/${videoId}/frame/${frameName}`;
}

export function getThumbnailUrl(videoId: string): string {
  return `/api/video/${videoId}/thumbnail`;
}

export async function listVideos(): Promise<VideoMeta[]> {
  const data = await request<VideoListResponse>("/videos");
  return data.videos;
}

export async function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<{ video_id: string; filename: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  return uploadFormData("/ingest-video", formData, {
    timeout: 600000,
    onProgress,
  });
}

export async function processVideo(videoId: string): Promise<void> {
  await requestJson("/process-video", { video_id: videoId });
}

export async function deleteVideo(videoId: string): Promise<void> {
  await request<void>(`/video/${videoId}`, { method: "DELETE" });
}

// ── Batch processing ────────────────────────────────────────

export async function processBatch(videoIds: string[]): Promise<{
  queued: string[];
  queue_size: number;
  errors: { video_id: string; error: string }[];
}> {
  return requestJson("/process-batch", { video_ids: videoIds });
}

export async function getProcessQueue(): Promise<{
  queue: string[];
  queue_size: number;
  is_busy: boolean;
}> {
  return request("/process-queue");
}

// ── Progress SSE ────────────────────────────────────────────

export interface VideoProgress {
  video_id: string;
  stage: string;
  stage_number: number;
  total_stages: number;
  percent: number;
  elapsed_sec?: number;
  done?: boolean;
  error?: boolean;
}

export async function getVideoProgress(videoId: string): Promise<VideoProgress> {
  return request<VideoProgress>(`/video/${videoId}/progress`);
}

export function subscribeToProgress(
  videoId: string,
  onProgress: (progress: VideoProgress) => void,
  onError?: (error: Event) => void
): () => void {
  const eventSource = new EventSource(`/api/video/${videoId}/progress/stream`);

  eventSource.onmessage = (event) => {
    try {
      const progress = JSON.parse(event.data) as VideoProgress;
      onProgress(progress);
      if (progress.done) {
        eventSource.close();
      }
    } catch (e) {
      console.error("Failed to parse progress event:", e);
    }
  };

  eventSource.onerror = (error) => {
    onError?.(error);
    eventSource.close();
  };

  return () => eventSource.close();
}

// ── Find similar ────────────────────────────────────────────

export async function findSimilar(
  segmentId: string,
  topK = 10,
  videoId?: string
): Promise<SearchResult[]> {
  const data = await requestJson<{ results: SearchResult[] }>("/similar", {
    segment_id: segmentId,
    top_k: topK,
    video_id: videoId,
  });
  return data.results;
}

// ── Clip export ─────────────────────────────────────────────

export function getClipExportUrl(
  videoId: string,
  _startTime: number,
  _endTime: number
): string {
  // We'll POST to get the clip, but for direct download we need a form submit
  return `/api/video/${videoId}/clip`;
}

export async function exportClip(
  videoId: string,
  startTime: number,
  endTime: number
): Promise<Blob> {
  return requestBlob(`/video/${videoId}/clip`, {
    method: "POST",
    timeout: 120000,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ start_time: startTime, end_time: endTime }),
  });
}

// ── Knowledge graph ─────────────────────────────────────────

export async function getVideoGraph(videoId: string): Promise<GraphData> {
  return request<GraphData>(`/video/${videoId}/graph`);
}

// ── Events ──────────────────────────────────────────────────

export async function getVideoEvents(
  videoId: string
): Promise<VideoEvent[]> {
  const data = await request<{ events: VideoEvent[] }>(`/video/${videoId}/events`);
  return data.events;
}

// ── Annotations ─────────────────────────────────────────────

export async function getAnnotations(videoId: string): Promise<Annotation[]> {
  const data = await request<{ annotations: Annotation[] }>(`/video/${videoId}/annotations`);
  return data.annotations;
}

export async function createAnnotation(ann: {
  video_id: string;
  start_time: number;
  end_time: number;
  label: string;
  note?: string;
  color?: string;
}): Promise<Annotation> {
  return requestJson<Annotation>("/annotations", ann);
}

export async function deleteAnnotation(annotationId: string): Promise<void> {
  await request<void>(`/annotations/${annotationId}`, { method: "DELETE" });
}

export interface Stats {
  videos: {
    total: number;
    by_status: Record<string, number>;
    total_duration_human: string;
  };
  segments: { total: number };
  live_feeds: { active: number; total: number };
  config: {
    pipeline_mode: string;
    nvidia_cloud: boolean;
    vlm_model: string;
    stt_model: string;
    embedding_model: string;
  };
}

export async function getStats(): Promise<Stats> {
  return request<Stats>("/stats", { timeout: 10000 });
}

// ── Live feeds ──────────────────────────────────────────────

export interface LiveFeedInfo {
  video_id: string;
  url: string;
  state: string;
  chunks_processed: number;
  total_segments: number;
  fps: number;
  started_at: string;
  last_chunk_at: string;
  error: string;
}

export interface LiveWsEvent {
  video_id: string;
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export async function startLiveFeed(
  url: string,
  videoId: string,
  chunkSec = 15
): Promise<{ video_id: string; state: string; message: string }> {
  return requestJson("/live/start", {
    url,
    video_id: videoId,
    chunk_sec: chunkSec,
  });
}

export async function stopLiveFeed(
  videoId: string
): Promise<{ video_id: string; message: string }> {
  return requestJson("/live/stop", { video_id: videoId });
}

export async function getLiveFeedStatus(): Promise<LiveFeedInfo[]> {
  const data = await request<{ feeds: LiveFeedInfo[] }>("/live/status");
  return data.feeds;
}

export function connectLiveWebSocket(
  videoId: string,
  onEvent: (event: LiveWsEvent) => void,
  onClose?: () => void
): { send: (msg: Record<string, unknown>) => void; close: () => void } {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/live/${videoId}`);

  ws.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as LiveWsEvent;
      if (parsed.event_type !== "ping") {
        onEvent(parsed);
      }
    } catch {
      // ignore parse errors
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();

  return {
    send: (msg) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
      }
    },
    close: () => ws.close(),
  };
}

// ── Browser camera (phone / screen share) ───────────────────

export async function uploadBrowserChunk(
  videoId: string,
  chunkIndex: number,
  chunkStart: number,
  blob: Blob
): Promise<{ segments_stored: number; keyframes_extracted: number }> {
  const formData = new FormData();
  formData.append("file", blob, `chunk_${chunkIndex}.webm`);

  return request(
    `/live/browser-chunk?video_id=${encodeURIComponent(videoId)}&chunk_index=${chunkIndex}&chunk_start=${chunkStart}`,
    {
      method: "POST",
      body: formData,
      timeout: 120000,
    }
  );
}

export async function stopBrowserFeed(
  videoId: string
): Promise<{ total_chunks: number; message: string }> {
  return requestJson(
    `/live/browser-stop?video_id=${encodeURIComponent(videoId)}`,
    undefined,
    { method: "POST" }
  );
}
