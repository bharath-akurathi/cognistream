import { useCallback, useState } from "react";
import { searchVideos } from "../api/client";
import type { SearchResult, SearchMode } from "../types";

interface UseSearchReturn {
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  query: string;
  searchMode: SearchMode;
  setSearchMode: (mode: SearchMode) => void;
  search: (query: string) => Promise<void>;
  clear: () => void;
  setResults: (results: SearchResult[]) => void;
}

export function useSearch(videoId?: string): UseSearchReturn {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<SearchMode>("hybrid");

  const search = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;

    setQuery(trimmed);
    setIsLoading(true);
    setError(null);

    try {
      const response = await searchVideos({
        query: trimmed,
        video_id: videoId,
        top_k: 20,
        search_mode: searchMode,
      });
      setResults(orderSearchResults(response.results, searchMode));
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Search failed. Is the backend running?";
      setError(message);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [videoId, searchMode]);

  const clear = useCallback(() => {
    setResults([]);
    setQuery("");
    setError(null);
  }, []);

  return { results, isLoading, error, query, searchMode, setSearchMode, search, clear, setResults };
}

export function orderSearchResults(results: SearchResult[], searchMode: SearchMode): SearchResult[] {
  if (searchMode !== "hybrid") {
    return results;
  }

  return [...results].sort((a, b) => {
    const aGroup = isSpeechResult(a) ? 0 : 1;
    const bGroup = isSpeechResult(b) ? 0 : 1;
    if (aGroup !== bGroup) return aGroup - bGroup;
    return b.score - a.score;
  });
}

function isSpeechResult(result: SearchResult): boolean {
  return (
    result.source_type === "speech" ||
    result.source_type === "audio" ||
    Boolean(result.speech_snippet)
  );
}
