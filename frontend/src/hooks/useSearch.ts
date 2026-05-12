import { useCallback, useState } from "react";
import { searchVideos } from "../api/client";
import type { SearchResult } from "../types";

interface UseSearchReturn {
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  query: string;
  search: (query: string) => Promise<void>;
  clear: () => void;
  setResults: (results: SearchResult[]) => void;
}

export function useSearch(videoId?: string): UseSearchReturn {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

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
      });
      setResults(response.results);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Search failed. Is the backend running?";
      setError(message);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [videoId]);

  const clear = useCallback(() => {
    setResults([]);
    setQuery("");
    setError(null);
  }, []);

  return { results, isLoading, error, query, search, clear, setResults };
}
