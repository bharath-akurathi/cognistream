import { describe, expect, it } from "vitest";
import { orderSearchResults } from "./useSearch";
import type { SearchResult } from "../types";

function result(
  segmentId: string,
  sourceType: SearchResult["source_type"],
  score: number,
  speechSnippet?: string
): SearchResult {
  return {
    video_id: "video-1",
    segment_id: segmentId,
    start_time: 0,
    end_time: 1,
    text: segmentId,
    source_type: sourceType,
    score,
    speech_snippet: speechSnippet,
  };
}

describe("orderSearchResults", () => {
  it("puts speech results before video results in All mode", () => {
    const ordered = orderSearchResults(
      [
        result("visual-high", "visual", 0.99),
        result("audio-mid", "audio", 0.75),
        result("fused-speech", "fused", 0.70, "<mark>hello</mark>"),
        result("speech-low", "speech", 0.60),
        result("fused-video", "fused", 0.95),
      ],
      "hybrid"
    );

    expect(ordered.map((r) => r.segment_id)).toEqual([
      "audio-mid",
      "fused-speech",
      "speech-low",
      "visual-high",
      "fused-video",
    ]);
  });

  it("keeps backend order outside All mode", () => {
    const results = [
      result("visual", "visual", 0.99),
      result("speech", "speech", 0.60),
    ];

    expect(orderSearchResults(results, "visual")).toBe(results);
    expect(orderSearchResults(results, "speech")).toBe(results);
  });
});
