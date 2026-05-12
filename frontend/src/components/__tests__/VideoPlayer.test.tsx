import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { createRef } from "react";
import VideoPlayer from "../VideoPlayer";

describe("VideoPlayer", () => {
  const defaultProps = {
    videoUrl: null,
    videoRef: createRef<HTMLVideoElement>(),
    currentTime: 0,
    duration: 0,
    isPlaying: false,
    onTimeUpdate: vi.fn(),
    onLoadedMetadata: vi.fn(),
    togglePlay: vi.fn(),
    onSeek: vi.fn(),
    activeResult: null,
  };

  it("shows placeholder when no video URL", () => {
    render(<VideoPlayer {...defaultProps} />);
    expect(screen.getByText(/select a video/i)).toBeInTheDocument();
  });

  it("renders video element when URL provided", () => {
    render(<VideoPlayer {...defaultProps} videoUrl="/api/video/123/stream" />);
    const video = document.querySelector("video");
    expect(video).toBeInTheDocument();
    expect(video?.src).toContain("/api/video/123/stream");
  });

  it("renders play button", () => {
    render(<VideoPlayer {...defaultProps} videoUrl="/stream" />);
    const btn = screen.getByRole("button", { name: /play video/i });
    expect(btn).toBeInTheDocument();
  });

  it("calls togglePlay when play button is clicked", () => {
    render(<VideoPlayer {...defaultProps} videoUrl="/stream" />);
    fireEvent.click(screen.getByRole("button", { name: /play video/i }));
    expect(defaultProps.togglePlay).toHaveBeenCalled();
  });

  it("seeks when the scrubber is pressed", () => {
    const onSeek = vi.fn();
    render(
      <VideoPlayer
        {...defaultProps}
        videoUrl="/stream"
        duration={120}
        currentTime={15}
        onSeek={onSeek}
      />
    );

    const slider = screen.getByRole("slider", { name: /video seek bar/i });
    Object.defineProperty(slider, "getBoundingClientRect", {
      value: () => ({
        left: 0,
        width: 200,
        top: 0,
        height: 24,
        right: 200,
        bottom: 24,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }),
    });

    fireEvent.pointerDown(slider, { clientX: 100 });
    expect(onSeek).toHaveBeenCalledWith(60);
  });

  it("shows a preview timestamp while hovering over the seek bar", () => {
    render(
      <VideoPlayer
        {...defaultProps}
        videoUrl="/stream"
        duration={120}
      />
    );

    const slider = screen.getByRole("slider", { name: /video seek bar/i });
    Object.defineProperty(slider, "getBoundingClientRect", {
      value: () => ({
        left: 0,
        width: 240,
        top: 0,
        height: 24,
        right: 240,
        bottom: 24,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }),
    });

    fireEvent.pointerMove(slider, { clientX: 60 });
    expect(screen.getByText("0:30")).toBeInTheDocument();
  });
});
