import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import SearchBar from "../SearchBar";

describe("SearchBar", () => {
  it("renders input and buttons", () => {
    render(<SearchBar onSearch={vi.fn()} onClear={vi.fn()} isLoading={false} />);
    expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
  });

  it("calls onSearch when form is submitted", () => {
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} onClear={vi.fn()} isLoading={false} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.change(input, { target: { value: "red car" } });
    fireEvent.submit(input.closest("form")!);
    expect(onSearch).toHaveBeenCalledWith("red car");
  });

  it("disables input when loading", () => {
    render(<SearchBar onSearch={vi.fn()} onClear={vi.fn()} isLoading={true} />);
    const input = screen.getByPlaceholderText(/search/i);
    expect(input).toBeDisabled();
  });

  it("does not call onSearch with empty query", () => {
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} onClear={vi.fn()} isLoading={false} />);
    const input = screen.getByPlaceholderText(/search/i);
    fireEvent.submit(input.closest("form")!);
    expect(onSearch).not.toHaveBeenCalled();
  });
});
