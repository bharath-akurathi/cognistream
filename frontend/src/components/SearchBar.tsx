import { useState, type FormEvent, type KeyboardEvent } from "react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  onClear: () => void;
  isLoading: boolean;
}

export default function SearchBar({ onSearch, onClear, isLoading }: SearchBarProps) {
  const [input, setInput] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (input.trim()) onSearch(input);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      setInput("");
      onClear();
    }
  }

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.inputWrapper}>
        <svg style={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder='Search video content — e.g. "when did the red car arrive"'
          style={styles.input}
          disabled={isLoading}
        />
        {input && (
          <button
            type="button"
            onClick={() => { setInput(""); onClear(); }}
            style={styles.clearBtn}
            title="Clear search"
          >
            &times;
          </button>
        )}
      </div>
      <button type="submit" style={styles.button} disabled={isLoading || !input.trim()}>
        {isLoading ? "Searching..." : "Search"}
      </button>
    </form>
  );
}

const styles: Record<string, React.CSSProperties> = {
  form: {
    display: "flex",
    gap: "12px",
    width: "100%",
  },
  inputWrapper: {
    flex: 1,
    position: "relative",
    display: "flex",
    alignItems: "center",
  },
  icon: {
    position: "absolute",
    left: "14px",
    width: "18px",
    height: "18px",
    color: "#94a3b8",
    pointerEvents: "none",
  },
  input: {
    width: "100%",
    padding: "14px 40px 14px 44px",
    fontSize: "15px",
    border: "2px solid #e2e8f0",
    borderRadius: "12px",
    outline: "none",
    background: "#f8fafc",
    color: "#1e293b",
    transition: "border-color 0.2s",
  },
  clearBtn: {
    position: "absolute",
    right: "12px",
    background: "none",
    border: "none",
    fontSize: "20px",
    color: "#94a3b8",
    cursor: "pointer",
    padding: "0 4px",
    lineHeight: 1,
  },
  button: {
    padding: "14px 28px",
    fontSize: "15px",
    fontWeight: 600,
    color: "#fff",
    background: "#3b82f6",
    border: "none",
    borderRadius: "12px",
    cursor: "pointer",
    whiteSpace: "nowrap",
    transition: "background 0.2s",
  },
};
