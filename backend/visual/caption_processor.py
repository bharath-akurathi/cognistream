"""
CogniStream — Caption Processor

Intelligence layer for the Visual Narrative Engine.  Two responsibilities:

1. **PromptLibrary** — Four focused prompt templates that instruct the VLM
   to return semi-structured text (labelled fields, comma-separated lists).
   Prompts are deliberately short because small VLMs (Moondream2, LLaVA-7B)
   follow brief, direct instructions more reliably than long ones.

2. **CaptionProcessor** — Parses the raw VLM text from each pass into
   structured Python objects.  Since VLMs don't reliably produce valid JSON,
   parsing uses a three-tier fallback strategy:
       a) Try JSON extraction (braces / brackets in the response).
       b) Try labelled-field regex (e.g. "Objects: car, tree, person").
       c) Fall back to using the raw text as-is.

Usage:
    processor = CaptionProcessor()
    caption = processor.build_caption(
        keyframe=kf,
        scene_raw="A busy intersection with traffic ...",
        objects_raw="car, traffic light, pedestrian, crosswalk",
        activity_raw="Vehicles are stopped at a red light ...",
        anomaly_raw="None",
    )
    # caption.objects == ["car", "traffic light", "pedestrian", "crosswalk"]
"""

from __future__ import annotations

import json
import logging
import re

from backend.db.models import Keyframe, VisualCaption

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PromptLibrary:
    """Static prompt templates for each VLM analysis pass.

    Design principles:
    - One task per prompt — small VLMs degrade on compound instructions.
    - Ask for a specific output format (list, yes/no + explanation).
    - Keep prompts under ~60 words — longer prompts hurt small models.
    """

    @staticmethod
    def scene_prompt() -> str:
        """Describe the overall scene in the image."""
        return (
            "Describe this image in 2-3 sentences. "
            "Include the setting, lighting, and overall atmosphere. "
            "Be specific and factual. Do not speculate."
        )

    @staticmethod
    def objects_prompt() -> str:
        """List prominent objects / entities visible in the image."""
        return (
            "List all prominent objects and entities visible in this image. "
            "Return ONLY a comma-separated list, nothing else. "
            "Example: car, tree, person, traffic light"
        )

    @staticmethod
    def activity_prompt() -> str:
        """Describe the primary action or event happening in the image."""
        return (
            "What action or activity is happening in this image? "
            "Describe the primary activity in one sentence. "
            "If no clear activity is visible, say 'static scene'."
        )

    @staticmethod
    def anomaly_prompt() -> str:
        """Check for anything unusual, suspicious, or out of place."""
        return (
            "Is there anything unusual, suspicious, or out of place in this image? "
            "If yes, describe it in one sentence. "
            "If nothing is unusual, respond with exactly: none"
        )

    @staticmethod
    def combined_prompt() -> str:
        """Single-pass prompt that extracts all four fields at once.

        ~75% faster than four separate calls.  Designed for small VLMs
        that can handle brief structured output.
        """
        return (
            "Analyze this image. Respond with EXACTLY this format:\n"
            "SCENE: [2-3 sentence description of setting and atmosphere]\n"
            "OBJECTS: [comma-separated list of visible objects]\n"
            "ACTIVITY: [one sentence describing what is happening]\n"
            "ANOMALY: [anything unusual, or 'none']"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Response parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CaptionProcessor:
    """Parse raw VLM outputs and assemble a structured VisualCaption."""

    def build_caption(
        self,
        keyframe: Keyframe,
        scene_raw: str,
        objects_raw: str,
        activity_raw: str,
        anomaly_raw: str,
    ) -> VisualCaption:
        """Combine four raw VLM responses into one structured caption.

        Each response goes through its own parser.  A parsing failure
        never raises — it logs a warning and uses a safe default.
        """
        return VisualCaption(
            keyframe=keyframe,
            scene_description=self._parse_scene(scene_raw),
            objects=self._parse_objects(objects_raw),
            activity=self._parse_activity(activity_raw),
            anomaly=self._parse_anomaly(anomaly_raw),
        )

    def to_json(self, caption: VisualCaption) -> dict:
        """Serialise a VisualCaption to a JSON-compatible dict.

        Useful for logging, storage, and API responses.
        """
        return {
            "video_id": caption.keyframe.video_id,
            "segment_index": caption.keyframe.segment_index,
            "frame_number": caption.keyframe.frame_number,
            "timestamp": caption.keyframe.timestamp,
            "frame_path": caption.keyframe.file_path,
            "scene_description": caption.scene_description,
            "objects": caption.objects,
            "activity": caption.activity,
            "anomaly": caption.anomaly,
        }

    def captions_to_json(self, captions: list[VisualCaption]) -> list[dict]:
        """Serialise a batch of captions."""
        return [self.to_json(c) for c in captions]

    # ── individual parsers ──────────────────────────────────────

    def _parse_scene(self, raw: str) -> str:
        """Parse scene description — accept as-is after cleanup."""
        text = self._clean(raw)
        if not text:
            logger.debug("Empty scene response — using fallback.")
            return "No scene description available."
        return text

    def _parse_objects(self, raw: str) -> list[str]:
        """Parse a comma-separated object list from the VLM response.

        Fallback strategy:
            1. Try to extract a JSON array if the model wrapped it in [].
            2. Split on commas (the expected format).
            3. Split on newlines if the model used a bulleted list.
        """
        text = self._clean(raw)
        if not text:
            return []

        # Strategy 1: JSON array
        objects = self._try_json_array(text)
        if objects is not None:
            return objects

        # Strategy 2: comma-separated
        if "," in text:
            return self._split_and_clean(text, sep=",")

        # Strategy 3: newline / bullet list
        #   - car\n- tree\n- person
        lines = text.splitlines()
        if len(lines) > 1:
            return self._split_and_clean("\n".join(lines), sep="\n")

        # Single item
        cleaned = self._strip_bullets(text)
        return [cleaned] if cleaned else []

    def _parse_activity(self, raw: str) -> str:
        """Parse activity description — accept as-is after cleanup."""
        text = self._clean(raw)
        if not text:
            return "static scene"
        return text

    def _parse_anomaly(self, raw: str) -> str | None:
        """Parse anomaly response.

        Returns None if the model indicated nothing unusual, otherwise
        returns the anomaly description string.
        """
        text = self._clean(raw)
        if not text:
            return None

        # The prompt asks the model to say "none" if nothing unusual
        normalised = text.lower().strip(" .\n")
        negative_signals = {
            "none",
            "no",
            "nothing",
            "nothing unusual",
            "no anomaly",
            "no anomalies",
            "none detected",
            "n/a",
            "na",
            "not applicable",
        }
        if normalised in negative_signals:
            return None

        # Some models prefix with "No, ..." or "There is nothing ..."
        if normalised.startswith(("no,", "no.", "no ", "nothing", "there is nothing")):
            return None

        return text

    def build_caption_from_combined(
        self,
        keyframe: Keyframe,
        combined_raw: str,
    ) -> VisualCaption:
        """Parse a single-pass combined VLM response into a VisualCaption.

        Expected format:
            SCENE: ...
            OBJECTS: ...
            ACTIVITY: ...
            ANOMALY: ...

        Falls back gracefully if the model doesn't follow the format exactly.
        """
        fields = self._parse_combined_fields(combined_raw)

        return VisualCaption(
            keyframe=keyframe,
            scene_description=self._parse_scene(fields.get("scene", "")),
            objects=self._parse_objects(fields.get("objects", "")),
            activity=self._parse_activity(fields.get("activity", "")),
            anomaly=self._parse_anomaly(fields.get("anomaly", "")),
        )

    @staticmethod
    def _parse_combined_fields(raw: str) -> dict[str, str]:
        """Extract labelled fields from a combined VLM response."""
        fields: dict[str, str] = {}
        labels = ["scene", "objects", "activity", "anomaly"]

        for label in labels:
            # Match "LABEL:" or "Label:" at start of line
            pattern = rf"(?i)^{label}\s*:\s*(.+?)(?=\n(?:{'|'.join(labels)})\s*:|$)"
            match = re.search(pattern, raw, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if match:
                fields[label] = match.group(1).strip()

        # Fallback: if no fields parsed, treat entire text as scene description
        if not fields:
            fields["scene"] = raw.strip()

        return fields

    # ── text utilities ──────────────────────────────────────────

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip whitespace and remove enclosing quotes."""
        text = raw.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        return text

    @staticmethod
    def _strip_bullets(text: str) -> str:
        """Remove leading bullet characters (-, *, •, numbered)."""
        return re.sub(r"^[\s\-\*\•\d\.]+", "", text).strip()

    @classmethod
    def _split_and_clean(cls, text: str, sep: str) -> list[str]:
        """Split text, strip bullets/whitespace, drop empty entries."""
        items: list[str] = []
        for chunk in text.split(sep):
            cleaned = cls._strip_bullets(chunk.strip())
            if cleaned:
                items.append(cleaned)
        return items

    @staticmethod
    def _try_json_array(text: str) -> list[str] | None:
        """Try to extract a JSON array from the text.

        Looks for the first ``[...]`` in the string and attempts to parse it.
        Returns None if no valid array is found.
        """
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            return None
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return None
