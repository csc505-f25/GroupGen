"""
Official VAK inventory — paper scoring method.

**From the research paper:**
  1. Count how many **A**, **B**, and **C** choices the student selected (30 items).
  2. Whichever letter has the **highest count** is their learning style:
       - mostly **A** → **Visual**
       - mostly **B** → **Auditory**
       - mostly **C** → **Kinesthetic**

On every question, option **(a)** is an A, **(b)** is a B, **(c)** is a C.

Each row in ``_VAK_OPTIONS`` is:
  (question number, paper letter, exact option text in Google Forms)

Example question 1:
  (1, "A", "read the instructions first")
  (1, "B", "listen to an explanation from someone who has used it before")
  (1, "C", "go ahead and have a go, i can figure it out as i use it")
"""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Dict, Iterable, Literal, Optional

PaperLetter = Literal["A", "B", "C"]

# (question #, A|B|C per paper, exact option wording)
_VAK_OPTIONS: list[tuple[int, PaperLetter, str]] = [
    (1, "A", "read the instructions first"),
    (1, "B", "listen to an explanation from someone who has used it before"),
    (1, "C", "go ahead and have a go, i can figure it out as i use it"),
    (2, "A", "look at a map"),
    (2, "B", "ask for spoken directions"),
    (2, "C", "follow my nose and maybe use a compass"),
    (3, "A", "follow a written recipe"),
    (3, "B", "call a friend for an explanation"),
    (3, "C", "follow my instincts, testing as i cook"),
    (4, "A", "write instructions down for them"),
    (4, "B", "give them a verbal explanation"),
    (4, "C", "demonstrate first and then let them have a go"),
    (5, "A", "watch how i do it"),
    (5, "B", "listen to me explain"),
    (5, "C", "you have a go"),
    (6, "A", "going to museums and galleries"),
    (6, "B", "listening to music and talking to my friends"),
    (6, "C", "playing sport or doing diy"),
    (7, "A", "imagine what they would look like on"),
    (7, "B", "discuss them with the shop staff"),
    (7, "C", "try them on and test them out"),
    (8, "A", "read lots of brochures"),
    (8, "B", "listen to recommendations from friends"),
    (8, "C", "imagine what it would be like to be there"),
    (9, "A", "read reviews in newspapers and magazines"),
    (9, "B", "discuss what i need with my friends"),
    (9, "C", "test-drive lots of different types"),
    (10, "A", "watching what the teacher is doing"),
    (10, "B", "talking through with the teacher exactly what i'm supposed to do"),
    (10, "C", "giving it a try myself and work it out as i go"),
    (11, "A", "imagine what the food will look like"),
    (11, "B", "talk through the options in my head or with my partner"),
    (11, "C", "imagine what the food will taste like"),
    (12, "A", "watching the band members and other people in the audience"),
    (12, "B", "listening to the lyrics and the beats"),
    (12, "C", "moving in time with the music"),
    (13, "A", "focus on the words or the pictures in front of me"),
    (13, "B", "discuss the problem and the possible solutions in my head"),
    (13, "C", "move around a lot, fiddle with pens and pencils and touch things"),
    (14, "A", "their colors and how they look"),
    (14, "B", "the descriptions the sales-people give me"),
    (14, "C", "their textures and what it feels like to touch them"),
    (15, "A", "looking at something"),
    (15, "B", "being spoken to"),
    (15, "C", "doing something"),
    (16, "A", "visualize the worst-case scenarios"),
    (16, "B", "talk over in my head what worries me most"),
    (16, "C", "can't sit still, fiddle and move around constantly"),
    (17, "A", "how they look"),
    (17, "B", "what they say to me"),
    (17, "C", "how they make me feel"),
    (18, "A", "write lots of revision notes and diagrams"),
    (18, "B", "talk over my notes, alone or with other people"),
    (18, "C", "imagine making the movement or creating the formula"),
    (19, "A", "show them what i mean"),
    (19, "B", "explain to them in different ways until they understand"),
    (19, "C", "encourage them to try and talk them through my idea as they do it"),
    (20, "A", "watching films, photography, looking at art or people watching"),
    (20, "B", "listening to music, the radio or talking to friends"),
    (20, "C", "taking part in sporting activities, eating fine foods and wines or dancing"),
    (21, "A", "watching television"),
    (21, "B", "talking to friends"),
    (21, "C", "doing physical activity or making things"),
    (22, "A", "arrange a face to face meeting"),
    (22, "B", "talk to them on the telephone"),
    (22, "C", "try to get together whilst doing something else, such as an activity or a meal"),
    (23, "A", "look and dress"),
    (23, "B", "sound and speak"),
    (23, "C", "stand and move"),
    (24, "A", "keep replaying in my mind what it is that has upset me"),
    (24, "B", "raise my voice and tell people how i feel"),
    (24, "C", "stamp about, slam doors and physically demonstrate my anger"),
    (25, "A", "faces"),
    (25, "B", "names"),
    (25, "C", "things i have done"),
    (26, "A", "they avoid looking at you"),
    (26, "B", "their voices changes"),
    (26, "C", "they give me funny vibes"),
    (27, "A", "i say it's great to see you"),
    (27, "B", "i say it's great to hear from you"),
    (27, "C", "i give them a hug or a handshake"),
    (28, "A", "writing notes or keeping printed details"),
    (28, "B", "saying them aloud or repeating words and key points in my head"),
    (28, "C", "doing and practicing the activity or imagining it being done"),
    (29, "A", "writing a letter"),
    (29, "B", "complaining over the phone"),
    (29, "C", "taking the item back to the store or posting it to head office"),
    (30, "A", "i see what you mean"),
    (30, "B", "i hear what you are saying"),
    (30, "C", "i know how you feel"),
]

LETTER_TO_LEARNING_STYLE: Dict[PaperLetter, str] = {
    "A": "Visual",
    "B": "Auditory",
    "C": "Kinesthetic",
}


def _is_blank(val: object) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return not s or s.lower() in {"nan", "none"}


# Wrapping punctuation Google Forms may add around options (not word apostrophes).
_STRIP_CHARS = '"!\u201c\u201d\u201e\u00ab\u00bb'  # " " « » and !


def _apply_vak_text_fixes(s: str) -> str:
    """Common Google Forms / student typos before catalog lookup."""
    s = re.sub(r"\balot\b", "a lot", s)
    s = re.sub(r"\bits great to (see|hear)\b", r"it's great to \1", s)
    s = re.sub(r"\bplaying sports\b", "playing sport", s)
    s = re.sub(r",\s*going to the gym or\b", " or", s)
    s = re.sub(r"\bfood would look\b", "food will look", s)
    s = re.sub(r"\bfood would taste\b", "food will taste", s)
    s = re.sub(r"\bstand and more\b", "stand and move", s)
    s = re.sub(r"\bcan't stand still\b", "can't sit still", s)
    s = re.sub(
        r"talk through the options with my head or with someone else",
        "talk through the options in my head or with my partner",
        s,
    )
    s = re.sub(
        r"talk through the options in my head or with someone else",
        "talk through the options in my head or with my partner",
        s,
    )
    s = re.sub(
        r",\s*i fiddle and move around constantly",
        ", fiddle and move around constantly",
        s,
    )
    return s


def normalize_vak_text(text: str) -> str:
    """Lowercase; strip a)/b)/c); remove wrapping quotes and ! for export matching."""
    if _is_blank(text):
        return ""
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"^[abc]\)\s*", "", s)
    for ch in _STRIP_CHARS:
        s = s.replace(ch, "")
    s = re.sub(r"\s+", " ", s).strip()
    return _apply_vak_text_fixes(s)


# Wording that differs from the paper text but maps to the same A/B/C option.
_VAK_TEXT_ALIASES: list[tuple[str, PaperLetter]] = [
    # Auditory — form often says "friends" instead of paper Q7b/Q8b phrasing
    ("discuss them with my friends", "B"),
    # Kinesthetic Q6c — form adds gym / plural "sports"
    ("playing sports, going to the gym or doing diy", "C"),
    # Visual Q11a — form uses "would" instead of paper "will"
    ("imagine what the food would look like", "A"),
    # Visual Q9a — form says "magazines or online" instead of "newspapers and magazines"
    ("read reviews in magazines or online", "A"),
    # Auditory Q11b — form says "someone else" instead of "my partner"
    ("talk through the options in my head or with someone else", "B"),
    # Kinesthetic Q16c — form inserts "I" before "fiddle"
    ("can't sit still, i fiddle and move around constantly", "C"),
]


def _build_lookup() -> Dict[str, PaperLetter]:
    lookup: Dict[str, PaperLetter] = {}
    for _q, letter, phrase in _VAK_OPTIONS:
        lookup[normalize_vak_text(phrase)] = letter
    for alias, letter in _VAK_TEXT_ALIASES:
        lookup[normalize_vak_text(alias)] = letter
    return lookup


_TEXT_TO_LETTER: Dict[str, PaperLetter] = _build_lookup()


def classify_vak_letter(raw: object) -> Optional[PaperLetter]:
    """
    Map one Google Form cell to paper letter A, B, or C.

    Accepts the exact option text, or a lone ``A`` / ``B`` / ``C``.
    """
    if _is_blank(raw):
        return None
    text = str(raw).strip()
    bare = text.upper()
    if bare in ("A", "B", "C"):
        return bare  # type: ignore[return-value]

    norm = normalize_vak_text(text)
    return _TEXT_TO_LETTER.get(norm)


def letters_from_vak_cell(raw: object) -> list[PaperLetter]:
    """
    Letters scored from one cell — whole-cell match, or comma-split fragments
    when a ragged CSV glued multiple options into one field.
    """
    letter = classify_vak_letter(raw)
    if letter:
        return [letter]
    if _is_blank(raw):
        return []
    text = str(raw).strip()
    if "," not in text:
        return []
    found: list[PaperLetter] = []
    for part in re.split(r",\s*", text):
        part = part.strip()
        if not part:
            continue
        sub = classify_vak_letter(part)
        if sub:
            found.append(sub)
    return found


def is_known_vak_answer(text: str) -> bool:
    """True if normalized text exactly matches a catalog option (after repair join)."""
    norm = normalize_vak_text(text)
    return bool(norm) and norm in _TEXT_TO_LETTER


def _tie_break_seed(answers: Iterable[object]) -> int:
    """Stable seed from a student's VAK answers so tie picks are reproducible."""
    import hashlib

    parts = tuple(
        normalize_vak_text(str(v))
        for v in answers
        if not _is_blank(v)
    )
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def learning_style_from_abc_counts(
    counts: Dict[PaperLetter, int],
    *,
    tie_seed: int | None = None,
) -> str:
    """
    Paper rule: highest A/B/C count wins.

    If two or three letters tie for the lead, pick one of the tied styles
    (deterministic pseudo-random choice from ``tie_seed``).
    """
    total = counts["A"] + counts["B"] + counts["C"]
    if total == 0:
        raise ValueError("No A/B/C learning-style answers could be counted.")

    max_count = max(counts.values())
    tied: list[PaperLetter] = [letter for letter in "ABC" if counts[letter] == max_count]

    if len(tied) == 1:
        return LETTER_TO_LEARNING_STYLE[tied[0]]

    seed = 0 if tie_seed is None else tie_seed
    winner = random.Random(seed).choice(tied)
    return LETTER_TO_LEARNING_STYLE[winner]


def score_learning_style_from_row(answers, *, tie_seed: int | None = None) -> str:
    """Count A/B/C across all learning-style columns, then apply the paper rule."""
    counts: Dict[PaperLetter, int] = {"A": 0, "B": 0, "C": 0}
    unknown: list[str] = []
    for val in answers:
        letters = letters_from_vak_cell(val)
        if letters:
            for letter in letters:
                counts[letter] += 1
        elif not _is_blank(val):
            unknown.append(str(val).strip()[:80])
    if unknown:
        preview = ", ".join(unknown[:3])
        extra = f" (+{len(unknown) - 3} more)" if len(unknown) > 3 else ""
        raise ValueError(
            f"Could not match {len(unknown)} answer(s) to the VAK option list, e.g. "
            f'"{preview}"{extra}. '
            "Use the exact option text from the research paper in Google Forms."
        )
    if tie_seed is None:
        tie_seed = _tie_break_seed(answers)
    return learning_style_from_abc_counts(counts, tie_seed=tie_seed)


def classify_vak_answer(raw: object) -> Optional[str]:
    """Return Visual / Auditory / Kinesthetic for one cell (letter → style)."""
    letter = classify_vak_letter(raw)
    if letter is None:
        return None
    return LETTER_TO_LEARNING_STYLE[letter]
