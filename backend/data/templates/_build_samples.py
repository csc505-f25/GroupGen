"""One-off builder for template CSVs (30 students). Run from repo root:
python backend/data/templates/_build_samples.py
"""

import csv
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent

NAMES = [
    ("Alex Chen", "Male", "Asian American"),
    ("Jordan Lee", "Female", "Asian American"),
    ("Sam Rivera", "Male", "Hispanic"),
    ("Taylor Brooks", "Female", "White American"),
    ("Morgan Davis", "Male", "Black American"),
    ("Riley Kim", "Female", "Asian American"),
    ("Casey Nguyen", "Male", "Hispanic"),
    ("Jamie Wilson", "Female", "White American"),
    ("Drew Martinez", "Male", "Hispanic"),
    ("Quinn Patel", "Female", "Asian American"),
    ("Avery Thompson", "Non-binary", "White American"),
    ("Blake Johnson", "Male", "Black American"),
    ("Cameron Ortiz", "Female", "Hispanic"),
    ("Dakota Reed", "Male", "White American"),
    ("Emery Scott", "Female", "Asian American"),
    ("Finley Adams", "Male", "Hispanic"),
    ("Gray Morgan", "Female", "Black American"),
    ("Harper Bell", "Male", "Asian American"),
    ("Indigo Price", "Female", "White American"),
    ("Jules Cooper", "Non-binary", "Hispanic"),
    ("Kai Foster", "Male", "Asian American"),
    ("Logan Perry", "Female", "Black American"),
    ("Marley Hayes", "Male", "White American"),
    ("Noel Bryant", "Female", "Hispanic"),
    ("Oakley Griffin", "Male", "Asian American"),
    ("Parker Diaz", "Female", "White American"),
    ("Quincy Ross", "Male", "Black American"),
    ("Reese Powell", "Female", "Hispanic"),
    ("Sage Turner", "Non-binary", "Asian American"),
    ("Tatum Ward", "Female", "White American"),
]

V_A = "read the instructions first"
V_B = "follow a written recipe"
V_C = "give them a verbal explanation"
V_OPTIONS = [
    ("read the instructions first", "listen to an explanation from someone who has used it before", "go ahead and have a go, I can figure it out as I use it"),
    ("look at a map", "ask for spoken directions", "follow my nose and maybe use a compass"),
    ("watch how I do it", "listen to me explain", "you have a go"),
    ("going to museums and galleries", "listening to music and talking to my friends", "playing sport or doing DIY"),
    ("read lots of brochures", "listen to recommendations from friends", "imagine what it would be like to be there"),
]

WAND = "If you had a magic wand,  what would you change about these questions?"


def build_classroom() -> None:
    rows = []
    for i, (name, gender, div) in enumerate(NAMES):
        rows.append({
            "Name": name,
            "Gender": gender,
            "Motivation": 1 + (i % 4),
            "Self_Esteem": 1 + ((i * 2) % 4),
            "Work_Ethic": 1 + ((i + 1) % 4),
            "Learning_Style": ["Visual", "Auditory", "Kinesthetic"][i % 3],
            "Diversity": div,
        })
    pd.DataFrame(rows).to_csv(
        DIR / "classroom_template.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    print(f"classroom_template.csv: {len(rows)} students")


def build_google_form() -> None:
    cols = [
        "Timestamp",
        "Email Address",
        "What is your first and last name",
        "When I operate new equipment I generally:",
        "When I cook a new dish I like to",
        "If I am teaching someone something new I tend to",
        WAND,
        "I believe I will receive an excellent grade in this class.",
        "I'm certain I can understand the most difficult material presented in the readings for this course.",
        "I'm confident I can understand the basic concepts taught in this course.",
        WAND,
        "I sit near the front of the class if possible.",
        "I am alert in classes",
        "I ask the instructor questions when clarification is needed.",
        WAND,
        "I arrive at classes and other meetings on time.",
        "I devote sufficient study time to each of my courses.",
        "I schedule definite times and outline specific goals for my study time.",
        WAND,
        "To which gender identity do you most identify?",
        "To which ethnicity do you most identify?",
    ]
    rows = []
    for i, (name, gender, div) in enumerate(NAMES):
        v1, v2, v3 = V_OPTIONS[i % len(V_OPTIONS)]
        pick = i % 3
        ls = [v1, v2, v3][pick], [v1, v2, v3][(pick + 1) % 3], [v1, v2, v3][(pick + 2) % 3]
        email = name.lower().replace(" ", ".") + "@example.edu"
        rows.append({
            cols[0]: f"2026-05-{1 + (i % 28):02d} 10:{i % 60:02d}:00",
            cols[1]: email,
            cols[2]: name,
            cols[3]: ls[0],
            cols[4]: ls[1],
            cols[5]: ls[2],
            cols[6]: "",
            cols[7]: str(3 + (i % 5)),
            cols[8]: str(3 + ((i + 1) % 5)),
            cols[9]: str(3 + ((i + 2) % 5)),
            cols[10]: "",
            cols[11]: str(2 + (i % 3)),
            cols[12]: str(2 + ((i + 1) % 3)),
            cols[13]: str(2 + ((i + 2) % 3)),
            cols[14]: "",
            cols[15]: str(2 + (i % 3)),
            cols[16]: str(2 + ((i + 1) % 3)),
            cols[17]: str(2 + ((i + 2) % 3)),
            cols[18]: "",
            cols[19]: gender,
            cols[20]: div,
        })
    pd.DataFrame(rows, columns=cols).to_csv(DIR / "google_form_sample.csv", index=False)
    print(f"google_form_sample.csv: {len(rows)} students")


if __name__ == "__main__":
    build_classroom()
    build_google_form()
