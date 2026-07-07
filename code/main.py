"""Deterministic baseline for HackerRank Orchestrate.

The challenge allows VLM/LLM systems, but this implementation intentionally
keeps the default path reproducible and dependency-free. It uses the claim
conversation, declared object type, user history, and local file checks to
produce the required output schema. The rule engine is designed to be easy to
swap with a vision model later: keep the same ``predict_row`` contract and add
image observations before final scoring.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


VALID_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}


ISSUE_PATTERNS = [
    ("glass_shatter", r"\b(shatter|shattered|smashed glass)\b"),
    ("stain", r"\b(stain|oily mark|oil stain|coffee|spill|spilled)\b"),
    ("water_damage", r"\b(water|wet|liquid|rain)\b"),
    ("torn_packaging", r"\b(torn|tear|phati|opened|open jaisa|seal side se open)\b"),
    ("crushed_packaging", r"\b(crushed|crush|dab gaya|badly crushed)\b"),
    ("missing_part", r"\b(missing|not inside|not in the box|keycaps came off|came off)\b"),
    ("crack", r"\b(crack|cracked|cracking|cracke|danado|dano)\b"),
    ("dent", r"\b(dent|dented|hail dents|pressed|deformation)\b"),
    ("scratch", r"\b(scratch|scrape|mark|scuff)\b"),
    ("broken_part", r"\b(broken|broke|breakage|toot|damaged|affected|does not sit|wobbles)\b"),
]


PART_PATTERNS = {
    "car": [
        ("front_bumper", r"\b(front bumper|front side|front-end|front end|parachoques delantero)\b"),
        ("rear_bumper", r"\b(rear bumper|back bumper|back of the car|from behind|parachoques trasero|parachoques de atras)\b"),
        ("side_mirror", r"\b(side mirror|mirror|left side mirror)\b"),
        ("windshield", r"\b(windshield|front glass)\b"),
        ("headlight", r"\b(headlight|front light)\b"),
        ("taillight", r"\b(taillight|tail light|back light)\b"),
        ("quarter_panel", r"\b(quarter panel)\b"),
        ("fender", r"\b(fender)\b"),
        ("door", r"\b(door|door panel)\b"),
        ("hood", r"\b(hood|top panel)\b"),
        ("body", r"\b(body|body panel|car body)\b"),
    ],
    "laptop": [
        ("hinge", r"\b(hinge)\b"),
        ("keyboard", r"\b(keyboard|keys|keycaps|teclas|keys missing)\b"),
        ("trackpad", r"\b(trackpad|touchpad)\b"),
        ("screen", r"\b(screen|display|pantalla)\b"),
        ("lid", r"\b(lid|outer lid)\b"),
        ("corner", r"\b(corner)\b"),
        ("port", r"\b(port)\b"),
        ("base", r"\b(base|bottom)\b"),
        ("body", r"\b(body|outer body|side edge)\b"),
    ],
    "package": [
        ("contents", r"\b(contents|missing contents|missing item|not inside|inside)\b"),
        ("seal", r"\b(seal|tape|flap)\b"),
        ("label", r"\b(label|shipping label)\b"),
        ("package_corner", r"\b(corner|package corner|box corner)\b"),
        ("package_side", r"\b(side|surface|outside|exterior)\b"),
        ("item", r"\b(item inside|product|broken item)\b"),
        ("box", r"\b(box|parcel|package|cardboard)\b"),
    ],
}


SEVERITY_HIGH = re.compile(r"\b(severe|badly|shattered|smashed|missing contents|not inside)\b")
SEVERITY_MEDIUM = re.compile(r"\b(crack|broken|broke|torn|crushed|water|stain|liquid|dent)\b")


@dataclass
class ImageEvidence:
    paths: List[str]
    existing_paths: List[str]
    image_ids: List[str]

    @property
    def valid(self) -> bool:
        return bool(self.paths) and len(self.paths) == len(self.existing_paths)


def normalize_bool(value: bool) -> str:
    return "true" if value else "false"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def load_user_history(path: Path) -> Dict[str, dict]:
    return {row["user_id"]: row for row in read_csv(path)}


def image_evidence(dataset_dir: Path, image_paths: str) -> ImageEvidence:
    paths = [part.strip() for part in (image_paths or "").split(";") if part.strip()]
    existing = [p for p in paths if (dataset_dir / p).exists()]
    image_ids = [Path(p).stem for p in existing]
    return ImageEvidence(paths=paths, existing_paths=existing, image_ids=image_ids)


def detect_issue(text: str, claim_object: str) -> str:
    lowered = text.lower()
    if claim_object == "laptop":
        if re.search(r"\b(keyboard|keys|keycaps|teclas)\b", lowered) and re.search(r"\b(stain|coffee|spill|spilled|liquid|water|missing|came off)\b", lowered):
            return "missing_part" if re.search(r"\b(missing|came off|faltan)\b", lowered) else "stain"
        if "screen" in lowered and re.search(r"\b(crack|cracked|shatter|shattered|pantalla)\b", lowered):
            return "crack"
        if "corner" in lowered and "dent" in lowered:
            return "dent"
    if claim_object == "package":
        if re.search(r"\b(oil stain|oily mark|stain)\b", lowered):
            return "stain"
        if re.search(r"\b(water|wet)\b", lowered):
            return "water_damage"
        if re.search(r"\b(seal|tape|flap)\b", lowered) and re.search(r"\b(torn|opened|open jaisa|phati|broken)\b", lowered):
            return "torn_packaging"
        if re.search(r"\b(corner|box|package|parcel|cardboard)\b", lowered) and re.search(r"\b(crushed|crush|dab gaya)\b", lowered):
            return "crushed_packaging"
    for issue, pattern in ISSUE_PATTERNS:
        if re.search(pattern, lowered):
            if issue in {"torn_packaging", "crushed_packaging"} and claim_object != "package":
                continue
            return issue
    return "unknown"


def detect_part(text: str, claim_object: str) -> str:
    lowered = text.lower()
    for part, pattern in PART_PATTERNS.get(claim_object, []):
        if re.search(pattern, lowered):
            return part
    return "unknown"


def dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def risk_flags(text: str, history: dict | None, evidence: ImageEvidence) -> List[str]:
    lowered = text.lower()
    flags: List[str] = []
    if not evidence.valid:
        flags.append("damage_not_visible")
        flags.append("manual_review_required")
    if re.search(r"\b(ignore|skip manual review|approve the claim immediately|follow .*instruction|previous instructions)\b", lowered):
        flags.append("text_instruction_present")
        flags.append("manual_review_required")
    if history:
        for flag in (history.get("history_flags") or "").split(";"):
            flag = flag.strip()
            if flag and flag != "none":
                flags.append(flag)
    if "user_history_risk" in flags:
        flags.append("manual_review_required")
    return dedupe(flags) or ["none"]


def evidence_reason(evidence_met: bool, part: str, issue: str, evidence: ImageEvidence) -> str:
    if not evidence.valid:
        return "One or more referenced image files are missing, so the image set is not usable for automated review."
    if part == "unknown":
        return "The submitted image set exists, but the transcript does not identify a specific reviewable object part."
    if issue == "unknown":
        return f"The submitted image set exists and shows evidence for review, but the claimed issue type is ambiguous for the {part}."
    if evidence_met:
        return f"The image set is available and provides enough submitted evidence to review the claimed {issue} on the {part}."
    return f"The submitted evidence is not sufficient to verify the claimed {issue} on the {part}."


def estimate_severity(issue: str, text: str, status: str) -> str:
    if status == "not_enough_information":
        return "unknown"
    lowered = text.lower()
    if issue == "none":
        return "none"
    if issue in {"glass_shatter", "missing_part"} or SEVERITY_HIGH.search(lowered):
        return "high"
    if issue in {"crack", "broken_part", "torn_packaging", "crushed_packaging", "water_damage", "stain", "dent"} or SEVERITY_MEDIUM.search(lowered):
        return "medium"
    if issue == "scratch":
        return "low"
    return "unknown"


def status_for(evidence_met: bool, issue: str, part: str, flags: Sequence[str], strategy: str) -> str:
    if not evidence_met or issue == "unknown" or part == "unknown":
        return "not_enough_information"
    if strategy == "text_only":
        return "supported"
    if "damage_not_visible" in flags:
        return "not_enough_information"
    return "supported"


def predict_row(row: dict, user_history: Dict[str, dict], dataset_dir: Path, strategy: str = "final") -> dict:
    text = clean_text(row.get("user_claim", ""))
    claim_object = (row.get("claim_object") or "").strip()
    evidence = image_evidence(dataset_dir, row.get("image_paths", ""))
    issue = detect_issue(text, claim_object)
    part = detect_part(text, claim_object)

    if part not in VALID_PARTS.get(claim_object, {"unknown"}):
        part = "unknown"

    history = user_history.get(row.get("user_id", ""))
    flags = ["none"] if strategy == "text_only" else risk_flags(text, history, evidence)
    evidence_met = evidence.valid and part != "unknown"
    status = status_for(evidence_met, issue, part, flags, strategy)
    severity = estimate_severity(issue, text, status)

    if status == "supported":
        justification = f"The submitted images are available for review and the transcript claims {issue} on the {part}."
    elif status == "not_enough_information":
        justification = f"The claim cannot be fully verified automatically because the evidence is incomplete or the claimed {part} / {issue} is ambiguous."
    else:
        justification = f"The submitted evidence does not support the claimed {issue} on the {part}."

    if flags != ["none"] and "user_history_risk" in flags:
        justification += " User history adds review risk but does not override the evidence decision."
    if "text_instruction_present" in flags:
        justification += " Instruction-like text in the conversation is ignored for the evidence decision."

    supporting_ids = ";".join(evidence.image_ids) if evidence_met and evidence.image_ids else "none"

    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": claim_object,
        "evidence_standard_met": normalize_bool(evidence_met),
        "evidence_standard_met_reason": evidence_reason(evidence_met, part, issue, evidence),
        "risk_flags": ";".join(flags),
        "issue_type": issue,
        "object_part": part,
        "claim_status": status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_ids,
        "valid_image": normalize_bool(evidence.valid),
        "severity": severity,
    }


def run(input_csv: Path, output_csv: Path, dataset_dir: Path, strategy: str = "final") -> List[dict]:
    user_history = load_user_history(dataset_dir / "user_history.csv")
    rows = [predict_row(row, user_history, dataset_dir, strategy=strategy) for row in read_csv(input_csv)]
    write_csv(output_csv, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HackerRank Orchestrate predictions.")
    parser.add_argument("--input", default="dataset/claims.csv", help="Input claims CSV.")
    parser.add_argument("--output", default="output.csv", help="Destination predictions CSV.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory.")
    parser.add_argument("--strategy", default="final", choices=["text_only", "final"], help="Prediction strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.input), Path(args.output), Path(args.dataset_dir), strategy=args.strategy)


if __name__ == "__main__":
    main()
