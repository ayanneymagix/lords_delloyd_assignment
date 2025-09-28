import difflib

def string_similarity(s1, s2):
    """
    Compare two strings and return similarity percentage + match report.
    """
    matcher = difflib.SequenceMatcher(None, s1, s2)
    similarity = matcher.ratio() * 100

    report_lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            report_lines.append(f"Match: {s1[i1:i2]}")
        elif tag == "replace":
            report_lines.append(f"Mismatch: {s1[i1:i2]} -> {s2[j1:j2]}")
        elif tag == "delete":
            report_lines.append(f"Deleted: {s1[i1:i2]}")
        elif tag == "insert":
            report_lines.append(f"Inserted: {s2[j1:j2]}")

    return similarity, "\n".join(report_lines)

import re

def is_valid_plate(plate: str) -> bool:
    """Check if a license plate is valid Indian format (e.g., MH12AB1234)."""
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"
    return bool(re.match(pattern, plate))
