

# String Similarity & Automated License Plate Testing

## Project Overview

This project combines **string similarity matching** and **automated license plate validation** in Python.

**Functionalities:**

1. **String Similarity Matching (Q5)**

   * Accepts two strings (6–10 characters).
   * Calculates **percentage similarity** between the two strings.
   * Generates a **character-wise match report**.
   * Performs **alignment** to improve comparison accuracy.

2. **Automated Testing of License Plate Matching (Q6)**

   * Uses **Pytest** to test 1000 valid and invalid Indian license plate strings.
   * Generates a **summary report** of the results.
   * Relies on the Q5 similarity package for string comparison.

**Type of Problem:**

* Q5: **String similarity / keypoint matching in text**
* Q6: **Automated testing / unit testing using Pytest**

---

## File Structure

```
q5_and_q6_solution/
│
├── q5_similarity/              # Python package for string similarity
│   ├── __init__.py
│   ├── similarity.py           # Core similarity functions
│   └── __pycache__/            # Cache files
│
├── q6_solution/                # Pytest testing setup
│   ├── __init__.py
│   ├── conftest.py             # Fixtures and test setup
│   ├── q6_main_code.py         # Main testing logic
│   └── __pycache__/
│
├── q5_main_code.py             # Script to run string similarity manually
├── q6_report.txt               # Sample report of automated license plate tests
└── README.md                   # Project overview and instructions
```

---


## How to Run

### 1. String Similarity Matching

Run the main script for Q5:

```bash
python q5_main_code.py
```

* Enter two strings (6–10 characters) when prompted.
* Output includes:

  * **Percentage similarity**
  * **Matching characters**
  * **Non-matching characters**
  * **Aligned comparison** if needed.

**Example Output:**

```
Enter first string: lordsins
Enter second string: lordansa
Similarity: 75.00%
Match: lord
Mismatch: si -> a
Match: ns
Inserted: a
```

**Note:** `q5_similarity/similarity.py` is used as a package for reusable similarity functions.

---

### 2. Automated License Plate Testing

Run Pytest for Q6:

```bash
pytest q6_solution/q6_main_code.py
```

* Tests 1000 valid and invalid license plate strings.
* Uses `q5_similarity` package for string comparison.
* Generates a **summary report** in `q6_report.txt`.

**Example Pytest Output:**

```
=========================== test session starts ============================
collected 1000 items

==== Automated License Plate Similarity Tests ====

Total tests run: 1000
✅ Passed: 951
❌ Failed: 49
⚠️ Skipped: 0

============================ 951 passed in 5.24s ==========================
```

---

## Code Highlights

* **Reusing the similarity package:**

```python
from q5_similarity.similarity import calculate_similarity, match_report

similarity = calculate_similarity(str1, str2)
report = match_report(str1, str2)
```

* **Automated testing with Pytest:**

```python
def test_license_plate(plate, expected_result):
    assert validate_plate(plate) == expected_result
```

* Q6 test scripts use fixtures from `conftest.py` to load sample license plate strings.

---

## Notes & Tips

* Strings should be **6–10 characters** for accurate similarity.
* Pytest provides detailed reports for validation and debugging.
* The `q5_similarity` package allows easy reuse of similarity functions in other projects.
* Alignment in string comparison improves results for slightly misaligned or partially incorrect plates.

---

## Example Workflow

1. Run `q5_main_code.py` to test string similarity manually.
2. Run Pytest via `q6_main_code.py` to validate a large batch of license plates.
3. Review the summary report in `q6_report.txt` for results.
