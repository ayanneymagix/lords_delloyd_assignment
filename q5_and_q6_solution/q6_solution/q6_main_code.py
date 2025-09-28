import pytest
import random
import string
from q5_similarity.similarity import string_similarity


def generate_valid_plate():
    """Generate a random valid Indian license plate (e.g., MH12AB1234)."""
    state = ''.join(random.choices(string.ascii_uppercase, k=2))
    district = ''.join(random.choices(string.digits, k=2))
    series = ''.join(random.choices(string.ascii_uppercase, k=2))
    number = ''.join(random.choices(string.digits, k=4))
    return state + district + series + number


def generate_invalid_plate():
    """Generate a random invalid plate (wrong format)."""
    choices = [
        ''.join(random.choices(string.ascii_lowercase, k=10)),  # lowercase
        ''.join(random.choices(string.digits, k=8)),            # only numbers
        ''.join(random.choices(string.ascii_uppercase, k=5)),   # too short
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=12)), # too long
        "1234MHAB"  # scrambled format
    ]
    return random.choice(choices)


# Create 500 valid and 500 invalid test pairs
valid_pairs = [(generate_valid_plate(), generate_valid_plate()) for _ in range(500)]
invalid_pairs = [(generate_invalid_plate(), generate_invalid_plate()) for _ in range(500)]


@pytest.mark.parametrize("s1, s2", valid_pairs)
def test_valid_plates(s1, s2):
    similarity, report = string_similarity(s1, s2)
    # Valid plates should almost always have some similarity
    assert similarity > 0, f"Valid plates wrongly returned 0%\n{s1} vs {s2}\nReport:\n{report}"


@pytest.mark.parametrize("s1, s2", invalid_pairs)
def test_invalid_plates(s1, s2):
    similarity, report = string_similarity(s1, s2)
    # Invalid plates can have random similarity, but it must stay under 25%
    assert 0 <= similarity <= 25, f"Invalid plate similarity too high ({similarity}%)\n{s1} vs {s2}\nReport:\n{report}"
