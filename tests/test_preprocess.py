
from meqsum.preprocess import normalize_text, prepare_example


def test_normalize_text():
    text = "What is\n  the dosage of  5 mg?"
    expected = "What is the dosage of 5 mg?"
    assert normalize_text(text) == expected


def test_prepare_example():
    example = {
        "File": "001.txt",
        "CHQ": "Can I take\n aspirin?",
        "Summary": "Is aspirin safe?",
    }

    result = prepare_example(example)

    assert result["id"] == "001.txt"
    assert result["source"] == "Can I take aspirin?"
    assert result["target"] == "Is aspirin safe?"
