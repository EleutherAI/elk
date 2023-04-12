import pytest
from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from elk.utils import convert_span


# Define a fixture with session scope that initializes the tokenizer
@pytest.fixture(scope="session")
def tokenizer():
    yield AutoTokenizer.from_pretrained("gpt2")


# Hypothesis will generate really bizarre Unicode strings for us
@st.composite
def string_and_span(draw) -> tuple[str, tuple[int, int]]:
    """Generate a non-empty string and a non-empty span within that string."""
    text = draw(st.text(min_size=1))
    start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end = draw(st.integers(min_value=start + 1, max_value=len(text)))
    return text, (start, end)


@given(string_and_span())
def test_convert_span(tokenizer, text_and_span: tuple[str, tuple[int, int]]):
    text, span = text_and_span

    tokenizer_output = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer_output["input_ids"]

    # Convert the span in string coordinates to a span in token coordinates
    token_start, token_end = convert_span(tokenizer_output["offset_mapping"], span)
    assert token_start < token_end

    string_start, string_end = span
    substring = text[string_start:string_end]
    token_subsequence = tokens[token_start:token_end]

    # Decode the subsequence of tokens back to a string
    decoded_string = tokenizer.decode(token_subsequence)

    # Assert that the original substring is fully contained within the decoded string
    assert substring in decoded_string
