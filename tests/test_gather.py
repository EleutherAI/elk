
from hypothesis import given, strategies as st
import pytest
import os
import sys
from elk.evaluation.gather import get_metric_across_layers
from elk.files import elk_reporter_dir



# Define a Hypothesis strategy to generate invalid metric names
invalid_metric_names = st.text().filter(lambda x: x not in ["accuracy", "precision", "recall"])

# Define a Hypothesis strategy to generate invalid reporter names
invalid_reporter_names = st.text()

# Define a Hypothesis strategy to generate invalid reporter directories
invalid_reporter_dirs = st.text()

# Define a Hypothesis strategy to generate invalid save_csv flags
invalid_save_csv_flags = st.one_of(st.integers(), st.booleans(), st.text())

# Define the Hypothesis test
@given(
    metric_name=invalid_metric_names,
    reporter_name=invalid_reporter_names,
    reporter_dir=invalid_reporter_dirs,
    save_csv=invalid_save_csv_flags
)
def test_get_metric_across_layers_with_invalid_input(metric_name, reporter_name, reporter_dir, save_csv):
    # Assert that the function raises a ValueError with invalid inputs
    with pytest.raises(ValueError):
        get_metric_across_layers(metric_name, reporter_name, reporter_dir, save_csv)
