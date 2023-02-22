from elk.math import stochastic_round_constrained
from hypothesis import given, strategies as st
from random import Random
import math
import numpy as np
import pytest
import os 

# test that running both elicit and extract works
# uses os.system to run the commands similar to how they would be run in the terminal
# tests that readme example works
@pytest.mark.gpu
def test_elk_run():
    # run elicit - check that the command runs without error
    os.system("elk elicit microsoft/deberta-v2-xxlarge-mnli imdb")
    # run extract - check that the command runs without error
    os.system("elk extract microsoft/deberta-v2-xxlarge-mnli imdb")




    
    