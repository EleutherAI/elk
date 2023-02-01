import pytest
import pandas as pd

class TestExtractionOutput:
    def testExtractionOutput(self):
        # Read the contents of the csv files into pandas dataframes
        # TODO: Right now, the file paths are hardcoded. Improve on that.
        correct_output_file = pd.read_csv("./test_data/deberta-v2-xxlarge-mnli_confusion_0.csv")
        current_output_file = pd.read_csv("../evaluation_results/deberta-v2-xxlarge-mnli_confusion_0.csv")
        
        # Compare the contents of the dataframes
        assert correct_output_file.equals(current_output_file)

