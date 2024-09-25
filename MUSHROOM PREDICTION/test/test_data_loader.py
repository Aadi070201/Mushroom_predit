import os

import numpy as np
import pandas as pd
import pytest

from data_loader import load_data, sanitize_data_entry

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory


class TestDataLoader:
    def test_load_data(self):
        raw_data = load_data(data_path=TEST_DIR)

        # <ASSIGNMENT 2.1: Find and fix the bug (in another file) to make both tests pass>
        assert type(raw_data) == pd.DataFrame
        assert raw_data.shape == (
            30,
            21,
        )  # Our data set for the test suite has thirty entries

    def test_sanitize_data_entry(self):
        # Define feature names and descriptions for testing
        features = {
            "first_feature": "first description",
            "second_feature": "second description",
            "third_feature": "third description",
        }
        num_features = ["first_feature"]

        # <ASSIGNMENT 2.2: Write the five additional tests>

        # Valid modes of behavior
        assert sanitize_data_entry("first_feature", "1", features, num_features) == (
            "first_feature",
            1.0,
        )
        # Additional tests for valid modes of behaviour
        
    # 1st test For a numeric feature, an empty input (""), should be interpreted as nan
        assert sanitize_data_entry("first_feature", "", features, num_features) == (   
            "first_feature",                                                              
            np.nan,     #np.nan represents missing data                    #if the numeric feature is provided an empty string it should return nan (not a number)
        )
    # 2nd test For a categorical feature, an empty input (""), should also be interpreted as nan
        assert sanitize_data_entry("second_feature", "", features, num_features) == (
            "second_feature",
            np.nan,                          # if the categorical feature is an empty string, it should return nan   
        )
    # 3rd Test For a categorical feature, the input should be interpreted as a string   
        assert sanitize_data_entry("second_feature", "some_string", features, num_features) == (
            "second_feature",
            "some_string",                               # for some string in categorical feature it should keep it as valid.       
        )     
        # Invalid modes of behavior                                
        
    # 1st Test for A numeric entry with a non-numeric input should raise a ValueError
        with pytest.raises(ValueError):
            sanitize_data_entry("first_feature", "not_a_number", features, num_features)  # if it is a non numeric value , then error
            
    # 2nd Test for An entry for which the queried feature is not within the list of valid features should raise a ValueError
        with pytest.raises(ValueError):
            sanitize_data_entry("non_existent_feature", "some_value", features, num_features) 
                     #if the feature does not exist in the feature dictonary, then error
    
