import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Data loader marks the root directory
DATA_PATH = os.path.join(ROOT_DIR, "datasets")

# Features used for prediction
FEATURES = {
    "cap-diameter": "number in cm",
    "cap-shape": "bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o",
    "cap-surface": "fibrous=i, grooves=g, scaly=y, smooth=s, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e",
    "cap-color": "brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k",
    "does-bruise-or-bleed": "bruises-or-bleeding=t, no=f",
    "gill-attachment": "adnate=a, adnexed=x, decurrent=d, free=e, sinuate=s, pores=p, none=f, unknown=?",
    "gill-spacing": "close=c, distant=d, none=f",
    "gill-color": "brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k, none=f",
    "stem-height": "number in cm",
    "stem-width": "number in mm",
    "stem-root": "bulbous=b, swollen=s, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r",
    "stem-surface": "fibrous=i, grooves=g, scaly=y, smooth=s, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e, none=f",
    "stem-color": "brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k, none=f",
    "veil-type": "partial=p, universal=u",
    "veil-color": "brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k, none=f",
    "has-ring": "ring=t, none=f",
    "ring-type": "cobwebby=c, evanescent=e, flaring=r, grooved=g, large=l, pendant=p, sheathing=s, zone=z, scaly=y, movable=m, none=f, unknown=?",
    "spore-print-color": "brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k",
    "habitat": "grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d",
    "season": "spring=s, summer=u, autumn=a, winter=w",
}

# Numerical features
NUMERICAL = ["cap-diameter", "stem-height", "stem-width"]

# Quantity that is predicted
TARGET = "edible"

#To detect the delimiter 
def detect_delimiter(file_path):
    """
    Detect the delimiter of the given file by reading the first line.
    
    :param file_path: Path to the dataset file;
    :return: The detected delimiter.
    """
    with open(file_path, 'r') as file:
        sample = file.readline()  # Read the first line
        delimiters = [',', ';', '\t', '|']  
        
        # Check which delimiter appears the most in the first line
        delimiter_counts = {delimiter: sample.count(delimiter) for delimiter in delimiters}
        
        # Return the delimiter that appears most frequently
        return max(delimiter_counts, key=delimiter_counts.get)


def query_input_data(features=FEATURES, num_features=NUMERICAL):
    """
    Query user for input data.

    :param features: dictionary of valid features with additional info;
    :param num_features: list of numerical features;
    :return: a DataFrame with sanitized inputs.
    """

    # Initialize an empty dictionary of data entries
    input_data = {}
    for feature in features:
        input_data[feature] = []

    # Append sanitized values to dictionary
    for feature in features:
        raw_value = input(f"{feature} ({features[feature]}): ")  # Request input
        (_, sanitized_value) = sanitize_data_entry(
            feature, raw_value, features, num_features
        )
        input_data[feature].append(sanitized_value)  # Update dictionary with entry

    return pd.DataFrame(input_data)  # Convert entries to DataFrame
                        
def sanitize_data_entry(feature, raw_value, features=FEATURES, num_features=NUMERICAL):
    """
    Sanitize a user-provided data entry.

    :param feature: feature name (string)
    :param raw_value: user-provided input value (string)
    :param features: dictionary of valid features;
    :param num_features: list of numerical features;
    :return: a tuple of the original feature name and sanitized value.
    
    the fuction will handle 2 types of data :- a) Numerical features  b) Categorial features
    Numerical :- the value will be converted to float and empty will be given as nan
    Categorial :- The value is returned as it is and empty will be given as nan
    
    """
    # remove the whitespace and covert to lower case
    sanitized_value = raw_value.strip().lower() 

    # Handle numerical features
    if feature in num_features:
        if sanitized_value == "":  # Empty input should return as nan
            return feature, np.nan
        try:
            sanitized_value = float(sanitized_value)  # Convert to float
        except ValueError:                           # catch statement to raise an error for invalid numerical value
            raise ValueError(f"Invalid numerical value for {feature}: {raw_value}")

    # Handle categorical features
    elif feature in features:
        if sanitized_value == "":  # Empty input should be interpreted as nan
            return feature, np.nan
        # For categorical features, just return the sanitized value
        return feature, sanitized_value
    
    else:                                 # Catch statement to raise an error if the feature is not recognized
        raise ValueError(f"Feature '{feature}' is not recognized.")
     
    return feature, sanitized_value             # returning the value of feature name and the value.
                       
def load_data(data_path=DATA_PATH):
    """
    Load dataset in a DataFrame with automatic delimiter detection.
    
    :param data_path: root folder of data file;
    :return: DataFrame with raw dataset.
    """
    csv_path = os.path.join(data_path, "dataset.csv")
    
    # Automatically detect the delimiter
    delimiter = detect_delimiter(csv_path)
    
    # Load the data with the detected delimiter
    return pd.read_csv(csv_path, delimiter=delimiter)


def split_train_test(dataset, test_size=0.2, random_state=42):
    """
    Split input DataFrame into separate train- and test sets.

    :param dataset: input dataset (DataFrame)
    :return: train and test set as DataFrames
    """
    return train_test_split(dataset, test_size=test_size, random_state=random_state)


# Load the data
data = load_data()

# Split the dataset into train and test sets
train_set, test_set = split_train_test(data)

# <ASSIGNMENT 3.1: Split the dataset in a train and test set>
print(len(test_set))


