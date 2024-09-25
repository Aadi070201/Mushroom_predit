Feature: Predict mushroom edibility
    In order to guard against poisoning
    As a mushroom forager
    I want to estimate mushroom edibility

    Scenario: Pre-process a data set
        Given a raw data set
        When I pre-process the raw data set for analysis
        Then I obtain a pre-processed data set

    Scenario: Obtain predictions for new data
        Given a trained model
        When I use the trained model for prediction on new data
        Then I obtain estimated edibility
