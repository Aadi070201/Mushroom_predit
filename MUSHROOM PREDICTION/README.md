[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cAiLf4p-)
# Mushroom Project (Week 2 and 3)

In this project you will prototype an application that predicts the edibility of mushrooms. A client wishes to develop a mobile application for mushroom foragers. With this application, a user can enter some parameters of a mushroom they found and get an indication of edibility. Your assignment is to design a proof-of-concept for this application.


## Completing Assignments

You will need to complete the code, tests and notebooks. This project is split in two parts, where each part corresponds with one week of the course (Week 2 and 3). The `<ASSIGNMENT>` tags throughout the code indicate where you should provide your answers.

After completing a (set of) assignments, commit and push your changes to GitHub. Your project will then be automatically evaluated against the test suite. As you continue to complete assignments, more tests should pass. Your code and answers should be correct, concise and readable. We also look for clean and concise comments and commit history.

Your project will be graded _as-is_ on GitHub at the time of the deadline. Make sure to commit and push frequently.


# Getting Started

Before you start the assignment, you will need to have Git, GitHub and VSCode properly installed. If not, then first please follow the course-specific getting started guide.

## Clone the Repository

Accept the assignment invitation in GitHub Classroom. This creates a new personal remote with the skeleton code for the Mushroom project on GitHub. Clone the repository to your machine, and in VSCode select `File > Open Folder...` to open the project directory in your workspace.

## Activate and Initialize a Virtual Environment

A Virtual Environment creates a custom package-versioning environment for your specific Python project. This way, package versions of other projects do not interfere with your current project, and your runtime environment (including bugs) can be replicated across machines. In this project we use Anaconda to manage virtual environments and packages. The required package versions for the Mushroom project are defined in the `environment.yml` file.

Open the Mushroom project in VSCode and open a new terminal. Use the command `conda env create -f environment.yml` to create a virtual environment and install required packages (this might take a while).

This command also creates a new `mushroom312` virtual environment. Activate the newly created environment by `conda activate mushroom312`. The Virtual Environment can be deactivated by typing `conda deactivate`. The Virtual Environment only needs to be created once, but you may be required to reactivate it when reopening your project. Look for the prompt prefix that indicates the active Virtual Environment, and reactivate if needed.

## Linter
A linter automatically checks your source code against a style guide with generally accepted code formatting conventions. For Python, this guide is known as PEP 8. After activating your conda environment, you can run `ruff check` to display any style violations. You can use this tool as a starting point to enhance the readability of your code.


# Assignments for Week 2

Usually, a software project starts with a conversation with the client. The client might be an external company, your boss, a family member, or anyone that requires your services. This week we will focus on writing tests for a user interface and work towards a rudimentary implementation that can be communicated with the client. The `interface_and_prediction.ipynb` notebook offers a useful environment for prototyping and experimentation (make use of it).


## Test-Driven Development

The `pytest` library automatically executes pre-defined tests that verify the functionality of the software. You can run `pytest` from the console or via VSCode. In VSCode, open the testing tab (left menu). In this tab, click `Configure Python Tests` and select `pytest` as your testing library. Then select the `test` directory as the directory that houses your tests. You can now run the tests from this tab. You can alternatively run the tests from the file explorer, or via the command line with `python -m pytest`. When you run the tests you will see all of them fail (for now).

> ### Assignment 2.1
> There is a bug in **`data_loader.py`**. Run the `test_load_data` test and observe that it fails with an `AssertionError` in `test_data_loader.py`. Trace the error and fix the bug in the data loader.

We now shift our attention to the user-input validation. Ater an initial conversation with the client, it becomes priority to deliver something tangible. A mockup for the user-interface is therefore usually the first order of business. In the `interface_and_prediction.ipynb` notebook, the user can enter new data through an interface, and a prediction for edibility should be returned.

Currently there is no input sanitation. We are free to enter any value we like, which will lead to compatibility and possibly even security issues (e.g. code injection). Therefore, we wish to write a  `sanitize_data_entry` method (in `data_loader.py`) that accepts the raw user input and processes it. Before writing any new functionality however, the Test-Driven Development (TDD) approach prescribes that we should first write tests that outline the desired functionality.

One test is already implemented in `test_data_loader.py`. Valid modes of behavior specify allowed (desired) input-output combinations, while invalid modes of behavior specify which inputs are disallowed and should raise an error.

> ### Assignment 2.2
> In **`test_data_loader.py`**, write five _additional_ tests that verify the following functionality:
>
> Valid modes of behavior:
> - For a numeric feature, an empty input (`""`), should be interpreted as `nan`
> - For a categorical feature, an empty input (`""`), should also be interpreted as `nan`
> - For a categorical feature, the input should be interpreted as a `string`
>
> Invalid modes of behavior:
> - A numeric entry with a non-numeric input should raise a `ValueError`
> - An entry for which the queried feature is not within the list of valid features should raise a `ValueError`

We are currently in the "red" phase of the TDD cycle. After completing the tests, run the `test_sanitize_data_entry` test to verify that they fail. We now move to the "green" phase and start writing the actual functionality.

> ### Assignment 2.3
> In **`data_loader.py`**, complete the `sanitize_data_entry` method so that the sanitation tests pass. Refactor your code for readability and make sure to write appropriate comments.

Good times to commit are usually after completing a step in the TDD cycle. Commit your changes and push them to GitHub. Remember to include a short and to-the-point description of your changes. From now on we will stop reminding you to commit and push, and trust that you choose your commit strategy wisely.


## Behavior-Driven Development

Technical needs of the client can be captured in user stories that describe, from a user-perspective, the main modes of interaction with the application.

In the test directory, the `predict_edibility.feature` file already describes a feature through two scenarios, from the perspective of a user. This feature is not just any text file; it is an actual program that can be executed using `pytest`. The individual steps for each scenario are detailed in `test_predict_edibility.py`. In the testing tab, run the tests for `test_predict_edibility.py`. You should see two failing tests, indicating that the desired functionality is not implemented yet. At the end of this project, both scenarios should (hopefully) pass.

These scenarios imply datasets, trained models, estimates and other concepts that we have not yet defined, and that's ok for now. At the start of any project it is important to manage the expectations of the client. Therefore, scenarios should be written in the "language" of the client, which allows them to offer feedback on high-level funtionality early-on in the project. This Behavior-Driven Development (BDD) approach enables us to swiftly clear up any misunderstandings, and (hopefully) prevent double work.

The [Gherkin](https://en.wikipedia.org/wiki/Cucumber_(software)#Gherkin_language) syntax provides a convenient syntax for BDD. The following aspects are important to keep in mind when writing a feature:
- A feature describes a _single goal_ from the user perspective;
- A feature description follows the "In order to", "As a", "I want to" structure;
- Each scenario describes a _single path_ of user interaction;
- Each scenario follows the "Given", "When", "Then" structure;
- Multiple consecutive steps can be chained by an "And" statement;
- Steps describe behavior, not technical implementation.

The current user interface is very rudimentary and requires the subsequent entry of each individual feature. This user-interaction is quite cumbersome and not suitable for use by a mushroom forager in the field.

> ### Assignment 2.4
> In **`user_interface.feature`**, draft a feature description and _two_ scenarios in Gherkin syntax that propose an improvement of the user interface (see the `predict_edibility.feature` file for inspiration). You don't need to implement the code that executes the steps.

The rudimentary user-interface and your feature proposal for improvement are now ready for client feedback, so we need to make things presentable.

> ### Assignment 2.5
> In **`interface_and_prediction.ipynb`**, prepare the notebook for presentation to the client by adding appropriate markdown explanations and comments to the cells of the "Load Data" and "User Interface" sections. These should present a coherent story that informs the client about your design choices. You don't need to implement your proposed interface improvement or any other functionality for now.

Don't forget to commit and push your work (ok, that was really the last reminder).


# Assignments for Week 3

Last week we developed a rudimentary user interface. This week, we prepare and analyse a data set. We assume the client provides us with a data set on which we will train our models. This dataset is stored in the `datasets` folder. The `training_and_evaluation.ipynb` notebook offers an environment for prototyping and experimentation.


## Train and Test Sets

This week we will train prediction models on these data, using `scikit-learn`. Before training however, we should ensure that we are prepared to evaluate the model in an unbiased manner. In a production environment, the data that will be fed to your application (e.g. through the user interface) will be unknown to you beforehand. Therefore, in order to obtain an honest evaluation of practical model performance, we should make sure that we are prepared to evaluate our models against unknown data as well.

We divide the data provided into a `train_set` and a `test_set`. As the names suggest, the `train_set` is used for training our models, while the `test_set` is used for testing model performance. The `test_set` is set aside. It is important to resist the temptation to inspect the `test_set`, because this will only bias your conclusions. In order to remain truly unbiased against unseen data, leave the data that is meant to be unseen actually unseen (for now).

> ### Assignment 3.1
> In **`data_loader.py`**, write a `split_train_test` method that randomly divides the argument data set in a `train_set` and a `test_set` according to a 80/20 train/test split, with a fixed random state (for reproducibility). Hint: look at the sklearn documentation.

## Data Visualization

With a train and test set nicely separated, we can start visualizing the training set to get a feel for the data.

> ### Assignment 3.2
> In **`visualizers.py`**, visualize some informative property of the training set in relation to edibility. Hint: `DataFrame.plot` offers some nice functionality.


## Pre-Processing Pipelines

Now that you have a feel for the data, we can start pre-processing it for analysis. A preprocessing `Pipeline` automates this procedure. A `Pipeline` applies sequential operations to a data set, and can for example be used to impute missing values, combine features, scale feature values, and encode categorical features to a one-hot representation.

Some transformations need to be trained, such as imputing values and scaling. These transformations can be trained on the `train_set`, and later applied to the `test_set` and new datapoints in production. This pre-processing procedure ensures that transformations are consistently applied, remain decoupled from the algorithm implementation, and can be versioned.

Each step in a `Pipeline` is defined as a Python class with an `__init__`, `fit` and `transform` method. The pipeline then applies the `transform` and `fit` methods of each consecutive pre-processing step. Numeric and categorical features require separate pre-processing procedures. Therefore, we define separate procedures for numeric and categorical features, which are in the end combined.


> ### Assignment 3.3
> In **`pipelines.py`**, write a `build_pipeline` method that does the following:
>
> For numeric features:
> 1. Impute missing values by the median of the available feature values;
> 2. Scale the data to zero mean and unit variance.
>
> For categorical features:
> 1. Impute missing values by substituting `"missing"`;
> 2. Apply a one-hot encoder that ignores unknown categories.

The pipeline can be stored to a file, so that you can use it to transform incoming data once the system is in production.

> ### Assignment 3.4
> In **`training_and_evaluation.ipynb`**, use the imported functions to train the pipeline. Save the pipeline to a file and apply the pipeline to the train and test set.


## Model Search

With our data prepared and preprocessed, we can start training models.

> ### Assignment 3.5
> In **`models.py`**, implement a `build_model` method that defines a suitable model for the task.

Cross-validation allows us to freely search for models without becoming biased against unseen data (from the test-set). This way the test-set remains untouched, and only once we have discovered a satisfactory model we will evaluate performance against the test-set.

> ### Assignment 3.6
> In **`models.py`**, implement a `cross_validate_model` method that performs a 5-fold cross-validation on your model using a suitable metric for binary classification.

With the model-related methods defined, we can start the model discovery process. Similar to a pipeline, a trained model can also be stored to a file so that it can be easily deployed to production.

> ### Assignment 3.7
> In **`training_and_evaluation.ipynb`**, run the cross-validation and report the cross-validation scores.

We can use cross-validation in combination with a grid search over hyperparameters to fine-tune a model. In order to automate the tuning of hyperparameters, a `GridSearchCV` object can be used to train our model on possible combinations of hyperparameters, as defined by a `param_grid` dictionary. The `grid_search.fit` method then trains and evaluates the model on all defined hyperparameter combinations. We can then select the model with the the best cross-validation score.

> ### Assignment 3.8
> In **`models.py`**, implement a `finetune_model` method that performs a grid search over some hyperparameter combinations. Update the `build_model` method to define the model with optimal parameters.


## Model Evaluation

The ultimate test of strength is when we evaluate performance of the optimized model against the unseen test set. This provides us with a reference performance level that we can report to our client, and which can be used to compare other candidate models against.

> ### Assignment 3.9
> In **`training_and_evaluation.ipynb`**, use the imported functions to train the optimized model. Save the model to a file and evaluate the model performance on the test set by plotting a confusion matrix.

A confusion matrix offers insight in the types of prediction errors. Depending on the application, certain error types may be preferred over others.

> ### Assignment 3.10
> In **`training_and_evaluation.ipynb`**, interpret the confusion matrix. For this application, which quadrant is most important and why? Would you trust this app, and why (not)? Make a practical recommendation to the client based on your interpretation.

With an optimized model in hand, we can apply it to a new datapoint that is entered by the user.

> ### Assignment 3.11
> In **`interface_and_prediction.ipynb`**, use the imported functions to process a user-provided datapoint and make a prediction for edibility.

Your application is now ready for another round of client feedback, so again we need to make things presentable.

> ### Assignment 3.12
> In **`training_and_evaluation.ipynb`**, prepare the notebook for presentation to the client by adding appropriate markdown text and comments. You don't need to implement any additional functionality.
