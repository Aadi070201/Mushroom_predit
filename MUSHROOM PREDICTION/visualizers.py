import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_data_set(data_set):
    """
    Plot some interesting features of the dataset.

    :param data_set: dataset to plot (DataFrame).
    """

    # <ASSIGNMENT 3.2: Visualize some properties of the dataset>
    plt.figure(figsize=(10, 5))
    cap_color_counts = data_set.groupby(['cap-color', 'edible']).size().unstack()
    cap_color_counts.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
    plt.title("Distribution of Cap Color in Relation to Edibility")
    plt.xlabel("Cap Color")
    plt.ylabel("Count")
    plt.legend(["Non-Edible", "Edible"])
    plt.show()


def plot_confusion_matrix(score):
    ConfusionMatrixDisplay(score).plot()
