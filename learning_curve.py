import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generate_learning_curve(data):
    train_sizes = [i * 0.1 for i in range(1, 11)]
    decision_tree_curve = []
    decision_stump_curve = []
    dt_3_curve = []

    for trial in range(100):
        np.random.seed(trial)
        np.random.shuffle(data)
        fold_size = len(data) // 10

        for fold in range(10):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            test_data = data[test_start:test_end]
            train_data = np.concatenate([data[:test_start], data[test_end:]])

            X_train, y_train = train_data[:, 1:], train_data[:, 0]
            X_test, y_test = test_data[:, 1:], test_data[:, 0]

            for idx, size in enumerate(train_sizes):
                train_samples = int(size * len(X_train))
                X_subset_train, y_subset_train = X_train[:train_samples], y_train[:train_samples]

                clf = DecisionTreeClassifier()
                clf.fit(X_subset_train, y_subset_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                decision_tree_curve.append(accuracy)
                # ... (similar process for decision stump and 3-level decision tree)

    decision_tree_curve = np.array(decision_tree_curve).reshape((1000, 10))
    mean_tree = np.mean(decision_tree_curve, axis=0)
    std_tree = np.std(decision_tree_curve, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(train_sizes, mean_tree, yerr=std_tree, label='Decision Tree')
    # ... (similar process for other classifiers)
    
    plt.xlabel('Training Data Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load your SPECTF data here
    # data = np.loadtxt('path_to_your_data', delimiter=',')
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    generate_learning_curve(data)
    pass
