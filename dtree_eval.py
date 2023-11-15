import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def evaluatePerformance(data):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    num_trials = 1000
    num_folds = 10
    stats = np.zeros((3, 2))
    decision_tree_acc = []
    decision_stump_acc = []
    dt_3_acc = []

    for _ in range(num_trials):
        np.random.shuffle(data)
        fold_size = len(data) // num_folds

        for fold in range(num_folds):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            test_data = data[test_start:test_end]
            train_data = np.concatenate([data[:test_start], data[test_end:]])

            X_train, y_train = train_data[:, 1:], train_data[:, 0]
            X_test, y_test = test_data[:, 1:], test_data[:, 0]

            # Train Decision Tree
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            decision_tree_acc.append(accuracy)

            # Train Decision Stump
            clf_stump = DecisionTreeClassifier(max_depth=1)
            clf_stump.fit(X_train, y_train)
            y_pred_stump = clf_stump.predict(X_test)
            accuracy_stump = accuracy_score(y_test, y_pred_stump)
            decision_stump_acc.append(accuracy_stump)

            # Train 3-level Decision Tree
            clf_3 = DecisionTreeClassifier(max_depth=3)
            clf_3.fit(X_train, y_train)
            y_pred_3 = clf_3.predict(X_test)
            accuracy_3 = accuracy_score(y_test, y_pred_3)
            dt_3_acc.append(accuracy_3)

    stats[0, 0] = np.mean(decision_tree_acc)
    stats[0, 1] = np.std(decision_tree_acc)
    stats[1, 0] = np.mean(decision_stump_acc)
    stats[1, 1] = np.std(decision_stump_acc)
    stats[2, 0] = np.mean(dt_3_acc)
    stats[2, 1] = np.std(dt_3_acc)

    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    # Load data from SPECTF.dat file
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')

    stats = evaluatePerformance(data)
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
