#all necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#imports for ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import auc

#imports for random forest
from sklearn.ensemble import RandomForestClassifier

#creates roc curve

def create_ROC_curve(x_dimension, y_dimension, ml_model_names, ml_models, X_tests, y_tests, title, pos_label):
    false_positive_rates = [] 
    true_positive_rates = []
    roc_auc_scores = []
    for model, X_test, y_test in zip(ml_models, X_tests, y_tests):
        #calculate RF model roc auc
        pos_index = list(model.classes_).index(pos_label)
        y_prob = model.predict_proba(X_test)[:, pos_index]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label = pos_label)
        roc_auc = auc(fpr, tpr)
        false_positive_rates.append(fpr)
        true_positive_rates.append(tpr)
        roc_auc_scores.append(roc_auc)

    #set figure size
    plt.figure(figsize=(x_dimension, y_dimension))

    #add models
    for model, fpr, tpr, rcs in zip(ml_model_names, false_positive_rates, true_positive_rates, roc_auc_scores):
        plt.plot(fpr, tpr, label = model + ' (area = %0.2f)' % rcs)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def visualize_confusion_matrix(ml_model, actual_values, test_values, labels, title):
    predictions = ml_model.predict(test_values)
    confusion_matrix = metrics.confusion_matrix(actual_values, predictions)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    ax = display.plot().ax_
    ax.set_title(title) 
    plt.show()

def visualize_feature_importance(x_dimension, y_dimension, ml_model, title, color, variable_list, number_of_features):
    importances = ml_model.feature_importances_
    indices = np.argsort(importances)[::-1][:number_of_features]
    plt.figure(figsize=(x_dimension, y_dimension))
    plt.title(title)
    #add 'other' importances to the front
    all_importances = importances[indices][::-1].tolist()
    all_associated_variables = [variable_list[i]for i in indices[::-1]]
    if len(variable_list) > number_of_features: #if not enough features and 'other'
        all_importances.insert(0, 1 - np.sum(all_importances))
        all_associated_variables.insert(0, 'Other')
    #plot feature importance
    plt.barh(range(len(all_importances)), all_importances, color = color, align='center')
    plt.yticks(range(len(all_associated_variables)), all_associated_variables)
    plt.xlabel('Relative Importance')
    plt.show()


def estimate_ml_metrics(ml_model, model_name, actual_values, test_values, runs):
    timings = []
    number_of_runs = runs
    for i in range(number_of_runs):
        #run model estimate time
        start = time.perf_counter()
        predictions = ml_model.predict(test_values)
        end = time.perf_counter()
        timings.append(end - start)

    #all estimates ± standard deviation if applicable
    latency = f"{np.mean(timings):.4f} ± {np.std(timings):.4f} seconds"
    accuracy = metrics.accuracy_score(actual_values, predictions)
    precision = metrics.precision_score(actual_values, predictions, average = 'weighted')
    recall = metrics.recall_score(actual_values, predictions, average = 'weighted')
    f1 = metrics.f1_score(actual_values, predictions, average = 'weighted')

    #organize as dictionary
    all_metrics = {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1' : f1, 'Latency': latency}
    return all_metrics

def create_comparison_frame(all_metrics_list):
    comparisons = np.zeros((len(all_metrics_list), 5)) # 0 - name,  1 - accuracy, 2 - precision, 3 - recall, 4 - f1, 5 - latency

    #add all metrics from all models
    for metrics in all_metrics_list:
        comparisons[0, :].append(metrics['Model'])
        comparisons[1, :].append(metrics['Accuracy'])
        comparisons[2, :].append(metrics['Precision'])
        comparisons[3, :].append(metrics['Recall'])
        comparisons[4, :].append(metrics['F1'])
        comparisons[5, :].append(metrics['Latency'])

    #store metrics in comparison frame comparisons[0, :]
    comparison_frame = pd.DataFrame({'Model': comparisons[0, :], 'accuracy': comparisons[1, :], 'Precision': comparisons[2, :], 'Recall': comparisons[3, :], 'F1' : comparisons[4, :], 'Latency': comparisons[5, :]}) 
    return comparison_frame


def visualize_precision_recall_curve(x_dimension, y_dimension, ml_model_names, ml_models, X_tests, y_tests, title, pos_label):
    precisions = []
    recalls = []
    for model, X_test, y_test in zip(ml_models, X_tests, y_tests):
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores, pos_label = pos_label)
        precisions.append(precision)
        recalls.append(recall)

    #set figure size
    plt.figure(figsize=(x_dimension, y_dimension))

    #plot
    for model, recall, precision in zip(ml_model_names, recalls, precisions):
        plt.plot(recall, precision, label = model)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def visualize_learning_curve_classification(x_dimension, y_dimension, ml_model_names, ml_models, dataset_features, dataset_targets, train_sizes, cross_validation_splits):
    #set figure size
    plt.figure(figsize=(x_dimension, y_dimension))

    for model, name in zip(ml_models, ml_model_names):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator = model,
            X = dataset_features,
            y = dataset_targets,
            train_sizes = train_sizes,
            cv = cross_validation_splits,
            scoring='accuracy'
        )
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score ' + name)
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score ' + name)

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()