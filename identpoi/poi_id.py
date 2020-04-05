
# !/usr/bin/env python2

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
from time import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import logging

# create logger with 'poi_id'

logger = logging.getLogger('poi_id')

LOG_FILENAME = 'poi_id.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, filemode='w')
logging.Formatter('%(asctime)s - %(message)s')

logger.info('Start of the script')

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

POI_label = ['poi']

"""Used features
features_list is a list of strings, each of which is a feature name.
The first feature must be "poi"."""
features_list = POI_label + financial_features + ['to_messages', 'from_poi_to_this_person', 'from_messages',
                                                  'from_this_person_to_poi',
                                                  'shared_receipt_with_poi']  # ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus']  # You will need to use more features

# Load the dictionary containing the dataset
dataset = "final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

# Initial exploration
logger.info("The number of people contained in the dataset is:" + str(len(data_dict.keys())))
logger.info("The number of features for each person is:" + str(len(data_dict['METTS MARK'].keys())))

#Number of POI
poi = 0
for person in data_dict.keys():
    if data_dict[person]["poi"] == 1:
        poi += 1

logger.info("The number of POI contained in the dataset is: {poi}")


# Identifying the people with a lot of data missing to eliminate them from the dataset
threshold = 15
low_data_ppl = []
missing_features = []
for person in data_dict.keys():
    count = 0
    for feature in financial_features + email_features:
        if data_dict[person][feature] == "NaN":
            count += 1
    if count > threshold:
        low_data_ppl.append(person)
    data_dict[person]["count"] = count
    missing_features.append(count)

x_pos = np.arange(len(missing_features))
y_value = np.array(missing_features)
# split it up
above_threshold = np.maximum(y_value - threshold, 0)
below_threshold = np.minimum(y_value, threshold)

# and plot it
fig, ax = plt.subplots()
ax.bar(x_pos, below_threshold, 0.35, color="g")
ax.bar(x_pos, above_threshold, 0.35, color="r",
        bottom=below_threshold)

# horizontal line indicating the threshold
ax.plot([0., x_pos[-1]], [threshold, threshold], "k--")

plt.xlabel('People in the dataset')
plt.ylabel('[#] Missing values')
plt.title('Missing values in the dataset')
fig.savefig('missing_values.png')

# Elimination of the people with a lot of data missing. Verification that no POI is being eliminated
for person in low_data_ppl:
    logger.info(f"{person} will be eliminated from the dataset. POI status: {data_dict[person]['poi']}")
    data_dict.pop(person)

# Identification of the people with extreme of the features to find more outliers

for feature in financial_features + email_features + ["count"]:
    key_max = max(data_dict.keys(), key=lambda k: data_dict[k][feature]
    if isinstance(data_dict[k][feature], int) else float("-inf"))
    key_min = min(data_dict.keys(), key=lambda k: data_dict[k][feature]
    if isinstance(data_dict[k][feature], int) else float("+inf"))
    max_value = data_dict[key_max][feature]
    min_value = data_dict[key_min][feature]

    logger.info(f"{key_max} is the person with the max {feature}: {max_value} ")
    logger.info(f"{key_min} is the person with the min {feature}: {min_value}")


"""Task 2: Remove outliers"""

data_dict.pop("TOTAL", 0)

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

data = featureFormat(data_dict, features_list)

triplets = [['salary', 'bonus', 'poi'] , ['salary', 'total_stock_value', 'poi'],
            ['from_this_person_to_poi', 'from_poi_to_this_person', 'poi']]


def plot_features_scatter(data, features_list, x, y, z):
    x_feat = features_list.index(x)
    y_feat = features_list.index(y)
    z_feat = features_list.index(z)
    x_values = []
    y_values = []
    z_values = []
    for point in data:
        x_values.append(point[x_feat])
        y_values.append(point[y_feat])
        z_values.append(point[z_feat])
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, y_values, c=z_values, label=['non POI', 'POI'], cmap='cool')
    legend1 = ax.legend(*scatter.legend_elements(num=1),
                        loc="upper left", title="POI")
    ax.add_artist(legend1)
    plt.xlabel(x)
    plt.ylabel(y)
    filename = x + '_' + y + '_' + z + '_scatter.png'
    plt.savefig(filename)


for xyz in triplets:
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    plot_features_scatter(data, features_list, x, y, z)



"""Task 3: Create new feature(s)
Store to my_dataset for easy export below."""

"""Interaction with POI. Check how often mails from this persons are sent/received to POI"""
my_dataset = data_dict

for person in my_dataset.keys():
    if my_dataset[person]["from_messages"] != "NaN" and my_dataset[person]["to_messages"] != "NaN" and \
            my_dataset[person]["from_this_person_to_poi"] != "NaN" and \
            my_dataset[person]["from_poi_to_this_person"] != "NaN":
        my_dataset[person]["interaction_POI"] = \
            (my_dataset[person]["from_this_person_to_poi"] + my_dataset[person]["from_poi_to_this_person"]) / \
            (my_dataset[person]["from_messages"] + my_dataset[person]["to_messages"])
        my_dataset[person]["ratio_from_POI"] = \
            my_dataset[person]["from_poi_to_this_person"] / my_dataset[person]["from_messages"]
        my_dataset[person]["ratio_to_POI"] = \
            my_dataset[person]["from_this_person_to_poi"] / my_dataset[person]["from_messages"]
    else:
        my_dataset[person]["interaction_POI"] = "NaN"
        my_dataset[person]["ratio_from_POI"] = "NaN"
        my_dataset[person]["ratio_to_POI"] = "NaN"

# Extract features and labels from dataset for local testing
features_list.append("interaction_POI")
features_list.append('ratio_from_POI')
features_list.append('ratio_to_POI')

data = featureFormat(my_dataset, features_list, sort_keys=True)

x = 'ratio_from_POI'
y = "ratio_to_POI"
z = 'poi'
plot_features_scatter(data, features_list, x, y, z)

labels, features = targetFeatureSplit(data)



"""Task 4: Try a variety of classifiers
Please name your classifier clf for easy export below.
Note that if you want to do PCA or other multi-stage operations,
you'll need to use Pipelines. For more info:
http://scikit-learn.org/stable/modules/pipeline.html"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

features = np.array(features)
labels = np.array(labels)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2)

# As the labels are very unbalanced StratifiedShuffleSplit was used to separate train and test. Only 1 split was created.
#sss = StratifiedShuffleSplit(n_splits=200, test_size=0.2)

#for train_index, test_index in sss.split(features, labels):
#    features_train, features_test = features[train_index], features[test_index]
#    labels_train, labels_test = labels[train_index], labels[test_index]
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3)

# Param space definition for the PCA and the different models
param_space = {
    SVC: [
        {
            'pca__n_components': [5, 10, 12],
            'clf__kernel': ['sigmoid', 'rbf'],
            'clf__C': [1, 10, 100, 1000],
            'clf__gamma': ['scale'],
        }
    ],
    DecisionTreeClassifier: [
        {
            'pca__n_components': [5, 10, 12],
            'clf__criterion': ['gini', 'entropy'],
            'clf__min_samples_split': [2, 4, 6, 8],
            'clf__max_depth': [2, 5, None],
        }
    ],
    KNeighborsClassifier: [
        {
            'pca__n_components': [5, 10, 12],
            'clf__n_neighbors': [2, 3, 5, 8],
            'clf__weights': ['distance', 'uniform'],
            'clf__algorithm': ['kd_tree', 'ball_tree', 'auto']
        }
    ],
}

# Models testing
logger.info('Testing of the different models')
print('Testing of the different models')
cm = {}
cr = {}
best_param = {}

models_to_test = [SVC, DecisionTreeClassifier, KNeighborsClassifier]

for Model in models_to_test:
    logger.info('Model: {0}'.format(Model))
    t0 = time()
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('clf', Model()),
    ])
    parameters = param_space[Model]
    clf = GridSearchCV(pipe, parameters, scoring='f1', n_jobs=-1, cv=50, verbose=0)
    clf.fit(features_train, labels_train)

    labels_pred = clf.predict(features_test)

    cm[Model] = confusion_matrix(labels_test, labels_pred)
    cr[Model] = classification_report(labels_test, labels_pred)
    best_param[Model] = clf.best_params_

    logger.info("done in %0.3fs" % (time() - t0))

for Model in models_to_test:
    logger.info('Model: {0}'.format(Model))
    logger.info('The confusion matrix is: {0}'.format(cm[Model]))
    logger.info('The classification report is:')
    logger.info(cr[Model])
    logger.info('The parameters chosen are: {0}'.format(best_param[Model]))

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

for best_model in [
    SVC]:  # [SVC, DecisionTreeClassifier, KNeighborsClassifier]: #, RandomForestClassifier]:

    pipe_best = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('clf', best_model()),
    ])
    clf = pipe_best.set_params(**best_param[best_model])

    clf.fit(features_train, labels_train)

    labels_pred = clf.predict(features_test)

    logger.info('Model: {0}'.format(Model))
    logger.info('The confusion matrix is: {0}'.format(confusion_matrix(labels_test, labels_pred)))
    logger.info('The classification report is:')
    logger.info(classification_report(labels_test, labels_pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
