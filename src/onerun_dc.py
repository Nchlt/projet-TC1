from sklearn.metrics.pairwise import pairwise_distances
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv("../data/train.csv").drop("id",axis=1)
df_test = pd.read_csv("../data/test.csv").drop("id",axis=1)

labels = df["target"].values
print(type(labels))
print(labels.shape)

df_train = df.drop("target", axis=1)

X = df_train.values
X_test = df_test.values

# print(X.shape)
# print(X_test.shape)

y = []
for label in labels:
    y.append(int(label[-1:])-1)
y = np.asarray(y)

unique_labels = np.unique(y)

tf = TfidfTransformer()

print("Construction des centroïdes")

X_tf = tf.fit_transform(X)
X_test_tf = tf.fit_transform(X_test)

centroids = []
for label in unique_labels:
    same_labels = X_tf[np.where(y == label)]
    centroids.append(same_labels.sum(axis = 0))
centroids = np.asarray(centroids).reshape(9,93)
print(centroids.shape)
print("Construction terminée")
X_dc = pairwise_distances(X_tf, centroids, metric='euclidean')
X_test_dc = pairwise_distances(X_test_tf, centroids, metric='euclidean')

X_train = X_dc
X_test = X_test_dc

print("X_dc shape : ", X_dc.shape)
print("X_test_dc shape : ", X_test_dc.shape)


# clf_tmp = KNeighborsClassifier(n_neighbors=150, n_jobs=-1)
# clf = CalibratedClassifierCV(clf_tmp, method='isotonic', cv=5)
# clf = SVC(gamma=0.001, C=100, probability=True, cache_size=4000, verbose=True,
#             class_weight='balanced')
# %time clf_2.fit(X_train, y_train)
# print("Fit terminé")
# %time pred_2 = clf_2.predict_proba(X_test)
# pred_for_acc_2 = clf_2.predict(X_test)
# print("%.2f" % log_loss(y_test, pred_2, eps=1e-15, normalize=True))
# print(classification_report(y_test, pred_for_acc_2, target_names=target_names))
#clf_tmp = ExtraTreesClassifier(n_estimators=2000, random_state=0, n_jobs=-1)
clf_tmp = SVC(gamma=0.1, C=200, probability=True, cache_size=4000, verbose=True,
            class_weight='balanced')
clf = CalibratedClassifierCV(clf_tmp, cv=5)

# Entrainement
print('Entrainement en cours')
# clf.fit(X_train, y_train)
clf.fit(X_train, y)
print('Entrainement terminé')
pred_proba = clf.predict_proba(X_test)
pickle.dump(pred_proba, open('../data/brute.csv', 'wb'))
print('Données pickélisées')
