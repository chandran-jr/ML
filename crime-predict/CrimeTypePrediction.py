import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

data = pd.read_csv("crime.csv")
demo = data.iloc[0:10000]

visual = demo['TYPE'].value_counts()
fig, ax = plt.subplots()
width = 0.4
ax.barh(visual.index, visual.values, width, color='Orange')
ax.set_ylabel('No.of cases', color='aqua')
ax.set_xlabel('Cities', color='aqua')
ax.set_title('CRIME NUMBER GRAPH', color='red')
plt.savefig('barchart.png')

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(visual.values, explode=explode, labels=visual.index, autopct='%1.1f%%',
        shadow=True, startangle=30)
ax1.axis('equal')
plt.savefig('piechart.png')

# Removing Outliers
value_c = demo['Latitude'].value_counts()
rem = value_c[value_c < 10].index
demo = demo[~demo['Latitude'].isin(rem)]

# Separating inputs and outputs
x = demo.iloc[:, [10, 11]].values.reshape(-1, 2)
y = demo['TYPE'].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train Test spliting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.20)

# Normalization
from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
x_train = norm.fit_transform(x_train)
x_test = norm.transform(x_test)

cov_mat = np.cov(x_train.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# calculate cumulative sum of explained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
# plt.bar(range(1,3), var_exp, alpha=0.5,align='center', label='individual explained variance')
# plt.step(range(1,3), cum_var_exp, where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))  # , eigen_pairs[2][1][:, np.newaxis], eigen_pairs[3][1][:, np.newaxis]))

x_train_pca = x_train.dot(w)
x_test_pca = x_test.dot(w)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_pca, y_train)
pred = knn.predict(x_test_pca)
# print(confusion_matrix(y_test, pred))
score = cross_val_score(knn, x_train_pca, y_train, cv=5)


import pickle
pickle.dump(knn, open('model.pkl', 'wb'))
pickle.dump(norm, open('norm.pkl', 'wb'))
pickle.dump(le, open('le.pkl', 'wb'))
pickle.dump(w, open('weight.pkl', 'wb'))
