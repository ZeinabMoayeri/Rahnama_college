# -*- coding: utf-8 -*-
""" Question 1 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

q_1=pd.read_csv('question_1.csv')

y_coordinates = q_1.feature_1
x_coordinates = q_1.feature_2

# Scatter plot
plt.scatter(x_coordinates, y_coordinates, label='All Locations')
plt.xlabel('X Coordinate (2_feature)')
plt.ylabel('Y Coordinate (1_feature)')
plt.title('Locations')
plt.legend()
plt.show()

""" The solution with k-nearest neighbors (KNN) algorithm """

# Sample data creation for demonstration
# Remove this if you have your own data to load
data = {
    '1_feature': y_coordinates,
    '2_feature': x_coordinates
}
df = pd.DataFrame(data)

# Load your actual data
# df = pd.read_csv('your_data.csv')  # Uncomment and modify this line to load your data

def initialize_centroids(df, k):
    centroids = df.sample(n=k).to_numpy()
    return centroids

def assign_clusters(df, centroids):
    distances = np.linalg.norm(df.to_numpy()[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(df, labels, k):
    new_centroids = np.array([df.to_numpy()[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def knn_clustering(df, k, max_iters=100):
    centroids = initialize_centroids(df, k)
    for _ in range(max_iters):
        labels = assign_clusters(df, centroids)
        new_centroids = update_centroids(df, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

# Number of clusters (teams)
k = 5
df['Number_team'] = knn_clustering(df[['1_feature', '2_feature']], k) + 1  # Adding 1 to make teams 1-indexed instead of 0-indexed

# Plotting the points with their team assignments
plt.figure(figsize=(10, 8))
for team in range(1, k + 1):
    team_data = df[df['Number_team'] == team]
    plt.scatter(team_data['1_feature'], team_data['2_feature'], label=f'Team {team}')

plt.xlabel('1_feature')
plt.ylabel('2_feature')
plt.title('Team Distribution')
plt.legend()
plt.show()

# Save the dataframe with team assignments to a new CSV file
df.to_csv('team_assigned_data.csv', index=False)

""" The solution with sklearn KMeans"""

# Sample data creation for demonstration
# Remove this if you have your own data to load
data = {
    '1_feature': y_coordinates,
    '2_feature': x_coordinates
}
df = pd.DataFrame(data)

# Load your actual data
# df = pd.read_csv('your_data.csv')  # Uncomment and modify this line to load your data

# Apply KMeans to divide the points into 5 clusters (teams)
kmeans = KMeans(n_clusters=5, random_state=0).fit(df[['1_feature', '2_feature']])
df['Number_team'] = kmeans.labels_ + 1  # Adding 1 to make teams 1-indexed instead of 0-indexed

# Plotting the points with their team assignments
plt.figure(figsize=(10, 8))
for team in range(1, 6):
    team_data = df[df['Number_team'] == team]
    plt.scatter(team_data['1_feature'], team_data['2_feature'], label=f'Team {team}')

plt.xlabel('1_feature')
plt.ylabel('2_feature')
plt.title('Team Distribution')
plt.legend()
plt.show()

# Save the dataframe with team assignments to a new CSV file
df.to_csv('team_assigned_data.csv', index=False)



#-------------------------------------------------------------------------------------------------------------

""" Question 2 """

# !pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

import pandas as pd
import numpy as np
import pandas_profiling
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df=pd.read_csv('question_2.csv')


profile = ProfileReport(df, title="data set", html={'style' : {'full_width':True}})
profile.to_file(output_file="Overview_of_dataset.html")


# Separate features and target variable
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each classifier
best_classifier = None
best_accuracy = 0
accuracies = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = name

print("Accuracies of the classifiers:")
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy:.4f}")

print(f"\nBest classifier is: {best_classifier} with accuracy of {best_accuracy:.4f}")

# Now use the best classifier to get feature importances
if best_classifier == "Decision Tree":
    feature_importances = classifiers[best_classifier].feature_importances_
elif best_classifier == "Random Forest":
    feature_importances = classifiers[best_classifier].feature_importances_
else:
    print(f"Feature importance is not directly available for {best_classifier}.")
    feature_importances = None

if feature_importances is not None:
    sorted_indices = np.argsort(feature_importances)[::-1]
    top_6_indices = sorted_indices[:6]
    top_6_features = X.columns[top_6_indices]

    print("\nThe top 6 features for predicting the target y are:")
    for i, idx in enumerate(top_6_indices):
        print(f"{i+1}. {X.columns[idx]} with importance score {feature_importances[idx]}")

    # Plot feature importances
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_indices], feature_importances[sorted_indices], align='center')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Ranking')
    plt.gca().invert_yaxis()
    plt.show()