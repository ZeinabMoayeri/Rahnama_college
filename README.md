# Question 1

First, we call the desired libraries. Then we also call the database that we uploaded to the drive. We separate the first two rows under the title of the desired points and show an overview of all the places with the matplotlib library.
In the next step, we solved the algorithm using k-nearest neighbors (KNN) algorithm and then using kmeans.
 At the end, all selected teams are added to the dataset in an additional column and stored in the team_assigned_data file.

![download](C:\Users\nano\Desktop\Zeinab_Moayeri\images\download.png)

# Question 2

The impressive features are X4, X2, X5, and X6 respectively. And then X 1. And it can be said that the most unrelated feature is X3.
First, we call the desired libraries. Then we also call the dataset that we uploaded in the drive. Then we visualize our data using the pandas library so that we can fully understand what the data is. In the next part, we will save this illustration in HTML file and then we will notice the important features according to the heatmap dataset.
In the next part of the code, we want to understand two things by using all the classifier models: first, which features are important to us in this dataset, and second, which classifier model has the best performance on this dataset.
And according to our code, we have obtained the most accuracy from the Random Forest model with an accuracy of 0.96, which has recognized our features as follows: X4, X2, X5, X6, X1, and X3.

 

1. X4 with importance score 0.24704827702708296
2. X2 with importance score 0.24341428509726884
3. X5 with importance score 0.21275237611202358
4. X6 with importance score 0.15847297730558146
5. X1 with importance score 0.12478589348827071
6. X3 with importance score 0.013526190969772358

![download (2)](C:\Users\nano\Desktop\Zeinab_Moayeri\images\download (2).png)

Considering that in the last part of the code, we have said to use all the categories and we have also calculated its accuracy. It is as follows:

Accuracy of the classifiers:
Decision Tree: 0.9400
Random Forest: 0.9600
Logistic Regression: 0.8300
SVC: 0.8750
KNN: 0.9150
The best classification is Random Forest with an accuracy of 0.96.