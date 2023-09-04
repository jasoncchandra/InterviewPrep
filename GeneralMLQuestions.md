## 1. Say you are building an UNbalanced dataset for a binary classifier (have cancer, or no have cancer), how do you handle the situaiton?

Resampling:

Oversampling the Minority Class: This involves creating additional copies of instances from the minority class to balance the dataset. For example, if you have a dataset with 1000 no-cancer cases and only 100 cancer cases, you might randomly duplicate the 100 cancer cases to match the size of the majority class.
Undersampling the Majority Class: In this case, you randomly remove instances from the majority class to achieve a balance. For instance, if you have the same dataset, you might randomly select 100 samples from the no-cancer class to match the size of the minority class.
Example: In a fraud detection system, where fraudulent transactions are rare, you could oversample the fraud cases or undersample the non-fraud cases to create a balanced dataset.

Generate Synthetic Data:

Synthetic data generation techniques like SMOTE (Synthetic Minority Over-sampling Technique) create new instances for the minority class by interpolating between existing instances. SMOTE algorithmically generates synthetic data points between existing minority class points, reducing the risk of overfitting.
Example: In a medical diagnosis task, if you have limited data for a rare disease, SMOTE can be used to generate synthetic patient cases with the disease.

Cost-Sensitive Learning:

Assign different misclassification costs to classes to make the model more sensitive to the minority class. By doing this, you prioritize minimizing the errors for the minority class, even if it means accepting more errors in the majority class.
Example: In a credit fraud detection system, the cost of missing a fraudulent transaction is much higher than flagging a legitimate transaction as fraudulent. You can assign a higher cost to misclassifying the minority class (fraudulent transactions).

Ensemble Methods:

Ensemble methods combine the predictions of multiple models. When dealing with imbalanced datasets, they can be used with appropriate class weights or resampling strategies to improve model performance. Algorithms like Random Forests, AdaBoost, or Gradient Boosting can be configured to handle class imbalance.
Example: In a churn prediction task for a subscription service, you can use an ensemble of decision trees with adjusted class weights to predict customer churn.

Threshold Adjustment:

Adjusting the classification threshold determines the point at which a model assigns a class label. In imbalanced datasets, you can lower the threshold to increase the sensitivity to the minority class, at the cost of potentially more false positives in the majority class.
Example: In a medical test for a life-threatening disease, you might lower the classification threshold to ensure that as many true cases as possible are detected, even if it means more false alarms.

Evaluation Metrics:

Be cautious with evaluation metrics. Accuracy may not be a good measure in unbalanced datasets. Instead, focus on metrics like precision, recall, F1-score, or the area under the ROC curve (AUC-ROC) to assess model performance more accurately.



## 2. What are some differences when you minimize squared error vs absolute error? Which error cases would each metric be appropriate?

MSE:
Characteristics:
Squaring the errors penalizes large errors more heavily than small errors. This makes MSE sensitive to outliers.
It gives more weight to data points with large errors, which can be a problem if you want to prioritize accuracy for all data points.
MSE is differentiable, which makes it suitable for optimization using techniques like gradient descent.

Appropriate Use Cases:
MSE is often used when you want the model to have strong incentives to minimize the impact of outliers or extreme errors.
It is commonly used in situations where the distribution of errors is assumed to be Gaussian (normal), as many statistical methods are based on this assumption.

MAE:
Characteristics:
MAE treats all errors equally, regardless of their size. It is less sensitive to outliers compared to MSE.
It provides a more robust measure of central tendency and is less affected by extreme values.
MAE is not differentiable at zero, which can make optimization more challenging in some cases.

Appropriate Use Cases:
MAE is often preferred when you want the model's performance to be less influenced by outliers or when you have reason to believe that the error distribution is not necessarily Gaussian.
It is commonly used in situations where you want a more interpretable error metric, as the absolute values of errors are easier to understand than squared values.


## 3. When performing K-means clustering, how to choose K?

- K-means clustering is an unsupervised machine learning algorithm for data clustering and partitioning.
- It groups similar data points into clusters based on their features.
- The algorithm involves the following steps:
1) Initialize K cluster centroids randomly within the data space.
2) Assign each data point to the cluster whose centroid is closest (Assignment Step).
3) Recalculate the centroids by taking the mean of data points in each cluster (Update Step).
4) Repeat the Assignment and Update steps until convergence.
- The choice of the number of clusters (K) is crucial and can impact the quality of clustering results.
- Common methods to determine an optimal K include the Elbow method and the Silhouette method.
- K-means is sensitive to the initial centroids and is biased toward spherical clusters.
- It's widely used for data analysis, image processing, customer segmentation, and more.


Elbow Method:

The Elbow method is a graphical technique to find the optimal number of clusters (K).
It involves running K-means with different values of K and plotting the sum of squared distances (inertia) versus K.
Look for an "elbow point" in the plot, where the inertia starts to level off.
The K value at the elbow point is often considered the optimal number of clusters.


Silhouette Method:

The Silhouette method assesses the quality of clustering for different values of K.
It calculates a silhouette score for each K, measuring how similar data points are to their own cluster compared to other clusters.
Higher silhouette scores indicate better cluster separation.
Calculate the mean silhouette score for each K and choose the K with the highest score as the optimal number of clusters.

In summary, the Elbow method looks for an inflection point in the inertia plot, while the Silhouette method evaluates the quality of clustering using silhouette scores. These methods help determine the most suitable number of clusters for a given dataset and problem.
