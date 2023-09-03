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



