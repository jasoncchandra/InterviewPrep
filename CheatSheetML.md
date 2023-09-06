## Cheat Sheet for ML - basic overview.

General Topics/Concepts:

**Linear Algebra:**

- In machine learning, linear algebra is a foundational mathematical framework used to represent and manipulate data and models. It provides tools to understand relationships between variables, optimize models, and perform various operations efficiently. Key concepts include vectors and matrices for data representation, linear transformations for feature engineering, eigenvalues and eigenvectors for dimensionality reduction, and matrix operations for model training and optimization. Understanding linear algebra is crucial for grasping the inner workings of machine learning algorithms, from linear regression to deep neural networks, and for effectively working with data in higher-dimensional spaces.

**Gradient Descent:**

- Gradient descent is a fundamental optimization algorithm in machine learning used to minimize the cost or loss function of a model during training. It operates by iteratively adjusting model parameters in the direction of the steepest descent of the cost function, as determined by the gradient (derivative). In each iteration, it calculates the gradient of the cost function with respect to the model parameters and updates the parameters by taking a step proportional to the negative gradient. This process continues until convergence, where the cost function reaches a minimum or a predetermined number of iterations is reached. Gradient descent is vital for training various machine learning models, including neural networks, linear regression, and logistic regression, and is crucial for finding the optimal model parameters that best fit the data.

Objective Function (Cost Function): Typically denoted as J(θ), where θ represents the model parameters (weights and biases).

Gradient of the Cost Function: ∇J(θ), which is the vector of partial derivatives of J(θ) with respect to each parameter.

Update Rule: θ := θ - α * ∇J(θ), where α (alpha) is the learning rate, a hyperparameter that controls the step size in each iteration.

Iteration: This update rule is applied iteratively until convergence is reached or a specified number of iterations is completed.

- Stoachstic Gradient Descent adds an element of randomness so that the gradient does not get stuck. It uses one data point at a time and uses smaller subsect of datapoint.

**Model Evaluation and Selection**

In model evaluation, the primary objective is to measure a model's performance using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the problem type (classification or regression). Cross-validation techniques, like k-fold cross-validation, are often employed to obtain a robust estimate of a model's performance and assess its generalization ability. It is essential to carefully select the right evaluation metrics that align with the problem's goals and interpret the results in the context of the problem domain.

- It is also important to recognize how well the TEST SET performs on the TRAIN SET (usually 80%)

In model selection, the focus shifts to comparing multiple candidate models to identify the one that performs the best. Common techniques include grid search and randomized search, which explore different hyperparameter settings to find the optimal configuration. Additionally, domain knowledge and intuition can guide the selection process. The chosen model should strike a balance between complexity and generalization, ensuring it can make accurate predictions on new, unseen data. Overall, model evaluation and selection are iterative processes that require a deep understanding of the data, problem, and the strengths and weaknesses of different algorithms, making them essential topics for machine learning interviews.


**Bias Variance Tradeoff**


