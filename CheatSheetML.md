## Cheat Sheet for ML - basic overview.
General Topics:

**Linear Algebra:**

- In machine learning, linear algebra is a foundational mathematical framework used to represent and manipulate data and models. It provides tools to understand relationships between variables, optimize models, and perform various operations efficiently. Key concepts include vectors and matrices for data representation, linear transformations for feature engineering, eigenvalues and eigenvectors for dimensionality reduction, and matrix operations for model training and optimization. Understanding linear algebra is crucial for grasping the inner workings of machine learning algorithms, from linear regression to deep neural networks, and for effectively working with data in higher-dimensional spaces.

**Gradient Descent:**

- Gradient descent is a fundamental optimization algorithm in machine learning used to minimize the cost or loss function of a model during training. It operates by iteratively adjusting model parameters in the direction of the steepest descent of the cost function, as determined by the gradient (derivative). In each iteration, it calculates the gradient of the cost function with respect to the model parameters and updates the parameters by taking a step proportional to the negative gradient. This process continues until convergence, where the cost function reaches a minimum or a predetermined number of iterations is reached. Gradient descent is vital for training various machine learning models, including neural networks, linear regression, and logistic regression, and is crucial for finding the optimal model parameters that best fit the data.

Objective Function (Cost Function): Typically denoted as J(θ), where θ represents the model parameters (weights and biases).
Gradient of the Cost Function: ∇J(θ), which is the vector of partial derivatives of J(θ) with respect to each parameter.
Update Rule: θ := θ - α * ∇J(θ), where α (alpha) is the learning rate, a hyperparameter that controls the step size in each iteration.
Iteration: This update rule is applied iteratively until convergence is reached or a specified number of iterations is completed.
- Stoachstic Gradient Descent adds an element of randomness so that the gradient does not get stuck.

