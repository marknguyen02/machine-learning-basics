# Machine Learning Basics

This repository contains self-implemented Machine Learning algorithms based on the modular design of scikit-learn.

## Purpose

- Gain a deeper understanding of how ML algorithms work by implementing them from scratch.
- Learn the API structure and design principles of scikit-learn

## Algorithms

- Linear Model
- Tree-based Model
- Clustering
- Dimensionality Reduction
- Neural Network

## Usage Example

```python
from linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```