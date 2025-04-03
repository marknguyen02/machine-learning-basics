# Machine Learning Basics

Repository này chứa các thuật toán Machine Learning được tự cài đặt dựa trên thiết kế module của scikit-learn.

## Mục đích

- Hiểu sâu cách hoạt động của các thuật toán ML thông qua việc tự cài đặt
- Học cấu trúc API và thiết kế của scikit-learn


## Thuật toán

- Linear Model
- Tree-based Model
- Clustering
- Dimensionality Reduction
- Neural Network

## Ví dụ sử dụng

```python
from linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```