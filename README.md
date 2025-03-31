# Machine Learning Basics

Repository này chứa các thuật toán Machine Learning được tự cài đặt dựa trên thiết kế module của scikit-learn.

## Mục đích

- Hiểu sâu cách hoạt động của các thuật toán ML thông qua việc tự cài đặt
- Học cấu trúc API và thiết kế của scikit-learn

## Cấu trúc repository

```
.
├── algorithms/      # Các thuật toán ML
├── utils/           # Các công cụ hỗ trợ
└── tests/           # Unit tests
```

## Thuật toán

- Linear Models
- Tree-based Models
- Clustering
- Dimensionality Reduction
- Neural Networks

## Ví dụ sử dụng

```python
from algorithms.linear_models import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Tài liệu tham khảo

- Scikit-learn documentation
- Machine Learning cơ bản