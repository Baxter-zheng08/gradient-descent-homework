# Gradient Descent Assignment: UCI Real Estate Valuation Dataset
# Dataset ID = 477
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


real_estate_valuation = fetch_ucirepo(id=477)
X_raw = real_estate_valuation.data.features.values
y_raw = real_estate_valuation.data.targets.values.ravel()


X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m_train = X_train.shape[0]
X_train = np.hstack([np.ones((m_train, 1)), X_train])
m_test = X_test.shape[0]
X_test = np.hstack([np.ones((m_test, 1)), X_test])


def compute_cost(X, y, theta):
    m = len(y)
    y_pred = X @ theta
    cost = (1/(2*m)) * np.sum((y_pred - y)**2)
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        y_pred = X @ theta
        grad = (1/m) * X.T @ (y_pred - y)
        theta = theta - learning_rate * grad
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history


feature_num = X_train.shape[1]
theta_init = np.zeros(feature_num)
lr = 0.1
iter_num = 2000

theta_final, cost_hist = gradient_descent(X_train, y_train, theta_init, lr, iter_num)
print("Training finished, final parameter theta:")
print(theta_final)
print(f"Final loss on training set: {cost_hist[-1]:.4f}")


def evaluate_model(X_test, y_test, theta):
    y_predict = X_test @ theta
    mse = np.mean((y_predict - y_test)**2)
    rmse = np.sqrt(mse)
    print("\n======= Evaluation on unseen test dataset =======")
    print(f"MSE = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    return y_predict


y_pred_test = evaluate_model(X_test, y_test, theta_final)
