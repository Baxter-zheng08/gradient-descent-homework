import numpy as np

# Loss Class
class Loss:
    def __init__(self):
        pass

    def mse(self, y_pred, y_true):
        n = y_true.shape[0]
        return np.sum((y_pred - y_true) ** 2) / n

    def compute_gradient(self, X, y_pred, y_true):
        n = y_true.shape[0]
        grad = (2 / n) * X.T @ (y_pred - y_true)
        return grad

# GradientDescent Optimizer Class
class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weights, grad):
        weights = weights - self.lr * grad
        return weights

# 模拟UCI格式数据集（离线，无需网络）
def load_simulated_uci_data():
    np.random.seed(0)
    n_samples = 1500
    X_raw = np.random.randn(n_samples,5)
    y = (3*X_raw[:,0] + 2*X_raw[:,1] + X_raw[:,2] + np.random.randn(n_samples)*0.5).reshape(-1,1)

    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    X_scaled = (X_raw - mean)/std
    X = np.hstack([np.ones((X_scaled.shape[0],1)), X_scaled])
    return X,y

# 主程序
if __name__ == "__main__":
    X, y = load_simulated_uci_data()
    num_features = X.shape[1]
    w = np.zeros((num_features,1))
    loss_calculator = Loss()
    optimizer = GradientDescent(0.01)
    epochs = 1500
    for epoch in range(epochs):
        y_pred = X @ w
        loss = loss_calculator.mse(y_pred,y)
        grad = loss_calculator.compute_gradient(X,y_pred,y)
        w = optimizer.update(w,grad)
        if epoch%200==0:
            print(f"Iteration {epoch:4d} | MSE Loss = {loss:.4f}")
    print("\n训练完成，权重：")
    print(w)
