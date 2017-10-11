##### LINEAR REGRESSION - PSEUDOINVERSION #####

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats             # for verification
from sklearn import linear_model    # for verification
from sklearn.metrics import mean_squared_error, r2_score


data = np.loadtxt('data.txt')       # shape (50, 2)

x = data[:, 0]      # shape (50,)
y = data[:, 1]      # shape (50,)

plt.figure(1, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt.scatter(x, y, color='purple', label='original data')


### Pseudoinversion method

print("Pseudoinversion method")

X = np.ones((data.shape[0], 1))
X = np.concatenate((X, x[:, np.newaxis]), axis=1)

theta_optimal = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)       # shape (2,)

y_hat = theta_optimal[0] + theta_optimal[1] * x

plt.plot(x, y_hat, 'g-', linewidth=7.0, label = 'pseudoinv fitted line')

print("theta_optimal:", theta_optimal)      # [-10.61262592, 6.90359569]
print("R-squared:", r2_score(y, y_hat))     # 0.96


### Verification

# Scipy
print("\nScipy")
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("intercept, slope:", intercept, slope)
#print(np.allclose(theta_optimal, [intercept, slope]))
y_pred = intercept + slope * x
plt.plot(x, y_pred, color='red', linewidth=3.0, label = 'scipy fitted line')
print("R-squared:", r_value**2)             # 0.96
print("R-squared:", r2_score(y, y_pred))    # 0.96

# scikit-learn Linear Regression
print("\nscikit-learn Linear Regression")
reg = linear_model.LinearRegression()
reg.fit(x.reshape(-1, 1), y)
y_pred = reg.predict(x.reshape(-1, 1))
plt.plot(x, y_pred, color='black', linestyle='-', linewidth=1.0, label = 'scikit LR fitted line')
print("intercept, slope:", reg.intercept_, reg.coef_)
print("R-squared:", r2_score(y, y_pred))    # 0.96

# scikit-learn Ridge
print("\nscikit-learn Ridge")
clf = linear_model.Ridge()
clf.fit(x[:, np.newaxis], y)
y_pred = clf.predict(x[:, np.newaxis])
plt.plot(x, y_pred, color='orange', linestyle='', marker='2', label = 'scikit Ridge fitted line')
print("intercept, slope:", clf.intercept_, clf.coef_)
print("R-squared:", r2_score(y, y_pred))    # 0.96

plt.xlabel('data')
plt.ylabel('target')
plt.legend()

plt.figure(2, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt.scatter(x, y, color='purple', label='original data')
plt.plot(x, y_hat, 'k-', linewidth=2.0, label = 'pseudoinv fitted line')

# SGD works better with features centered around zero, with standard deviation of 1.
mean, std = x.mean(), x.std()
x_scaled = (x - mean) / std

# scikit-learn SGDRegressor
print("\nscikit-learn SGDRegressor")
clf = linear_model.SGDRegressor(loss="squared_loss", penalty="l2", max_iter=1000, tol=1e-3, shuffle=True)
clf.fit(x_scaled[:, np.newaxis], y)
y_pred = clf.predict(x_scaled[:, np.newaxis])
plt.plot(x, y_pred, color='yellow', linestyle='-', linewidth=1.0, label = 'scikit SGDRegressor fitted line')
print("intercept, slope:", clf.intercept_, clf.coef_)
print("R-squared:", r2_score(y, y_pred))    # 0.96

plt.xlabel('data')
plt.ylabel('target')
plt.legend()

plt.show()
