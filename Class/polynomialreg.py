import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
np.random.seed(100)
# Generate training datasets
N_train = 10
X_train = np.random.uniform(low=0,high=1,size=10)
print(X_train.shape)
f = 1
t_train = np.sin(2*np.pi*f*X_train) + np.random.normal(scale=0.1, size=len(X_train))

# Generate test datasets
X_test = np.random.uniform(0,1,100)
t_test = np.sin(2*np.pi*f*X_test) + np.random.normal(scale=0.1, size = len(X_test))
# plt.plot(t_test)

# Use the method of linear regression with non-linear models to fit polynomials of
# degree 𝑀𝑀 = 0,1,2, … ,9 to the training set.
M = 10
rmse_test_total = []
rmse_train_total = []
m_total = []
for m in range(M):
    m_total.append(m)
    t_train_poly = np.polyfit(X_train, t_train, deg=m)
    rmse_train = np.sqrt(mean_squared_error(t_train, np.polyval(t_train_poly, X_train)))
    print(rmse_train)
    rmse_train_total.append(rmse_train)
    t_test_poly = np.polyfit(X_test, t_test, deg=m)
    rmse_test = np.sqrt(mean_squared_error(t_test, np.polyval(t_test_poly, X_test)))
    print(rmse_test)
    rmse_test_total.append(rmse_test)

fig, ax3 = plt.subplots()
ax3.plot(m_total,rmse_train_total,'bo-', label='Training', mfc='none')
ax3.plot(m_total, rmse_test_total, 'ro-', label='Test', mfc='none')
plt.xlabel("M"), plt.ylabel("Erms")
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 10, step=3))  # Set label locations.
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(0, 1.1, step=0.5))  # Set label locations.
ax3.legend()

# Repeat the exercise for 𝑁𝑇rain = 100
# Generate training datasets
N_train = 100
X_train = np.linspace(0,1,100)
f = 1
t_train = np.sin(2*np.pi*f*X_train) + np.random.normal(scale=0.1, size=len(X_train))
# plt.plot(t_train)

# Generate test datasets
N_test = 100
X_test = np.linspace(0,1,100)
t_test = np.sin(2*np.pi*f*X_test) + np.random.normal(scale=0.1, size = len(X_test))
# plt.plot(t_test)

# Use the method of linear regression with non-linear models to fit polynomials of
# degree 𝑀𝑀 = 0,1,2, … ,9 to the training set.
M = 10
rmse_test_total = []
rmse_train_total = []
m_total = []
for m in range(M):
    m_total.append(m)
    t_train_poly = np.polyfit(X_train, t_train, deg=m)
    rmse_train = np.sqrt(np.square(np.linalg.norm(np.polyval(t_train_poly, X_train)-t_train, ord = 2))/N_train)
    # rmse_train = np.sqrt(mean_squared_error(t_train, np.polyval(t_train_poly, X_train)))
    print(rmse_train)
    rmse_train_total.append(rmse_train)
    t_test_poly = np.polyfit(X_test, t_test, deg=m)
    rmse_test = np.sqrt(np.square(np.linalg.norm(np.polyval(t_test_poly, X_test)-t_test, ord = 2))/N_test)
    # rmse_test = np.sqrt(mean_squared_error(t_test, np.polyval(t_test_poly, X_test)))
    print(rmse_test)
    rmse_test_total.append(rmse_test)

fig, ax4 = plt.subplots()
ax4.plot(m_total, rmse_train_total,'bo-', label='Training', mfc='none')
ax4.plot(m_total, rmse_test_total, 'ro-', label='Test', mfc='none')
ax4.legend()
plt.xlabel("M"), plt.ylabel("Erms")
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 10, step=3))  # Set label locations.
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(0, 1.1, step=0.5))  # Set label locations.
plt.show()