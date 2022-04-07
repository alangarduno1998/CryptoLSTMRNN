from matplotlib import pyplot as plt
import numpy as np


def _gauss_basis(x, y,s,  axis=None):
    arg = (x-y) / s
    return np.exp(-0.5 * arg ** 2)
def pure_t(X):
    return np.sin(2 * np.pi * X)
def noise(X):
    return np.random.normal(0, 0.3, size=X.shape)
def generate_data(N):
    X = np.random.uniform(0, 1, size=(N,))
    t = pure_t(X) + noise(X)
    return X, t
def plot_dataset_and_func(X, Y, g):
    """
    Plots (X, Y) and the function G
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([-2, 2])

    xs = np.arange(0, 1, 0.001)
    plt.plot(xs, g(xs), color='green')

    plt.scatter(X, Y, color='blue')
def plot_dataset_and_2func(X, Y, g, h, name_g='', name_h=''):
    """
    Plots (X, Y) and the functions g and h
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([-2, 2])
    xs = np.arange(0, 1, 0.001)
    plt.plot(xs, g(xs), color='green')
    plt.plot(xs, h(xs), color='magenta')
    plt.legend([name_g, name_h])

    plt.scatter(X, Y, color='blue')
def fit_polynomial(degree, X, t, lambda_reg):
    """
    Regularized Linear regression with non-linear model to find polynomial coefficients
    """
    N = len(t)
    # We are transforming X -> Z
    # (x) -> (1, x^1, x^2, ..., x^degree)
    Z = np.ones((N, degree + 1))

    # y = np.linspace(Z.min(), Z.max(), N)
    y = np.mean(Z)
    mu_values = [0, 0.2,0.2,0.3,0.3, 0.5]
    for i in range(degree + 1):
        Z[:, i] = X* (_gauss_basis(X,i*0.1, s=0.1, axis=1))
        Z[:, i] = X* (_gauss_basis(X,mu_values[i], s=0.1, axis=1))
    print(Z.shape)

    # Returning the matrix multiplication of pseudo-inverse and target values
    return np.matmul((np.linalg.inv(Z.T @ Z+ lambda_reg*np.identity(Z.shape[1])) @ Z.T), t)

def g(X, W):
    """
    Calculates the value of polynomial with coefficients W
    """
    N = len(X)
    degree = len(W) - 1
    mu_values = [0, 0.2,0.2,0.3,0.3, 0.5]
    # We are transforming X -> Z
    # (x) -> (1, x^1, x^2, ..., x^degree)
    Z = np.ones((N, degree + 1))
    for i in range(degree + 1):
        # Z[:, i] = X* _gauss_basis(X,i*0.1,s=0.1, axis=1)
        Z[:, i] = X* _gauss_basis(X,mu_values[i],s=0.1, axis=1)
    return np.dot(Z, W)


N_train = 25
N_test = 1000
X_train_tot, t_train_tot = [],[]
for i in range(100):
    X_train_sin, t_train_sin = generate_data(N_train)
    X_train_tot.append(X_train_sin)
    t_train_tot.append(t_train_sin)
X_train, t_train = np.mean(X_train_tot, axis=0), np.mean(t_train_tot, axis=0)
X_test, t_test = generate_data(N_test)
# plot_dataset_and_func(X_train, t_train, pure_t)

train_error = []
test_error = []
bias_error = []
variance_error = []
bias_variance_error = []

# degrees = list(range(0, 10))
degree = 5
lambda_regs = [np.exp(i) for i in list(range(-2,3))]
# lambda_regs = list(range(-3,2))
w_mul = []

for lambda_reg in lambda_regs:
    for i in range(100):
        w_single = fit_polynomial(degree, X_train_tot[i], t_train_tot[i], lambda_reg)
        w_mul.append(w_single)
    print(len(w_mul))
    w=np.mean(w_mul, axis=0)

    # plot_dataset_and_2func(X_train, t_train, pure_t, lambda x: g(x, w), 'Target', f'{degree} degree Polynomial')

    train_error.append(np.mean((g(X_train, w) - t_train) ** 2))
    # if (lambda_reg>=0):
    #     test_error.append(np.mean((g(X_test, w) - t_test) ** 2)+lambda_reg*(np.linalg.norm(w)**2))
    # else:
    #     test_error.append(np.mean((g(X_test, w) - t_test) ** 2))
    test_error.append(np.mean((g(X_test, w) - t_test) ** 2)/4)
    # variance = np.mean(g(X_train, w)**2) - np.mean(g(X_train, w))**2
    # variance = np.mean((np.mean(g(X_train, w))- g(X_train, w))**2)
    variance = np.mean((g(X_train,w) - np.mean(g(X_train,w)))**2)
    variance_error.append(variance)
    # bias = np.sum((t_train - np.mean(g(X_train, w)))**2)/N_train
    # bias = np.mean((t_train - np.mean(g(X_train, w))))**2
    bias = np.mean(((g(X_train, w))-t_train)**2)
    bias_error.append(bias)
    bias_variance_error.append(variance + bias)

plt.plot(np.log(lambda_regs)[::-1], bias_error, '-', color='blue') #[::-1]
plt.plot(np.log(lambda_regs), variance_error, '-', color='red')
plt.plot(np.log(lambda_regs)[::-1], bias_variance_error, '-', color='magenta')
plt.plot(np.log(lambda_regs), test_error, '-', color='black')
plt.legend(["(bias)$^2$", "variance", "(bias)$^2$+variance", "test error"])
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(0, 0.16, step=0.03))  # Set label locations.
plt.xlabel("ln $λ$")
plt.show()