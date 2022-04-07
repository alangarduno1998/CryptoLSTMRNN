import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
seed=200
for s in range(seed):
    np.random.seed(s+100)
    # Repeat the exercise for 𝑁𝑇rain = 10

    N_train = 10
    X_train = np.random.uniform(0,1,N_train)
    # X_train = (X - X.min()) / (X.max() - X.min())
    t_train = np.sin(2*np.pi*X_train) + np.random.normal(scale=0.1, size=N_train)
    # t_train = (t - t.min()) / (t.max() - t.min())
    N_test = 100
    X_test = np.random.uniform(0,1,N_test)
    # X_test = (X - X.min()) / (X.max() - X.min())
    t_test = np.sin(2*np.pi*X_test) + np.random.normal(scale=0.1, size = N_test)
    M = 10
    rmse_test_total, rmse_train_total, m_total = [], [], []
    params = [-3, -2, -1, 0, 1, 2] # using ridge regression so there is reg^2
    reg = -3
    for reg in params:
        m_total.append(reg)

        #uncomment this for second implementation
        for j in range(M):
            if j == 0:
                X = np.ones((N_train, 1))
            if j != 0:
                x_pow = np.power(X_train, j)
                X = np.append(X, x_pow.reshape(-1, 1), axis=1)
        # coeffs = np.linalg.pinv(X).dot(t_train)
        I = np.identity(X.shape[1])
        coeffs = (np.linalg.inv(X.T @ X + reg*I) @ X.T @ t_train)
        x_line = np.linspace(X_train.min(), X_train.max(), N_train)
        for j in range(M):
            if j == 0:
                t_train_poly = coeffs[0]
                x_line = np.linspace(X_train.min(), X_train.max(), 1)
            if j != 0: t_train_poly += coeffs[j] * np.power(X_train, j)
        rmse_train = np.sqrt(np.square(np.linalg.norm(t_train_poly-t_train, ord = 2))/N_train)

        print(rmse_train)
        rmse_train_total.append(rmse_train)

        #uncomment this for second implementation
        x_line_test = np.linspace(X_test.min(), X_test.max(), N_test)
        for j in range(M):
            if j == 0:
                t_test_poly = coeffs[0]
            if j >= 0: t_test_poly += coeffs[j] * np.power(X_test, j)
        rmse_test = np.sqrt(np.square(np.linalg.norm(t_test_poly-t_test, ord = 2))/N_test)

        print(rmse_test)
        rmse_test_total.append(rmse_test)
    print(m_total)
    print(rmse_test_total)
    fig, ax2 = plt.subplots()
    ax2.plot(m_total, rmse_train_total,'bo-', label='Training', mfc='none')
    ax2.plot(m_total, rmse_test_total, 'ro-', label='Test', mfc='none')
    ax2.legend()
    plt.xlabel("M"), plt.ylabel("Erms")
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 10, step=3))  # Set label locations.
    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(0, 1.1, step=0.5))  # Set label locations.
    plt.title("N_train = 10")
    print(s)
    plt.show()
    # Repeat the exercise for 𝑁𝑇rain = 100
    N_train = 100
    X_train = np.random.uniform(0,1,N_train)
    # X_train = (X - X.min()) / (X.max() - X.min())
    t_train = np.sin(2*np.pi*X_train) + np.random.normal(scale=0.1, size=N_train)
    # t_train = (t - t.min()) / (t.max() - t.min())
    N_test = 100
    X_test = np.random.uniform(0,1,N_test)
    # X_test = (X - X.min()) / (X.max() - X.min())
    t_test = np.sin(2*np.pi*X_test) + np.random.normal(scale=0.1, size = N_test)
    M = 10
    rmse_test_total, rmse_train_total, m_total = [], [], []
    for m in range(M):
        m_total.append(m)

        #uncomment this for second implementation
        for j in range(m + 1):
            if j == 0:
                X = np.ones((N_train, 1))
            if j != 0:
                x_pow = np.power(X_train, j)
                X = np.append(X, x_pow.reshape(-1, 1), axis=1)
        # coeffs = np.linalg.pinv(X).dot(t_train)
        I = np.identity(X.shape[1])
        coeffs = (np.linalg.inv(X.T @ X + reg*I) @ X.T @ t_train)
        x_line = np.linspace(X_train.min(), X_train.max(), N_train)
        for j in range(m + 1):
            if j == 0:
                t_train_poly = coeffs[0]
                x_line = np.linspace(X_train.min(), X_train.max(), 1)
            if j != 0: t_train_poly += coeffs[j] * np.power(X_train, j)
        rmse_train = np.sqrt(np.square(np.linalg.norm(t_train_poly-t_train, ord = 2))/N_train)

        print(rmse_train)
        rmse_train_total.append(rmse_train)

        #uncomment this for second implementation
        x_line_test = np.linspace(X_test.min(), X_test.max(), N_test)
        for j in range(m + 1):
            if j == 0:
                t_test_poly = coeffs[0]
            if j >= 0: t_test_poly += coeffs[j] * np.power(X_test, j)
        rmse_test = np.sqrt(np.square(np.linalg.norm(t_test_poly-t_test, ord = 2))/N_test)

        print(rmse_test)
        rmse_test_total.append(rmse_test)
    print(m_total)
    print(rmse_test_total)
    fig, ax2 = plt.subplots()
    ax2.plot(m_total, rmse_train_total,'bo-', label='Training', mfc='none')
    ax2.plot(m_total, rmse_test_total, 'ro-', label='Test', mfc='none')
    ax2.legend()
    plt.xlabel("M"), plt.ylabel("Erms")
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 10, step=3))  # Set label locations.
    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(0, 1.1, step=0.5))  # Set label locations.
    plt.title("N_train = 100")
    print(s)
    plt.show()