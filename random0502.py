import numpy as np


# step1: covariance matrix szamitas
def calculate_covariance_matrix(std_x, std_y, corr):
    cov_xy = std_x * std_y * corr
    cov_matrix = np.array([[std_x ** 2, cov_xy], [cov_xy, std_y**2]])
    return cov_matrix


# step2: hozamok generalasa
def calc_asset_returns(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


def test_calc_asset_returns(mean, cov, size):
    corr = 0.1
    covmat = calculate_covariance_matrix(0.2, 0.1, corr)
    print(covmat)
    means = [0.15, 0.05]
    nsim = 10000
    rets = calc_asset_returns(means, covmat, nsim)
    print(rets.mean(axis=0))
    print(np.cov(rets.transpose()))


# step3: atkonvertalni effektiv hozamokra
def convert_continuous_to_simple(rets):
    return np.log(rets) - 1

# step4: pf hozam = sulyozott eff hozamok osszege
# step5: pf erteke
