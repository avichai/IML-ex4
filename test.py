import numpy as np
from matplotlib import pyplot as plt

POLY_DEGREES = list(range(1, 16))
K_FOLD_CONST = 5


def create_X_mat(X, n_params):
    X = X.reshape(-1, 1)
    X_mat = np.repeat(X, n_params, axis=1)
    X_mat **= np.arange(n_params)
    return X_mat


def train(X, Y, d):
    n_params = d + 1
    X_mat = create_X_mat(X, n_params)
    X_mat_T = X_mat.T

    # least squares
    return np.dot(np.dot(np.linalg.pinv(np.dot(X_mat_T, X_mat)), X_mat_T), Y)


def predict(X, w):
    n_params = w.shape[0]
    X_mat = create_X_mat(X, n_params)
    return np.dot(X_mat, w)


def calc_error(Y, Y_hat):
    return np.sum((Y - Y_hat) ** 2) / Y.shape[0]


def kfold_val(X, Y, k):
    X_par = np.split(X, k)
    Y_par = np.split(Y, k)

    kfold_errs = []
    for d in POLY_DEGREES:
        err = 0
        for i in range(k):
            X_no_i = np.concatenate([X_par[j] for j in range(k) if j != i])
            Y_no_i = np.concatenate([Y_par[j] for j in range(k) if j != i])
            w = train(X_no_i, Y_no_i, d)
            err += calc_error(Y_par[i], predict(X_par[i], w))
        err /= k
        kfold_errs.append(err)

    val_d = POLY_DEGREES[np.argmin(kfold_errs)]
    val_w = train(X, Y, val_d)
    return val_w


def plot_errors(title, errors):
    plt.figure()
    plt.suptitle(title)
    plt.subplot(121)
    plt.title("d = 1,...,15")
    plt.plot(POLY_DEGREES, errors)
    plt.subplot(122)
    plt.title("d = 2,...,15")
    plt.plot(POLY_DEGREES[1:], errors[1:])


def main():
    # load data
    X = np.load("X_poly.npy")
    Y = np.load("y_poly.npy")

    # split data
    X_train, X_val, X_test = np.split(X, 3)
    Y_train, Y_val, Y_test = np.split(Y, 3)

    # train w_d for each d in degrees (each w_d represents the coefficients of the polynomial)
    ws = [train(X_train, Y_train, d) for d in POLY_DEGREES]

    # calc the train and validation error of each classifier
    train_errs = [calc_error(Y_train, predict(X_train, w)) for w in ws]
    val_errs = [calc_error(Y_val, predict(X_val, w)) for w in ws]

    # find the classifier which minimizes the validation loss
    w_star_ind = np.argmin(val_errs)
    w_star = ws[w_star_ind]

    # running k-fold to find best classifier
    X_kfold = np.concatenate((X_train, X_val))
    Y_kfold = np.concatenate((Y_train, Y_val))
    w_star_kfold = kfold_val(X_kfold, Y_kfold, K_FOLD_CONST)

    # print validated classifier's test error
    w_star_test_err = calc_error(Y_test, predict(X_test, w_star))
    w_star_kfold_test_err = calc_error(Y_test, predict(X_test, w_star_kfold))
    print(
        "w* =          {}".format([float("{:.6f}".format(c)) for c in w_star]))
    print("w*_kfold =    {}".format(
        [float("{:.6f}".format(c)) for c in w_star_kfold]))
    print("w*-w*_kfold = {}".format(
        [float("{:.6f}".format(c)) for c in (w_star - w_star_kfold)]))
    print()
    print("w* test error =       {:.6f}".format(w_star_test_err))
    print("w*_kfold test error = {:.6f}".format(w_star_kfold_test_err))

    # plotting train and validation errors
    plot_errors("Train error", train_errs)
    plot_errors("Validation error", val_errs)

    plt.show()


if __name__ == "__main__":
    main()
