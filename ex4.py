import numpy as np
import matplotlib.pyplot as plt


def create_train_val_test(num, data):
    data = data.reshape(-1, 1)
    x = np.repeat(data, num, axis=1)
    x **= np.arange(num)
    return x.transpose()


def train_model(all_w, loss_on_valid, loss_on_train, num, y_train, y_valid,
                x_train, x_valid):
    x = create_train_val_test(num + 1, x_train)
    x_x_t = np.dot(x, x.transpose())
    x_x_t_inv = np.linalg.pinv(x_x_t)
    x_y = np.dot(x, y_train)
    w_star = np.dot(x_x_t_inv, x_y)
    all_w.append(w_star)

    x_t = x.transpose()
    result = np.dot(x_t, w_star)
    loss = np.sum(np.square(result - y_train), 0) / y_train.shape[0]
    loss_on_train.append(loss)

    x_val = create_train_val_test(num + 1, x_valid)
    x_val_t = x_val.transpose()
    result = np.dot(x_val_t, w_star)
    loss = np.sum(np.square(result - y_valid), 0) / y_valid.shape[0]
    loss_on_valid.append(loss)


def run_least_square_alg(y_train, y_valid, y_test, x_train, x_valid, x_test):
    all_w = []
    loss_on_train = []
    loss_on_valid = []
    for i in range(1, 16):
        train_model(all_w, loss_on_valid, loss_on_train, i, y_train, y_valid,
                    x_train, x_valid)

    all_w = np.array(all_w)
    loss_on_valid = np.array(loss_on_valid)
    loss_on_train = np.array(loss_on_train)
    ind = np.argmin(loss_on_valid)
    print(ind)
    w_star = all_w[ind]
    x_final_test = create_train_val_test(w_star.shape[0], x_test)
    x_test_t = x_final_test.transpose()
    result = np.dot(x_test_t, w_star)
    loss = np.sum(np.square(result - y_test), 0) / y_test.shape[0]

    print(w_star)
    print(loss_on_valid[ind])
    print(loss)

    d = np.arange(1, 16)
    plt.figure()
    plt.plot(d, loss_on_valid, 'g^', d, loss_on_valid, 'g',
             label='loss on valid')

    plt.plot(d, loss_on_train, 'bs', d, loss_on_train, 'b',
             label='loss on train')
    plt.title('train and validate errors degree=1,...,15')
    plt.legend(loc='upper right')

    plt.figure()
    plt.plot(d[1:], loss_on_valid[1:], 'g^', d[1:], loss_on_valid[1:], 'g',
             label='loss on valid')
    plt.plot(d[1:], loss_on_train[1:],
             'bs', d[1:], loss_on_train[1:], 'b', label='loss on train')
    plt.title('train and validate errors degree=2,...,15')
    plt.legend(loc='upper right')

    plt.show()


def least_square(train_set, train_label_set, d):
    x = create_train_val_test(d + 1, train_set)
    x_x_t = np.dot(x, x.transpose())
    x_x_t_inv = np.linalg.pinv(x_x_t)
    x_y = np.dot(x, train_label_set)
    w_star = np.dot(x_x_t_inv, x_y)
    return w_star


def get_loss(h, untrained_set, untrained_set_label):
    x = create_train_val_test(h.shape[0], untrained_set)
    x_t = x.transpose()
    result = np.dot(x_t, h)
    loss = np.sum(np.square(result - untrained_set_label), 0) / \
           untrained_set_label.shape[0]
    return loss


def run_k_fold_alg(x_fold, y_fold):
    x_split = np.split(x_fold, 5)
    y_split = np.split(y_fold, 5)

    error = []

    for i in range(1, 16):
        loss_h = []
        for j in range(5):
            x = np.concatenate([x_split[k] for k in range(5) if k != j])
            y = np.concatenate([y_split[k] for k in range(5) if k != j])

            h = least_square(x, y, i)
            loss = get_loss(h, x_split[j], y_split[j])
            loss_h.append(loss)
        loss_h = np.array(loss_h)
        error.append(np.sum(loss_h) / 5)

    error = np.array(error)
    # print(error)
    ind = np.argmin(error)
    print(ind)

    h_star = least_square(x_fold, y_fold, ind + 1)
    print(h_star)

    d = np.arange(1, 16)
    plt.figure()
    plt.plot(d, error, 'g^', d, error, 'g')
    plt.title('errors degree=1,...,15')

    plt.figure()
    plt.plot(d[1:], error[1:], 'g^', d[1:], error[1:], 'g')
    plt.title('errors degree=2,...,15')

    plt.show()


if __name__ == "__main__":
    y_poly = np.load("./Y_poly.npy")
    y_train, y_valid, y_test = np.split(y_poly, 3)
    x_poly = np.load("./X_poly.npy")
    x_train, x_valid, x_test = np.split(x_poly, 3)

    run_least_square_alg(y_train, y_valid, y_test, x_train, x_valid, x_test)

    x_fold = np.concatenate((x_train, x_valid))
    y_fold = np.concatenate((y_train, y_valid))

    run_k_fold_alg(x_fold, y_fold)
