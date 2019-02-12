import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils


def Zero_padding(arr, rang = 2):
    if rang == 2:
        result = np.zeros((arr.shape[0] + 2, arr.shape[1] + 2))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i + 1, j + 1] = arr[i, j]
        return result
    elif rang == 3:
        result = np.zeros((arr.shape[0], arr.shape[1] + 2, arr.shape[2] + 2))
        for l in range(arr.shape[0]):
            for i in range(arr.shape[1]):
                for j in range(arr.shape[2]):
                    result[l,i + 1, j + 1] = arr[l, i, j]
        return result



def Convolution(input_data, weights, rang = 2, back_prop = True):
    if rang == 2:
        l0 = input_data.copy()
        fill = np.zeros((weights.shape[0], l0.shape[0] - weights.shape[1] + 1, l0.shape[1] - weights.shape[2] + 1))
        for l in range(weights.shape[0]):
            k = 0
            maxim = []
            for i in range(l0.shape[0] - weights.shape[1] + 1):
                for j in range(l0.shape[1] - weights.shape[2] + 1):
                    if back_prop == False:
                        sum = np.sum(l0[i:i + weights.shape[1], j:j + weights.shape[2]] * weights[l, :, :])
                    elif back_prop == True:
                        sum = np.sum(
                            l0[i:i + weights.shape[1], j: j + weights.shape[2]] * (np.rot90(weights[l, :, :])))
                        sum = ReLU(sum)
                    maxim.append(sum)
            for i in range(int(np.sqrt(len(maxim)))):
                for j in range(int(np.sqrt(len(maxim)))):
                    fill[l, i, j] = maxim[k]
                    k += 1
        return fill

    elif rang == 3:
        l0 = input_data.copy()
        fill = np.zeros((weights.shape[0], l0.shape[1] - weights.shape[1] + 1, l0.shape[2] - weights.shape[2] + 1))
        for m in range(l0.shape[0]):
            for l in range(weights.shape[0]):
                k = 0
                maxim = []
                for i in range(l0.shape[1] - weights.shape[1] + 1):
                    for j in range(l0.shape[2] - weights.shape[2] + 1):
                        if back_prop == False:
                            sum = np.sum(l0[m, i:i + weights.shape[1], j:j + weights.shape[2]] * weights[l, :, :])
                        elif back_prop == True:
                            sum = np.sum(l0[m, i:i + weights.shape[1], j: j + weights.shape[2]] * (np.rot90(weights[l, :, :])))
                            sum = ReLU(sum)
                        maxim.append(sum)
                for i in range(int(np.sqrt(len(maxim)))):
                    for j in range(int(np.sqrt(len(maxim)))):
                        fill[l, i, j] = maxim[k]
                        k += 1
        return fill

def max_pool(input_data):
    rows = input_data.shape[1]/2
    cols = input_data.shape[2]/2
    l0 = input_data.copy()
    fill = np.zeros((l0.shape[0], int(rows), int(cols)))
    for l in range(l0.shape[0]):
        maxim = []
        k = 0
        for i in range(0, input_data.shape[1], 2):
            for j in range(0, input_data.shape[2], 2):
                maxim.append(np.max(l0[l, i:i + 2, j:j + 2]))

        for i in range(int(np.sqrt(len(maxim)))):
            for j in range(int(np.sqrt(len(maxim)))):
                fill[l, i, j] = maxim[k]
                k += 1
    return fill


def max_pool_delta(input_data, rows, cols, delta):
    k = 0
    l0 = input_data.copy()
    for m in range(l0.shape[0]):
        for i in range(0, rows, 2):
            for j in range(0, cols, 2):
                if l0[m, i, j] == np.max(l0[m, i:i + 2, j:j + 2]):
                    l0[m, i, j] = delta[k]
                    k += 1
                elif l0[m, i, j + 1] == np.max(l0[m, i:i + 2, j:j + 2]):
                    l0[m, i, j + 1] = delta[k]
                    k += 1
                elif l0[m, i + 1, j] == np.max(l0[m, i:i + 2, j:j + 2]):
                    l0[m, i + 1, j] = delta[k]
                    k += 1
                elif l0[m, i + 1, j + 1] == np.max(l0[m, i:i + 2, j:j + 2]):
                    l0[m, i + 1, j + 1] = delta[k]
                    k += 1
    return l0


def Dense(arr, weights):
    k = 0
    sum = 0
    l0 = arr.copy()
    result = []
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            sum += weights[i, j] * l0[k]
            k += 1

        sum = ReLU(sum)
        result.append(sum)
        sum = 0
        k = 0
    return result


def ReLU (x, type = False, deriv=False):
    if (x < 0):
        return 0
    if (x >= 0):
        return x


def Derivative_ReLU_dence(x):
    for i in range(x.shape[0]):
        if (x[i] < 0):
            x[i] = 0
        if (x[i] >= 0 ):
            x[i] = 1
    return x


def Derivative_ReLU_Conv(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if (x[i, j, k] < 0):
                    x[i, j, k] = 0
                if (x[i, j, k] >= 0 ):
                    x[i, j, k] = 1
    return x


def SoftMax(x, deriv=False):
    if (deriv == True):
        return np.exp(x)/np.exp(x).sum()*(1 - np.exp(x)/np.exp(x).sum())
    return np.exp(x) / np.exp(x).sum()


X = np.array([[[2, 1, 2, 3],
               [1, 4, 5, 6],
               [2, 7, 8, 9],
               [5, 4, 3, 2]],

              [[1, 4, 5, 6],
               [2, 7, 8, 9],
               [1, 4, 5, 6],
               [2, 1, 2, 3]]])
Y = np.ones((2, 4, 4))
Weights_X = np.ones((4, 3, 3))
Weights_Y = np.ones((4, 3, 3))

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_test /= 255
X_train /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

img_rows, img_cols = 28, 28


#np.random.seed(1000)

#Weights = np.ones((3,3))
# Weights for convolution
Weights_Conv_1_1 = np.random.uniform(0.0001, 0.001, (4, 3, 3))
Weights_Conv_1_2 = np.random.uniform(0.0001, 0.001, (4, 3, 3))
# Weights for convolution
Weights_Conv_2_1 = np.random.uniform(0.0001, 0.001, (8, 3, 3))
Weights_Conv_2_2 = np.random.uniform(0.0001, 0.001, (8, 3, 3))
# Weights for convolution
Weights_Conv_3_1 = np.random.uniform(0.0001, 0.001, (8, 3, 3))
Weights_Conv_3_2 = np.random.uniform(0.0001, 0.001, (4, 3, 3))
Weights_Conv_3_3 = np.random.uniform(0.0001, 0.001, (2, 3, 3))
Weights_Conv_3_4 = np.random.uniform(0.0001, 0.001, (1, 3, 3))
# Weights for FullyConnected layers
Weights_dense_0 = np.random.uniform(0.0001, 0.001, (10, 49))
Weights_dense_1 = np.random.uniform(0.001, 0.01, (64, 64))
Weights_dense_2 = np.random.uniform(0.001, 0.01, (10, 64))

# Matrix for keeping activation function result
convar = np.zeros((X_train.shape[0], 1, 7, 7))
convarr = np.zeros((X_train.shape[0], 2, 2))
flutt = np.zeros((X_train.shape[0], convar.shape[1] * convar.shape[2]*convar.shape[3]))
endarr = np.zeros((Y_test.shape[0], Y_test.shape[1]))


for itr in range(X_train.shape[0]):
    print("PICTURE № ", itr + 1, " :\n")
    layer0 = X_train[itr, :, :].copy()
    layer0_zero_pad = Zero_padding(layer0, 2)

    layer1 = Convolution(layer0_zero_pad, Weights_Conv_1_1)
    layer1_zero_pad = Zero_padding(layer1, 3)

    layer2 = Convolution(layer1_zero_pad, Weights_Conv_1_2, 3)
    layer2_max_pool = max_pool(layer2)
    layer2_zero_pad = Zero_padding(layer2_max_pool, 3)

    layer3 = Convolution(layer2_zero_pad, Weights_Conv_2_1, 3)
    layer3_zero_pad = Zero_padding(layer3, 3)

    layer4 = Convolution(layer3_zero_pad, Weights_Conv_2_2, 3)
    layer4_max_pool = max_pool(layer4)
    layer4_zero_pad = Zero_padding(layer4_max_pool, 3)

    layer5 = Convolution(layer4_zero_pad, Weights_Conv_3_1, 3)
    layer5_zero_pad = Zero_padding(layer5, 3)

    layer6 = Convolution(layer5_zero_pad, Weights_Conv_3_2, 3)
    layer6_zero_pad = Zero_padding(layer6, 3)

    layer7 = Convolution(layer6_zero_pad, Weights_Conv_3_3, 3)
    layer7_zero_pad = Zero_padding(layer7, 3)

    layer8 = Convolution(layer7_zero_pad, Weights_Conv_3_4, 3)
    convar[itr, :, :, :] = layer8
    flutt[itr, :] = convar[itr, :, :, :].ravel()

    layer9 = np.array(Dense(flutt[itr, :], Weights_dense_0))

    layer10 = SoftMax(np.array(layer9))

    # Backpropagation, Error function for each layer

    layer10_err = np.array((Y_train[itr, :] - layer10))
    layer10_del = np.array(layer10_err * SoftMax(layer10, True))

    flutt_err = np.transpose(Weights_dense_0).dot(layer10_del)
    flutt_del = flutt_err * Derivative_ReLU_dence(flutt[itr, :])
    layer8_del = flutt_del.reshape(layer8.shape)
    layer8_del_zp = Zero_padding(layer8_del, 3)

    layer7_err = np.zeros(layer7.shape)
    layer7_err[0:1, :, :] = Convolution(layer8_del_zp, Weights_Conv_3_4, 3, False)
    layer7_del = layer7_err * Derivative_ReLU_Conv(layer7)
    layer7_del_zp = Zero_padding(layer7_del, 3)

    layer6_err = np.zeros((layer6.shape))
    layer6_err[0:2, :, :] = Convolution(layer7_del_zp[0, :, :], Weights_Conv_3_3, 2, False)
    layer6_err[2:4, :, :] = Convolution(layer7_del_zp[1, :, :], Weights_Conv_3_3, 2, False)
    layer6_del = layer6_err * Derivative_ReLU_Conv(layer6)
    layer6_del_zp = Zero_padding(layer6_del, 3)

    layer5_err = np.zeros((layer5.shape))
    layer5_err[0:4, :, :] = Convolution(layer6_del_zp[0:2, :, :], Weights_Conv_3_2, 3, False)
    layer5_err[4:8, :, :] = Convolution(layer6_del_zp[2:4, :, :], Weights_Conv_3_2, 3, False)
    layer5_del = layer5_err * Derivative_ReLU_Conv(layer5)
    layer5_del_zp = Zero_padding(layer5_del, 3)
    layer5_del_1 = max_pool_delta(layer4, layer4.shape[1], layer4.shape[2], layer5_del.ravel())
    layer5_del_1_zp = Zero_padding(layer5_del_1, 3)

    layer4_err = Convolution(layer5_del_1_zp, Weights_Conv_3_1, 3, False)
    layer4_del = layer4_err * Derivative_ReLU_Conv(layer4)
    layer4_del_zp = Zero_padding(layer4_del, 3)

    layer3_err = Convolution(layer4_del_zp, Weights_Conv_2_2, 3, False)
    layer3_del = layer3_err * Derivative_ReLU_Conv(layer3)
    layer2_del_mp = max_pool_delta(layer2, layer2.shape[1], layer2.shape[2], layer3_del.ravel())
    layer2_del_mp_zp = Zero_padding(layer2_del_mp, 3)

    layer2_err = Convolution(layer2_del_mp_zp, Weights_Conv_2_1, 3, False)
    layer2_err = layer2_err[0:4, :, :] + layer2_err[4:8, :, :]
    layer2_del = layer2_err * Derivative_ReLU_Conv(layer2)
    layer2_del_zp = Zero_padding(layer2_del, 3)

    layer1_err = Convolution(layer2_del_zp, Weights_Conv_1_2, 3, False)
    layer1_del = layer1_err * Derivative_ReLU_Conv(layer1)
    layer1_del_zp = Zero_padding(layer1_del, 3)

    layer0_del = Convolution(layer1_del_zp, Weights_Conv_1_1, 3, False)

    # Gradient Descent for Convolution & MLP matrix
    WD0_del = np.zeros((Weights_dense_0.shape))
    for i in range(layer10_del.shape[0]):
        WD0_del[i, :] = Weights_dense_0[i, :] * layer10_del[i]

    WC34_del = Convolution(layer7_zero_pad, layer8_del, 3)
    WC33_del = Convolution(layer6_zero_pad, layer7_del, 3)
    WC32_del = Convolution(layer5_zero_pad, layer6_del, 3)
    WC31_del = Convolution(layer4_zero_pad, layer5_del, 3)

    WC22_del = Convolution(layer3_zero_pad, layer4_del, 3)
    WC21_del = Convolution(layer2_zero_pad, layer3_del, 3)

    WC12_del = Convolution(layer1_zero_pad, layer2_del, 3)
    WC11_del = Convolution(layer0_zero_pad, layer1_del, 2)

    # Changing the matrix of weights
    Weights_Conv_1_1 += WC11_del
    Weights_Conv_1_2 += WC12_del

    Weights_Conv_2_1 += WC21_del
    Weights_Conv_2_2 += WC22_del

    Weights_Conv_3_1 += WC31_del
    Weights_Conv_3_2 += WC32_del
    Weights_Conv_3_3 += WC33_del
    Weights_Conv_3_4 += WC34_del

    Weights_dense_0 += WD0_del














































