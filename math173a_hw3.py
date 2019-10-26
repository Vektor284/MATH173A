import numpy as np
import matplotlib.pyplot as plt

n = 50
T = 100
x = np.random.normal(0.0,1, [T, n])
w_0 = np.random.normal(0.0, 1, n)


def rho_tilde(x):
    return np.log(1 + np.exp(x))


def rho_tilde_gradient(x, w, w_0):
    partial_grad = np.zeros(n)
    w_t_x = np.zeros(T)
    c_i = np.zeros(T)

    for i in range(T):
        w_t_x[i] = np.dot(x[i], w)
    # For c_i's
    for i in range(T):
        c_i[i] = np.dot(x[i], w_0)

    for i in range(T):
        temp1 = (np.exp(w_t_x[i])) / (1 + np.exp(w_t_x[i]))
        temp2 = 2 * (rho_tilde(w_t_x[i]) - c_i[i])
        partial_grad += temp2 * temp1 * x[i]

    return partial_grad


def gradient_descent(x, w_0, num_itera, mu):
    temp_w = w_0
    w_t_stack = np.empty((0,len(w_0)))

    for i in range(num_itera):
        temp_w = temp_w - (mu * rho_tilde_gradient(x, temp_w, w_0))
        w_t_stack = np.vstack((w_t_stack,temp_w))

    return temp_w, w_t_stack


partial = rho_tilde_gradient(x, w_0,w_0)

optimal_w, stack = gradient_descent(x, w_0, 100, 0.00001)

# print(partial)
print(w_0)
print(optimal_w)
print(stack)

def g(stack,x, num_iter):
    w_t_x = np.zeros(num_iter)


    #
    # for i in range(num_iter):
    #     w_t_x = np.dot(stack[i],x[i])
    #

# plt.scatter(stack)
# plt.show()
