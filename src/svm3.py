import numpy as np


def confusion_matrix(y_actual, y_predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_actual)):
        if y_actual[i] > 0:
            if y_actual[i] == y_predicted[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        if y_actual[i] < 1:
            if y_actual[i] == y_predicted[i]:
                tn = tn + 1
            else:
                fp = fp + 1

    cm = [[tn, fp], [fn, tp]]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sens = tp/(tp+fn)
    prec = tp/(tp+fp)
    f_score = (2*prec*sens)/(prec+sens)
    return cm, accuracy, f_score

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    numerator = np.linalg.norm(x-y)**2
    denominator = 2 * (sigma ** 2)
    return np.exp(-numerator / denominator)


class SVM(object):

    def __init__(self, kernel=linear_kernel, tol=1e-3, C=0.1,
                 max_passes=5, sigma=0.1):

        self.kernel = kernel
        self.tol = tol
        self.C = C
        self.max_passes = max_passes
        self.sigma = sigma
        self.model = dict()

    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"kernel={self.kernel.__name__}, "
                f"tol={self.tol}, "
                f"C={self.C}, "
                f"max_passes={self.max_passes}, "
                f"sigma={self.sigma}"
                ")")
    
    def svmTrain(self, X, Y):
        # Data parameters
        m = X.shape[0]

        # Map 0 to -1
        Y = np.where(Y == 0, -1, 1)

        # Variables
        alphas = np.zeros((m, 1), dtype=float)
        b = 0.0
        E = np.zeros((m, 1), dtype=float)
        passes = 0

        # Pre-compute the kernel matrix
        if self.kernel.__name__ == 'linear_kernel':
            print(f'Pre-computing {self.kernel.__name__} matrix')
            K = X @ X.T

        elif self.kernel.__name__ == 'gaussian_kernel':
            print(f'Pre-computing {self.kernel.__name__} matrix')
            X2 = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
            K = X2 + (X2.T - (2 * (X @ X.T)))
            K = np.power(self.kernel(1, 0, self.sigma), K)

        else:
            # Pre-compute the Kernel Matrix
            # The following can be slow due to lack of vectorization
            print(f'Pre-computing {self.kernel.__name__} matrix')
            K = np.zeros((m, m))

            for i in range(m):
                for j in range(m):
                    x1 = np.transpose(X[i, :])
                    x2 = np.transpose(X[j, :])
                    K[i, j] = self.kernel(x1, x2)
                    K[i, j] = K[j, i]
