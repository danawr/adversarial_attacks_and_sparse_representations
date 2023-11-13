import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def get_attack_delta_closed_form(A, eps):
    print('Started SVD.')
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    print('Finished SVD.')
    return eps * u[:, -1]


def get_classifier_attack_delta(s, vh, A, cons_factor, eps, W, c_src, c_tgt, norm='2'):
    A_col_norms = np.linalg.norm(A, axis=0)
    A = A / A_col_norms
    attack_direction = vh @ (W[c_tgt, :] - W[c_src, :])
    curv_squared = np.diagonal(np.diag(s) * cons_factor)
    C = np.diag(curv_squared)

    def f(x):
        return - (x.T @ attack_direction)

    def J(x):
        return - attack_direction

    def H(x):
       return np.zeros((x.shape[0], x.shape[0]))

    def cons_f(x):
        if norm == '2':
            return curv_squared.T @ (x ** 2)
        elif norm == 'inf':
            return (np.max(abs(x)) - eps)**2
        else:
            print(f'Constraint norm should be 2 or inf, but got {norm}')

    def cons_J(x):
        if norm == '2':
            return 2 * curv_squared * x
        elif norm == 'inf':
            j = np.zeros_like(x)
            j[np.argmax(abs(x))] = np.sign(x[np.argmax(abs(x))])
            return 2 * (np.max(abs(x)) - eps) * j

    def cons_H(x, v):
        if norm == '2':
            return 2 * v * C
        elif norm == 'inf':
            H = np.zeros((x.shape[0], x.shape[0]))
            H[np.argmax(abs(x)), np.argmax(abs(x))] = 2 * np.max(abs(x)) * np.sign(x[np.argmax(abs(x))])
            return H

    nonlinear_constraint = NonlinearConstraint(cons_f, 0, eps**2, jac=cons_J, hess=cons_H)
    alpha_0 = np.random.randn(len(s))
    res = minimize(f, alpha_0, method='trust-constr', jac=J, hess=H,
                   constraints=nonlinear_constraint,
                   options={'disp': True})
    print(f'res.x norm: {np.linalg.norm(res.x)}, cons: {cons_f(res.x)}')
    z = (vh.T @ res.x) * A_col_norms
    return A @ z, z


def calc_save_dict_decomposition(A, name=''):
    np.save(f'./A{name}.npy', A)
    print('Started SVD.')
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    print('Finished SVD.')
    np.save(f'./s{name}.npy', s)
    np.save(f'./vh{name}.npy', vh)
    np.save(f'./u{name}.npy', u)

    A_col_norms = np.linalg.norm(A, axis=0)
    mult = A_col_norms * vh
    cons_factor = vh @ mult.T
    np.save(f'./cons_factor{name}.npy', cons_factor)
    return u, s, vh, cons_factor


def load_dict_decomposition(name=''):
    A = np.load(f'./A{name}.npy')
    s = np.load(f'./s{name}.npy')
    vh = np.load(f'./vh{name}.npy')
    u = np.load(f'./u{name}.npy')
    cons_factor = np.load(f'./cons_factor{name}.npy')
    return A, u, s, vh, cons_factor
