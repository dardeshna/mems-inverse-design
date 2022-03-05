import numpy as np
import scipy
import scipy.linalg

def N(xi, eta, zeta):

    t2 = eta+1.0
    t3 = eta**2
    t4 = xi**2
    t5 = zeta+1.0
    t6 = zeta**2
    t7 = eta-1.0
    t8 = -xi
    t9 = -zeta
    t10 = zeta-1.0
    t11 = xi/4.0
    t14 = xi/8.0
    t12 = t3-1.0
    t13 = t6-1.0
    t15 = t4/4.0
    t16 = t11+1.0/4.0
    t17 = t11-1.0/4.0
    t18 = t14+1.0/8.0
    t19 = t14-1.0/8.0
    t20 = t15-1.0/4.0

    return np.array([t7*t10*t19*(t2+t5+xi),-t7*t10*t18*(t2+t5+t8),-t2*t10*t18*(-t5+t7+xi),-t2*t10*t19*(t5-t7+xi),-t5*t7*t19*(t2+t9+xi+1.0),t5*t7*t18*(t2+t8+t9+1.0),t2*t5*t18*(t7+t10+xi),-t2*t5*t19*(t7+t8+t10),-t7*t10*t20,t10*t12*t16,t2*t10*t20,-t10*t12*t17,t5*t7*t20,-t5*t12*t16,-t2*t5*t20,t5*t12*t17,-t7*t13*t17,t7*t13*t16,-t2*t13*t16,t2*t13*t17])

def grad_N(xi, eta, zeta):

    t2 = eta+1.0
    t3 = eta**2
    t4 = xi**2
    t5 = zeta+1.0
    t6 = zeta**2
    t7 = -eta
    t8 = eta-1.0
    t9 = -xi
    t10 = -zeta
    t11 = zeta-1.0
    t13 = xi/4.0
    t17 = xi/8.0
    t12 = t2+t5+xi
    t14 = t3-1.0
    t15 = t6-1.0
    t16 = t8+t11+xi
    t18 = t4/4.0
    t19 = t2+t10+xi+1.0
    t20 = t2+t5+t9
    t21 = t5+t7+xi+1.0
    t22 = -t5+t8+xi
    t23 = t8+t9+t11
    t24 = t13+1.0/4.0
    t25 = t13-1.0/4.0
    t26 = t17+1.0/8.0
    t27 = t2+t9+t10+1.0
    t28 = t17-1.0/8.0
    t29 = t18-1.0/4.0
    t30 = (t5*t14)/4.0
    t31 = (t2*t15)/4.0
    t32 = (t11*t14)/4.0
    t33 = (t8*t15)/4.0
    t34 = t14*t24
    t36 = t15*t24
    t38 = t14*t25
    t40 = t15*t25
    t42 = t2*t5*t26
    t43 = t2*t5*t28
    t44 = t5*t8*t26
    t45 = t2*t11*t26
    t46 = t5*t8*t28
    t47 = t2*t11*t28
    t48 = t8*t11*t26
    t49 = t8*t11*t28
    t35 = t2*t29
    t37 = t5*t29
    t39 = t8*t29
    t41 = t11*t29
    t50 = -t43
    t51 = -t44
    t52 = -t45
    t53 = -t46
    t54 = -t47
    t55 = -t48
    return np.reshape([t49+(t8*t11*t12)/8.0,t48-(t8*t11*t20)/8.0,t52-(t2*t11*t22)/8.0,t54-(t2*t11*t21)/8.0,t53-(t5*t8*t19)/8.0,t51+(t5*t8*t27)/8.0,t42+(t2*t5*t16)/8.0,t43-(t2*t5*t23)/8.0,t8*t11*xi*(-1.0/2.0),t32,(t2*t11*xi)/2.0,-t32,(t5*t8*xi)/2.0,-t30,t2*t5*xi*(-1.0/2.0),t30,-t33,t33,-t31,t31,t49+t11*t12*t28,t55-t11*t20*t26,t52-t11*t22*t26,t47-t11*t21*t28,t53-t5*t19*t28,t44+t5*t26*t27,t42+t5*t16*t26,t50-t5*t23*t28,-t41,eta*t11*t24*2.0,t41,eta*t11*t25*-2.0,t37,eta*t5*t24*-2.0,-t37,eta*t5*t25*2.0,-t40,t36,-t36,t40,t49+t8*t12*t28,t55-t8*t20*t26,t45-t2*t22*t26,t54-t2*t21*t28,t46-t8*t19*t28,t51+t8*t26*t27,t42+t2*t16*t26,t50-t2*t23*t28,-t39,t34,t35,-t38,t39,-t34,-t35,t38,t8*t25*zeta*-2.0,t8*t24*zeta*2.0,t2*t24*zeta*-2.0,t2*t25*zeta*2.0],(3,20)).T

def gauss_quad_brick_3(integrand, sz):

    vals = [-np.sqrt(0.6), 0, np.sqrt(0.6)]
    weights = [5/9, 8/9, 5/9]
    
    res = np.zeros(sz)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                res += integrand(vals[i], vals[j], vals[k]) * weights[i] * weights[j] * weights[k]

    return res

def get_C(E, nu):

    return E / ((1+nu) * (1-2*nu)) * np.array([
        [1-nu, nu, nu, 0, 0, 0],
        [nu, 1-nu, nu, 0, 0, 0],
        [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0],
        [0, 0, 0, 0, (1-2*nu)/2, 0],
        [0, 0, 0, 0, 0, (1-2*nu)/2],
    ])

_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
        ])

def elem_K(nodes, E, nu):

    C = get_C(E, nu)
    
    def integrand(xi, eta, zeta):

        J = nodes.T @ grad_N(xi, eta, zeta)
        
        block = np.linalg.inv(J).T @ grad_N(xi, eta, zeta).T
        B = _b @ scipy.linalg.block_diag(*[block,]*3).reshape(-1, 3, 20).transpose().reshape(-1,9).transpose()

        return B.T @ C @ B * np.linalg.det(J)

    return gauss_quad_brick_3(integrand, (60,60))

def elem_M(nodes, rho):

    def integrand(xi, eta, zeta):

        J = nodes.T @ grad_N(xi, eta, zeta)

        n = N(xi, eta, zeta)

        return np.outer(n * np.linalg.det(J), n)
    
    res = rho * gauss_quad_brick_3(integrand, (20,20))

    out = np.empty((60,60))
    for i in range(20):
        for j in range(20):
            out[3*i:3*i+3, 3*j:3*j+3] = res[i,j] * np.eye(3)
    return out