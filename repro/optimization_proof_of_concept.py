from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, shgo, direct, basinhopping, differential_evolution, dual_annealing, brute

from gasp.simulation import simulate_ssfp, simulate_gasp, view_gasp_input, view_gasp_comparison, train_gasp, view_gasp
from gasp import responses


def cost(x, _K: int, _desired_fun, _width: int, _height: int, _gradient: float, _phantom_type: str) -> float:
    M = []
    for ii in range(_K):
        alpha = x[ii*3]
        TR = x[ii*3+1]
        PC = float(x[ii*3+2])
        # print(np.rad2deg(alpha), TR*1000, np.rad2deg(PC))
        M.append(simulate_ssfp(
            width=_width,
            height=_height,
            npcs=PC,
            TRs=(TR,),
            alpha=alpha,
            gradient=_gradient,
            phantom_type=_phantom_type,
            minTR=min(x[1::3]),
        ))

        # plt.imshow(np.abs(M[-1].squeeze()))
        # plt.title(f"{ii} PC")
        # plt.show()

    M = np.concatenate(M, axis=-1)
    # print(M.shape)
    # view_gasp_input(M=M)
    Ic, An = train_gasp(M=M, D=_desired_fun)

    # view_gasp(Ic, D=_desired_fun)
    # plt.show()
    # assert False

    # plt.plot(desired_fun, "k-")
    # plt.plot(Ic[_width // 2, :], "--")
    # plt.show()
    res = np.linalg.norm(desired_fun - Ic[_width // 2, :])**2
    print(res)
    #print(res, np.rad2deg(x[0::3]), 1000*x[1::3], np.rad2deg(x[2::3]))
    return res


if __name__ == "__main__":
    # x: [alpha, TR, PC]*K
    K = 48  # total number of phase-cycles
    width = 256
    height = 256
    desired_fun = responses.square(width, bw=0.3, shift=0.25)
    gradient = 2.0 * np.pi

    x0 = np.empty(3*K, dtype=float)

    x0[0::3] = np.deg2rad(60)

    x0[1::3][:K//3] = 5e-3
    x0[1::3][K//3:2*K//3] = 10e-3
    x0[1::3][2*K//3:] = 12e-3

    x0[2::3][:K//3] = np.linspace(0, 2*np.pi, K//3, endpoint=False)
    x0[2::3][K//3:2*K//3] = np.linspace(0, 2 * np.pi, K//3, endpoint=False)
    x0[2::3][2*K//3:] = np.linspace(0, 2 * np.pi, K//3, endpoint=False)

    print(np.reshape(x0, (K, 3)))

    bnds = []
    for ii in range(K):
        bnds.append((np.deg2rad(3), np.deg2rad(100)))
        bnds.append((3e-3, 30e-3))
        bnds.append((0, 2*np.pi))

    cost_fun = partial(cost, _K=K, _desired_fun=desired_fun, _width=width, _height=height, _gradient=gradient, _phantom_type="circle")

    print(f"FIRST GUESS IS {cost_fun(x0)}")
    assert False

    # res = minimize(
    #     fun=cost_fun,
    #     x0=x0,
    #     bounds=bnds,
    #     options={"disp": True},
    #     #method="TNC",
    # )

    # # haven't seen first iter yet...
    # def clbk(xk):
    #     print(cost_fun(xk))
    #
    # res = shgo(
    #     func=cost_fun,
    #     bounds=bnds,
    #     options={"disp": True},
    #     # workers=8,
    #     callback=clbk,
    # )

    # # seems best?
    # def clbk(xk):
    #     print(cost_fun(xk))
    #
    # res = direct(
    #     func=cost_fun,
    #     bounds=bnds,
    #     callback=clbk,
    # )

    # not significant progress
    res = basinhopping(
        func=cost_fun,
        x0=x0,
        disp=True,
    )

    # # Good progress, lots of iters possible!
    # def clbk(xk, convergence):
    #     print(xk)
    #
    # res = differential_evolution(
    #     func=cost_fun,
    #     x0=x0,
    #     bounds=bnds,
    #     workers=10,
    #     disp=True,
    #     callback=clbk,
    # )

    # # good progress, then plateaus
    # def clbk(x, f, context):
    #     print(f)
    #
    # res = dual_annealing(
    #     func=cost_fun,
    #     x0=x0,
    #     bounds=bnds,
    #     callback=clbk,
    # )

    print(res)
