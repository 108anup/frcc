# %%
import sympy as sp
from IPython.display import display

sp.init_printing(use_unicode=True)

# FRCC state update equations on ideal link with different assumptions about collisions.

USE_MAIN_CWND_UPDATE = True  # False corresponds to the alternate cwnd update
N_G = 2
C, R, theta = sp.symbols("C R theta", domain=sp.S.Reals, positive=True)
N_C = [sp.symbols(f"N_C_{i}", domain=sp.S.Reals, positive=True) for i in range(1, N_G)]
N_T = sp.symbols("N_T", domain=sp.S.Reals, positive=True)
# ^^ on dumbbell topology with ideal links, all flows have the same N_T measurement.
alpha = sp.symbols("alpha", domain=sp.S.Reals, positive=True)
alpha = 1

# Before probing. Assume all flows measure the same C, N_T.
RTT = N_T * theta + R
cwnd = [C * RTT / N_C[i] for i in range(N_G - 1)]
sum_all_but_last = sum(cwnd)
cwnd.append(C * RTT - sum_all_but_last)
N_C.append(C * RTT / cwnd[-1])

assert len(cwnd) == N_G
assert len(N_C) == N_G


# %%
def display_all(next_N_C, next_N_T, N_C, N_T):
    display("N_C after one round:\n   ", sp.simplify(next_N_C[0]))
    display("\nN_T after one round:\n   ", sp.simplify(next_N_T))
    display(
        "\n|N_C - N_G| after one round relative to before:\n   ",
        sp.simplify((next_N_C[0] - N_G) / (N_C[0] - N_G)),
    )
    display(
        "\n|N_T - N_G| after one round relative to before:\n   ",
        sp.simplify((next_N_T - N_G) / (N_T - N_G)),
    )


def synchronous():
    display("-" * 80)
    display("Synchronous")
    display("-" * 80)

    def update(cwnd, N_C, N_T, alpha):
        next_cwnd = []
        for i in range(N_G):
            if USE_MAIN_CWND_UPDATE:
                target_cwnd_scaling = (R + theta * N_C[i]) / (R + theta * N_T)
            else:
                target_cwnd_scaling = (N_C[i]) / (N_T)
            next_cwnd.append(cwnd[i] * (1 - alpha + alpha * target_cwnd_scaling))

        next_RTT = sum(next_cwnd) / C
        next_N_C = [C * next_RTT / next_cwnd[i] for i in range(N_G)]
        next_N_T = (next_RTT - R) / theta

        return next_cwnd, next_N_C, next_N_T

    next_cwnd, next_N_C, next_N_T = update(cwnd, N_C, N_T, alpha)
    display_all(next_N_C, next_N_T, N_C, N_T)
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


def sequential():
    assert N_G == 2

    display("-" * 80)
    display("Sequential")
    display("-" * 80)

    def update(cwnd, N_C, N_T, alpha, update_list=[]):
        next_cwnd = cwnd.copy()
        for i in update_list:
            if USE_MAIN_CWND_UPDATE:
                target_cwnd_scaling = (R + theta * N_C[i]) / (R + theta * N_T)
            else:
                target_cwnd_scaling = (N_C[i]) / (N_T)
            next_cwnd[i] = cwnd[i] * (1 - alpha + alpha * target_cwnd_scaling)

        next_RTT = sum(next_cwnd) / C
        next_N_C = [C * next_RTT / next_cwnd[i] for i in range(N_G)]
        next_N_T = (next_RTT - R) / theta

        return next_cwnd, next_N_C, next_N_T

    next_cwnd, next_N_C, next_N_T = update(cwnd, N_C, N_T, alpha, [0])
    next_cwnd, next_N_C, next_N_T = update(
        next_cwnd, next_N_C, next_N_T, alpha, [1]
    )
    display_all(next_N_C, next_N_T, N_C, N_T)
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


def full_overlap():
    assert N_G == 2

    display("-" * 80)
    display("Full overlap")
    display("-" * 80)

    def update(cwnd, N_C, N_T, alpha):
        """
        The bandwidths estimates will be:
        Ei = gamma * N_T * rate_i * \\Delta R
        Ei \\propto rate_i (all flows measure same N_T)
        total_delay = (E1 + E2)/C
        C1 = E1/(total_delay)
        C2 = E2/(total_delay)


        C = E1/(E1/C)
        So, C1 = C * E1/(E1 + E2) = C * rate1/(rate1 + rate2)
               = C * (1/N_1) / (1/N1 + 1/N2) = C N2/(N1 + N2)
        """

        next_cwnd = []
        if USE_MAIN_CWND_UPDATE:
            adjusted_flow_count_estimate = N_C[0] * N_C[1] / (N_C[0] + N_C[1])
            target_cwnd_scaling = (R + theta * adjusted_flow_count_estimate) / (
                R + theta * N_T
            )
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))
        else:
            # Both the scaling factors are the same.
            target_cwnd_scaling = (N_C[0]) / (N_T) * (N_C[1] / (N_C[0] + N_C[1]))
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))

            target_cwnd_scaling = (N_C[1]) / (N_T) * (N_C[0] / (N_C[0] + N_C[1]))
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))

        next_RTT = sum(next_cwnd) / C
        next_N_C = [C * next_RTT / next_cwnd[i] for i in range(N_G)]
        next_N_T = (next_RTT - R) / theta

        return next_cwnd, next_N_C, next_N_T

    next_cwnd, next_N_C, next_N_T = update(cwnd, N_C, N_T, alpha)
    display_all(next_N_C, next_N_T, N_C, N_T)
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


def partial_overlap():
    display("-" * 80)
    display("Partial overlap (this flow underestimates C)")
    display("-" * 80)

    def update(cwnd, N_C, N_T, alpha):
        """
        Flow 0 sees sum of delays, but flow 1 only sees its own delay.
        """

        next_cwnd = []
        if USE_MAIN_CWND_UPDATE:
            adjusted_flow_count_estimate = N_C[0] * N_C[1] / (N_C[0] + N_C[1])
            target_cwnd_scaling = (R + theta * adjusted_flow_count_estimate) / (
                R + theta * N_T
            )
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))

            target_cwnd_scaling = (R + theta * N_C[1]) / (R + theta * N_T)
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))
        else:
            target_cwnd_scaling = (N_C[0]) / (N_T) * (N_C[1] / (N_C[0] + N_C[1]))
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))

            target_cwnd_scaling = (N_C[1]) / (N_T)
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))

        next_RTT = sum(next_cwnd) / C
        next_N_C = [C * next_RTT / next_cwnd[i] for i in range(N_G)]
        next_N_T = (next_RTT - R) / theta

        return next_cwnd, next_N_C, next_N_T

    next_cwnd, next_N_C, next_N_T = update(cwnd, N_C, N_T, alpha)
    display_all(next_N_C, next_N_T, N_C, N_T)
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


def partial_overlap_other():
    display("-" * 80)
    display("Partial overlap (other underestimates C)")
    display("-" * 80)

    def update(cwnd, N_C, N_T, alpha):
        """
        Flow 0 sees sum of delays, but flow 1 only sees its own delay.
        """

        next_cwnd = []
        if USE_MAIN_CWND_UPDATE:
            target_cwnd_scaling = (R + theta * N_C[0]) / (R + theta * N_T)
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))

            adjusted_flow_count_estimate = N_C[1] * N_C[0] / (N_C[0] + N_C[1])
            target_cwnd_scaling = (R + theta * adjusted_flow_count_estimate) / (
                R + theta * N_T
            )
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))
        else:
            target_cwnd_scaling = (N_C[0]) / (N_T)
            next_cwnd.append(cwnd[0] * (1 - alpha + alpha * target_cwnd_scaling))

            target_cwnd_scaling = (N_C[1]) / (N_T) * (N_C[0] / (N_C[0] + N_C[1]))
            next_cwnd.append(cwnd[1] * (1 - alpha + alpha * target_cwnd_scaling))

        next_RTT = sum(next_cwnd) / C
        next_N_C = [C * next_RTT / next_cwnd[i] for i in range(N_G)]
        next_N_T = (next_RTT - R) / theta

        return next_cwnd, next_N_C, next_N_T

    next_cwnd, next_N_C, next_N_T = update(cwnd, N_C, N_T, alpha)
    display_all(next_N_C, next_N_T, N_C, N_T)
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


def partial_overlap_average():
    N_C_1, N_T_1 = partial_overlap()
    N_C_2, N_T_2 = partial_overlap_other()
    next_N_C = [(N_C_1[i] + N_C_2[i]) / 2 for i in range(N_G)]
    next_N_T = (N_T_1 + N_T_2) / 2
    next_N_C = [sp.simplify(next_N_C[i]) for i in range(N_G)]
    next_N_T = sp.simplify(next_N_T)
    return next_N_C, next_N_T


# %%
if __name__ == "__main__":
    synchronous()
    sequential()
    full_overlap()
    partial_overlap()
    partial_overlap_other()
