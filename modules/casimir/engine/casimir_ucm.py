# =============================================================================
# UCM-Engine prototype v0.2 — 1D Casimir toy model with medium parameters
#
# Назначение:
#   Минимальный численный стенд в духе UCM-T для задачи Казимира
#   между двумя параллельными пластинами в 1D-аппроксимации,
#   с явным использованием параметров среды и простой UCM-дисперсией.
#
#   Ядро:
#     - Строится 1D-Лапласиан с граничными условиями u(0) = u(L) = 0.
#     - Вычисляются собственные значения λ_n ≈ k_n^2 (числа волновые).
#     - По ним задаются частоты ω_n = ω(k_n; rho, kappa, eps, k0).
#
#   Язык параметров:
#     - L       — расстояние между пластинами.
#     - N       — число внутренних узлов сетки (разрешение по x).
#     - rho     — эффективная плотность среды.
#     - kappa   — эффективный модуль (упругость / сжимаемость).
#     - eps     — амплитуда UCM-поправки к частотам (по умолчанию ~0.3%).
#     - k0      — характерный масштаб по волновому числу k
#                 (задаёт, какие моды сильнее “чувствуют” среду).
#
#   Связь с UCM-T:
#     - Скорость волн определяется как c0 = sqrt(kappa / rho).
#     - Спектр мод между пластинами строится численно через Лапласиан.
#     - В простейшем виде реализована k-зависимая поправка ω(k)
#       как прототип средовой дисперсии.
#
#   Выход:
#     - casimir_energy(L, N, rho, kappa, eps)
#           → нулевая энергия вакуумных мод E(L).
#     - casimir_force(L, N, rho, kappa, eps)
#           → сила Казимира P(L) ≈ -dE/dL (центральная разность).
#     - В блоке __main__ выполняется “скан по L”:
#           * печать таблицы E0, P0 (без поправки) и E1, P1 (с поправкой),
#             а также относительной разницы dP/P0.
#           * построение графика P(L) для классического и UCM-случая.
#
#   Статус:
#     - Учебный / прототипный код для внутреннего использования
#       и отработки архитектуры UCM-Engine на задаче Казимира.
#     - Не является строгой реализацией конкретной журнальной статьи,
#       а иллюстрирует общую цепочку:
#           PDE → спектр k_n → ω(k_n; параметры среды) → E(L), P(L).
#
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt


def build_laplacian_1d(N, L):
    """
    1D Лапласиан с граничными условиями u(0)=u(L)=0
    N — число внутренних узлов
    L — расстояние между пластинами
    """
    h = L / (N + 1)
    main = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    Lap = (1.0 / h**2) * (
        np.diag(main) +
        np.diag(off, k=1) +
        np.diag(off, k=-1)
    )
    return Lap

def omega_ucm(lambdas, c0, eps=3e-3, k0=5.0):
    """
    Чуть более реалистичная игрушечная UCM-дисперсия:
    ω(k) = c0 * k * F(k),
    где F(k) = 1 + eps / (1 + (k/k0)^2).

    eps — амплитуда поправки ( ~0.3% ),
    k0  — характерный масштаб по k: при k << k0 поправка ≈ eps,
           при k >> k0 поправка затухает.
    """
    k = np.sqrt(lambdas)

    F = 1.0 + eps / (1.0 + (k / k0)**2)

    omega = c0 * k * F
    return omega

def casimir_energy(L, N, rho, kappa, eps=3e-3, hbar=1.0):
    """
    Численная энергия вакуумных мод между пластинами на расстоянии L.

    rho   — плотность среды (условная, безразмерная)
    kappa — эффективный модуль (сжимаемость / упругость)
    eps   — амплитуда UCM-поправки
    """
    # скорость волн из UCM-параметров:
    c0 = np.sqrt(kappa / rho)

    # 1D Лапласиан
    Lap = build_laplacian_1d(N, L)

    # собственные значения Лапласиана (≈ k^2)
    lambdas, _ = np.linalg.eigh(Lap)

    # частоты по UCM-дисперсии
    omegas = omega_ucm(lambdas, c0, eps=eps)

    # нулевая энергия: (1/2) * ħ * sum ω_n
    E = 0.5 * hbar * np.sum(omegas)
    return E

def casimir_force(L, N, rho, kappa, eps=3e-3, hbar=1.0, dL=1e-3):
    """
    Приблизительная сила Казимира как -dE/dL (центральная разность).
    Параметры среды задаются через rho и kappa.
    """
    E_plus  = casimir_energy(L + dL, N, rho, kappa, eps=eps, hbar=hbar)
    E_minus = casimir_energy(L - dL, N, rho, kappa, eps=eps, hbar=hbar)

    dE_dL = (E_plus - E_minus) / (2.0 * dL)
    P = -dE_dL
    return P

if __name__ == "__main__":
    # Геометрия и сетка:
    L_values = [0.5, 0.7, 1.0, 1.5, 2.0]  # несколько расстояний
    N = 50

    # Параметры среды:
    rho = 1.0
    kappa = 1.0

    eps_ucm = 3e-3  # амплитуда UCM-поправки

    print("Scan по L для rho =", rho, "kappa =", kappa, "N =", N)
    print("eps =", eps_ucm)
    print()
    print("{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "L", "E0", "P0", "E1", "P1", "dP/P0"
    ))
    print("-" * 70)

    P0_list = []
    P1_list = []

    for L in L_values:
        E0 = casimir_energy(L, N, rho, kappa, eps=0.0)
        P0 = casimir_force(L, N, rho, kappa, eps=0.0)

        E1 = casimir_energy(L, N, rho, kappa, eps=eps_ucm)
        P1 = casimir_force(L, N, rho, kappa, eps=eps_ucm)

        dP_rel = (P1 - P0) / P0 if P0 != 0 else 0.0

        print("{:6.2f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5e}".format(
            L, E0, P0, E1, P1, dP_rel
        ))

        P0_list.append(P0)
        P1_list.append(P1)

    # --- Визуализация силы Казимира ---
    plt.figure()
    plt.plot(L_values, P0_list, marker="o", label="P0 (без поправки)")
    plt.plot(L_values, P1_list, marker="s", linestyle="--", label="P1 (с UCM-поправкой)")

    plt.xlabel("L (расстояние между пластинами)")
    plt.ylabel("P(L) (условные единицы)")
    plt.title("Сила Казимира в 1D-модели UCM-Engine")
    plt.legend()
    plt.grid(True)

    plt.show()
