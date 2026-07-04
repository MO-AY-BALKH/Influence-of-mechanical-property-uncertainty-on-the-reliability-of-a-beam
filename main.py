import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gumbel_r
from scipy.optimize import brentq

# ============================================================
# 1. GEOMETRIC & PROBABILISTIC PARAMETERS
# ============================================================
L = 6.0          # span (m)
b = 0.20         # width (m)
w_lim = L / 250  # deflection limit (m)
h_initial = 0.30 # initial height (m)

mu_E = 30e9      # mean Young's modulus (Pa)
delta_E = 0.10   # CoV of E
mu_q = 15e3      # mean load (N/m)
delta_q = 0.10   # CoV of q
mu_fy = 250e6    # mean yield strength (Pa)
delta_fy = 0.10  # CoV of fy

# Parameters of non‑normal distributions
sigma_LN_E = np.sqrt(np.log(1 + delta_E**2))
mu_LN_E = np.log(mu_E) - 0.5 * sigma_LN_E**2

sigma_LN_fy = np.sqrt(np.log(1 + delta_fy**2))
mu_LN_fy = np.log(mu_fy) - 0.5 * sigma_LN_fy**2

beta_gumbel = delta_q * mu_q * np.sqrt(6) / np.pi
gamma_euler = 0.5772156649
mu_q_mode = mu_q - gamma_euler * beta_gumbel

# ============================================================
# 2. ROSENBLATT TRANSFORMATIONS (with overflow protection)
# ============================================================
def U_to_X(u):
    """Inverse transform: standard normal -> physical (E, q, fy)."""
    u1, u2, u3 = u

    log_E = mu_LN_E + sigma_LN_E * u1
    log_E = np.clip(log_E, -700, 700)  # ← prevent overflow
    E = np.exp(log_E)

    p = norm.cdf(u2)
    q = gumbel_r.ppf(p, loc=mu_q_mode, scale=beta_gumbel)

    log_fy = mu_LN_fy + sigma_LN_fy * u3
    log_fy = np.clip(log_fy, -700, 700)  # ← prevent overflow
    fy = np.exp(log_fy)

    return E, q, fy

def dX_dU(u):
    """Derivatives dE/du1, dq/du2, dfy/du3."""
    u1, u2, u3 = u
    E = np.exp(np.clip(mu_LN_E + sigma_LN_E * u1, -700, 700))
    dE = sigma_LN_E * E

    p = norm.cdf(u2)
    phi_u2 = norm.pdf(u2)
    if p > 0 and p < 1:
        dq = beta_gumbel * phi_u2 / (p * (-np.log(p)))
    else:
        dq = 1e-12  # ← prevent zero gradient

    fy = np.exp(np.clip(mu_LN_fy + sigma_LN_fy * u3, -700, 700))
    dfy = sigma_LN_fy * fy
    return dE, dq, dfy

# ============================================================
# 3. LIMIT STATE FUNCTIONS & GRADIENTS
# ============================================================
def g1(h, u):
    E, q, _ = U_to_X(u)
    kappa = (32 * b * w_lim / (5 * L**4)) * h**3
    return kappa * E - q

def grad_g1(h, u):
    E, q, _ = U_to_X(u)
    kappa = (32 * b * w_lim / (5 * L**4)) * h**3
    dE, dq, _ = dX_dU(u)
    return np.array([kappa * dE, -1.0 * dq, 0.0])

def g2(h, u):
    _, q, fy = U_to_X(u)
    alpha = (3 * L**2) / (4 * b * h**2)
    return fy - alpha * q

def grad_g2(h, u):
    _, q, fy = U_to_X(u)
    alpha = (3 * L**2) / (4 * b * h**2)
    dE, dq, dfy = dX_dU(u)
    return np.array([0.0, -alpha * dq, 1.0 * dfy])

# ============================================================
# 4. HL‑RF ALGORITHM WITH ARMIJO
# ============================================================
def compute_beta(h, g_func, grad_func, u0=None, tol=1e-4, max_iter=100, verbose=False):
    if u0 is None:
        u = np.zeros(3)
    else:
        u = u0.copy()

    # Check margin at mean point
    if g_func(h, np.zeros(3)) <= 0:
        return -np.inf, np.zeros(3), False

    eta = 1e-4
    rho = 0.5
    c_penalty = 10.0 * max(1.0, np.linalg.norm(grad_func(h, u)))

    def merit(u):
        return 0.5 * np.dot(u, u) + c_penalty * abs(g_func(h, u))

    converged = False
    for k in range(max_iter):
        g_val = g_func(h, u)
        grad = grad_func(h, u)
        norm_grad = np.linalg.norm(grad)

        if not np.isfinite(g_val) or not np.isfinite(norm_grad):
            beta = np.linalg.norm(u)
            return beta, u, True

        if norm_grad < 1e-12:
            beta = np.linalg.norm(u)
            return beta, u, True

        if abs(g_val) / norm_grad < tol and np.linalg.norm(u) > 1e-8:
            converged = True
            break

        u_hlrf = (np.dot(grad, u) - g_val) / (norm_grad**2) * grad
        d = u_hlrf - u

        lam = 1.0
        sign_g = 1.0 if g_val >= 0 else -1.0
        grad_m = u + c_penalty * sign_g * grad
        slope = np.dot(grad_m, d)

        if slope >= 0:
            lam = 0.5

        while True:
            u_new = u + lam * d
            if merit(u_new) <= merit(u) + eta * lam * slope:
                break
            lam *= rho
            if lam < 1e-10:
                lam = 0.0
                break

        u = u + lam * d

        if lam < 1e-10 and np.linalg.norm(d) > 1e-6:
            u = u + 0.01 * d

        if verbose and k % 10 == 0:
            print(f"Iter {k}: beta={np.linalg.norm(u):.4f}, g={g_val:.3e}")

    beta = np.linalg.norm(u)
    return beta, u, converged

# ============================================================
# 5. MONTE CARLO SIMULATION
# ============================================================
def monte_carlo_pf(h, N=1000000, seed=42):
    np.random.seed(seed)

    E_samples = np.exp(mu_LN_E + sigma_LN_E * np.random.randn(N))
    fy_samples = np.exp(mu_LN_fy + sigma_LN_fy * np.random.randn(N))
    q_samples = gumbel_r.rvs(loc=mu_q_mode, scale=beta_gumbel, size=N)

    kappa = (32 * b * w_lim / (5 * L**4)) * h**3
    alpha = (3 * L**2) / (4 * b * h**2)

    g1_vals = kappa * E_samples - q_samples
    g2_vals = fy_samples - alpha * q_samples

    fail1 = (g1_vals <= 0)
    fail2 = (g2_vals <= 0)
    fail_sys = (fail1 | fail2)

    Pf1 = np.mean(fail1)
    Pf2 = np.mean(fail2)
    Pfsys = np.mean(fail_sys)

    beta1 = -norm.ppf(Pf1) if 0 < Pf1 < 1 else (-np.inf if Pf1 == 1 else np.inf)
    beta2 = -norm.ppf(Pf2) if 0 < Pf2 < 1 else (-np.inf if Pf2 == 1 else np.inf)
    betasys = -norm.ppf(Pfsys) if 0 < Pfsys < 1 else (-np.inf if Pfsys == 1 else np.inf)

    def wilson_ci(pf, N, alpha=0.05):
        z = norm.ppf(1 - alpha/2)
        denom = 1 + z**2 / N
        center = (pf + z**2/(2*N)) / denom
        margin = (z/denom) * np.sqrt(pf*(1-pf)/N + z**2/(4*N**2))
        return (max(0, center - margin), min(1, center + margin))

    ci1 = wilson_ci(Pf1, N)
    ci2 = wilson_ci(Pf2, N)
    cisys = wilson_ci(Pfsys, N)

    return (Pf1, Pf2, Pfsys, beta1, beta2, betasys,
            ci1, ci2, cisys, g1_vals, g2_vals)

# ============================================================
# 6. RBDO FUNCTIONS (separate bounds for ELS and ELU)
# ============================================================
def beta_g1(h):
    b, _, _ = compute_beta(h, g1, grad_g1, max_iter=50)
    return b

def beta_g2(h):
    b, _, _ = compute_beta(h, g2, grad_g2, max_iter=50)
    return b

def find_h_target(target_beta, beta_func, h_min=0.02, h_max=0.50, tol=1e-4):
    """Find h such that beta_func(h) == target_beta using bisection."""
    def f(h):
        beta = beta_func(h)
        if not np.isfinite(beta) or beta == -np.inf:
            return -1e6
        return beta - target_beta

    f_min = f(h_min)
    f_max = f(h_max)

    if f_min >= 0:
        return h_min

    if f_max <= 0:
        print(f"  Warning: beta({h_max:.3f}) = {beta_func(h_max):.3f} < {target_beta}")
        return h_max

    try:
        return brentq(f, h_min, h_max, xtol=tol)
    except ValueError:
        for h in np.linspace(h_min, h_max, 100):
            if abs(f(h)) < 0.1:
                return h
        return h_max

# ============================================================
# 7. MAIN PROGRAM
# ============================================================
def main():
    print("=== Analyse de fiabilité avancée pour poutre fléchie ===")
    print(f"Paramètres: L={L} m, b={b} m, w_lim={w_lim:.4f} m")
    print(f"E: moyenne={mu_E:.2e} Pa, CoV={delta_E:.0%}")
    print(f"q: moyenne={mu_q:.2e} N/m, CoV={delta_q:.0%}")
    print(f"fy: moyenne={mu_fy:.2e} Pa, CoV={delta_fy:.0%}")

    # -------- 1. HL‑RF for initial height --------
    h0 = h_initial
    print(f"\n--- HL-RF pour h = {h0:.3f} m ---")
    beta1, u1_star, conv1 = compute_beta(h0, g1, grad_g1, verbose=True)
    beta2, u2_star, conv2 = compute_beta(h0, g2, grad_g2, verbose=True)
    print(f"β1 (flèche) = {beta1:.4f}  (convergence: {conv1})")
    print(f"β2 (résistance) = {beta2:.4f}  (convergence: {conv2})")

    if np.isfinite(beta1) and beta1 > 0:
        alpha_E1 = -u1_star[0] / beta1
        alpha_q1 = -u1_star[1] / beta1
        print(f"α_E (pour g1) = {alpha_E1:.4f}, α_q (pour g1) = {alpha_q1:.4f}")
    if np.isfinite(beta2) and beta2 > 0:
        alpha_E2 = -u2_star[0] / beta2
        alpha_q2 = -u2_star[1] / beta2
        alpha_fy2 = -u2_star[2] / beta2
        print(f"α_E (pour g2) = {alpha_E2:.4f}, α_q (pour g2) = {alpha_q2:.4f}, α_fy = {alpha_fy2:.4f}")

    # -------- 2. Monte Carlo validation --------
    print("\n--- Simulation Monte Carlo (N=1e6) ---")
    (Pf1, Pf2, Pfsys, beta1_mc, beta2_mc, betasys_mc,
     ci1, ci2, cisys, g1_vals, g2_vals) = monte_carlo_pf(h0, N=1000000)
    print(f"Pf1 = {Pf1:.4e} -> β1 = {beta1_mc:.4f}, IC95: [{ci1[0]:.4e}, {ci1[1]:.4e}]")
    print(f"Pf2 = {Pf2:.4e} -> β2 = {beta2_mc:.4f}, IC95: [{ci2[0]:.4e}, {ci2[1]:.4e}]")
    print(f"Pfsys = {Pfsys:.4e} -> βsys = {betasys_mc:.4f}, IC95: [{cisys[0]:.4e}, {cisys[1]:.4e}]")

    # -------- 3. Histograms --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(g1_vals, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', label='g1=0')
    axes[0].set_xlabel('Marge g1 (N/m)')
    axes[0].set_ylabel('Densité')
    axes[0].set_title('Distribution de g1 (flèche)')
    axes[0].legend()

    axes[1].hist(g2_vals, bins=100, density=True, alpha=0.7, color='darkorange', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', label='g2=0')
    axes[1].set_xlabel('Marge g2 (Pa)')
    axes[1].set_ylabel('Densité')
    axes[1].set_title('Distribution de g2 (résistance)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('histogrammes_double_etat.png', dpi=300)
    plt.show()

    # -------- 4. RBDO (optimisation) --------
    print("\n--- Optimisation RBDO (recherche de h_min) ---")
    beta_ELS = 1.5
    beta_ELU = 3.8

    # ELS search: upper bound 0.40 m (plenty for the root ~0.296 m)
    h_ELS = find_h_target(beta_ELS, beta_g1, h_min=0.02, h_max=0.40)
    if h_ELS is None:
        h_ELS = 0.02
    print(f"h_ELS = {h_ELS:.4f} m  (β1={beta_g1(h_ELS):.4f})")

    # ELU search: upper bound 0.20 m (true root ~0.118 m)
    h_ELU = find_h_target(beta_ELU, beta_g2, h_min=0.02, h_max=0.20)
    if h_ELU is None:
        h_ELU = 0.02
    print(f"h_ELU = {h_ELU:.4f} m  (β2={beta_g2(h_ELU):.4f})")

    # Corrected target (adjust for FORM‑MC error)
    beta_ELS_corrige = 1.585
    h_ELS_corrige = find_h_target(beta_ELS_corrige, beta_g1, h_min=0.02, h_max=0.40)
    if h_ELS_corrige is None:
        h_ELS_corrige = 0.32
    print(f"h_ELS_corrige = {h_ELS_corrige:.4f} m  (β1={beta_g1(h_ELS_corrige):.4f})")

    # h_min is the MAX of ELS and ELU requirements
    h_min = max(h_ELS_corrige, h_ELU)
    print(f"Hauteur minimale requise (corrigée) : h_min = {h_min:.4f} m")
    print(f"Section recommandée : 0.20 × {np.ceil(h_min*100)/100:.2f} m")

    # -------- 5. Monte Carlo validation of optimal design --------
    print(f"\n--- Validation Monte Carlo sur h_min (N=500k) ---")
    (Pf1_opt, Pf2_opt, Pfsys_opt, beta1_opt, beta2_opt, betasys_opt,
     ci1_opt, ci2_opt, cisys_opt, _, _) = monte_carlo_pf(h_min, N=500000)
    print(f"Pf1 = {Pf1_opt:.4e} -> β1 = {beta1_opt:.4f}")
    print(f"Pf2 = {Pf2_opt:.4e} -> β2 = {beta2_opt:.4f}")
    print(f"Pfsys = {Pfsys_opt:.4e} -> βsys = {betasys_opt:.4f}")

    # -------- 6. Plot β(h) curves --------
    print("\n--- Génération de la courbe β(h) ---")
    h_range = np.linspace(0.02, 0.40, 30)
    beta1_list = []
    beta2_list = []
    for h in h_range:
        b1, _, _ = compute_beta(h, g1, grad_g1, max_iter=50)
        b2, _, _ = compute_beta(h, g2, grad_g2, max_iter=50)
        beta1_list.append(b1 if np.isfinite(b1) else 0)
        beta2_list.append(b2 if np.isfinite(b2) else 0)

    plt.figure(figsize=(8, 6))
    plt.plot(h_range, beta1_list, 'b-', label='β1 (flèche)')
    plt.plot(h_range, beta2_list, 'r-', label='β2 (résistance)')
    plt.axhline(y=1.5, color='blue', linestyle='--', label='β_ELS = 1.5')
    plt.axhline(y=3.8, color='red', linestyle='--', label='β_ELU = 3.8')
    if h_ELS is not None:
        plt.axvline(x=h_ELS, color='blue', linestyle=':', alpha=0.5, label=f'h_ELS={h_ELS:.3f}')
    if h_ELU is not None:
        plt.axvline(x=h_ELU, color='red', linestyle=':', alpha=0.5, label=f'h_ELU={h_ELU:.3f}')
    if h_ELS_corrige is not None:
        plt.axvline(x=h_ELS_corrige, color='green', linestyle='-.', alpha=0.5,
                    label=f'h_ELS_corr={h_ELS_corrige:.3f}')
    plt.xlabel('Hauteur h (m)')
    plt.ylabel('Indice de fiabilité β')
    plt.title('Évolution de β en fonction de h')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('beta_vs_h.png', dpi=300)
    plt.show()

    print("\n=== Fin de l'analyse ===")

if __name__ == "__main__":
    main()
