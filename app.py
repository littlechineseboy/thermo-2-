"""
=============================================================
 Heat Equation Solver — Streamlit App
=============================================================
 Run with:  streamlit run app.py
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D  # noqa
import time
import streamlit as st

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heat Equation Solver",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 2D Heat Equation Solver")
st.markdown(
    "Numerical resolution of the steady-state heat equation using "
    "**LU decomposition** and **Gauss-Seidel** iteration. "
    "Adjust the parameters in the sidebar and explore the results."
)
st.divider()

# ─────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def build_system(N, T_top=100, T_bottom=0, T_left=0, T_right=0):
    A = np.zeros((N * N, N * N))
    b = np.zeros(N * N)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            if i == 0:
                A[k, k] = 1; b[k] = T_top
            elif i == N - 1:
                A[k, k] = 1; b[k] = T_bottom
            elif j == 0:
                A[k, k] = 1; b[k] = T_left
            elif j == N - 1:
                A[k, k] = 1; b[k] = T_right
            else:
                A[k, k]     = -4
                A[k, k - 1] =  1
                A[k, k + 1] =  1
                A[k, k - N] =  1
                A[k, k + N] =  1
    return A, b


def direct_solver(A, b):
    return np.linalg.solve(A, b)


def gauss_seidel(A, b, tol=1e-6, max_iter=2000):
    n = len(b)
    x = np.zeros(n)
    residuals = []
    iters_done = 0
    for it in range(max_iter):
        for i in range(n):
            s1 = np.dot(A[i, :i],     x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        r = np.linalg.norm(A @ x - b)
        residuals.append(r)
        iters_done = it + 1
        if r < tol:
            break
    return x, residuals, iters_done


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — PARAMETERS
# ─────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Parameters")

N = st.sidebar.slider("Grid size N (N×N)", min_value=5, max_value=30, value=15, step=1)

st.sidebar.subheader("🌡️ Boundary Temperatures (°C)")
T_top    = st.sidebar.slider("Top edge",    0, 200, 100, step=10)
T_bottom = st.sidebar.slider("Bottom edge", 0, 200,   0, step=10)
T_left   = st.sidebar.slider("Left edge",   0, 200,   0, step=10)
T_right  = st.sidebar.slider("Right edge",  0, 200,   0, step=10)

st.sidebar.subheader("🔧 Gauss-Seidel Settings")
tol      = st.sidebar.select_slider("Tolerance", options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], value=1e-6)
max_iter = st.sidebar.slider("Max iterations", 100, 5000, 2000, step=100)

colormap = st.sidebar.selectbox("Colormap", ["hot", "RdBu_r", "plasma", "viridis", "inferno", "coolwarm"], index=0)

run = st.sidebar.button("▶ Solve", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  SOLVE ON BUTTON PRESS
# ─────────────────────────────────────────────────────────────

if run:
    with st.spinner("Assembling system and solving..."):

        A, b = build_system(N, T_top, T_bottom, T_left, T_right)

        t0 = time.time()
        x_direct = direct_solver(A, b)
        t_direct = (time.time() - t0) * 1000

        t0 = time.time()
        x_gs, residuals, n_iters = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
        t_gs = (time.time() - t0) * 1000

        T_grid_direct = x_direct.reshape(N, N)
        T_grid_gs     = x_gs.reshape(N, N)
        diff_grid     = np.abs(x_direct - x_gs).reshape(N, N)

        error    = np.max(np.abs(x_direct - x_gs))
        rel_err  = error / (np.abs(x_direct).max() + 1e-12)
        converged = residuals[-1] < tol

    # ── METRICS ──
    st.subheader("📊 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Grid size",      f"{N}×{N} = {N*N} unknowns")
    col2.metric("Direct (LU)",    f"{t_direct:.2f} ms")
    col3.metric("Gauss-Seidel",   f"{t_gs:.2f} ms",   f"{n_iters} iters")
    col4.metric("Max error |GS−LU|", f"{error:.2e}")
    col5.metric("GS converged",   "✅ Yes" if converged else "⚠️ No (max iters)")

    st.divider()

    # ── TAB LAYOUT ──
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Heatmaps", "📈 Convergence", "🧊 3D Surface", "🔬 Error Map"])

    # ── TAB 1 : HEATMAPS ──
    with tab1:
        st.subheader("Temperature Field — LU vs Gauss-Seidel")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"N={N}  |  T: top={T_top}° / bottom={T_bottom}° / left={T_left}° / right={T_right}°", fontsize=12)

        for ax, grid, title in zip(axes,
                                   [T_grid_direct, T_grid_gs],
                                   ["Direct (LU)", f"Gauss-Seidel ({n_iters} iters)"]):
            im = ax.imshow(grid, cmap=colormap, origin='upper')
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, label="T (°C)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 2 : CONVERGENCE ──
    with tab2:
        st.subheader("Gauss-Seidel Convergence")
        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(residuals, color='steelblue', linewidth=2)
            ax.axhline(tol, color='tomato', linestyle='--', linewidth=1.5, label=f'tol = {tol:.0e}')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Residual ‖Ax − b‖₂  (log scale)")
            ax.set_title(f"Convergence — {n_iters} iterations to reach tol={tol:.0e}")
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("**Convergence details**")
            st.markdown(f"- Initial residual: `{residuals[0]:.2e}`")
            st.markdown(f"- Final residual: `{residuals[-1]:.2e}`")
            st.markdown(f"- Iterations: `{n_iters}`")
            st.markdown(f"- Converged: {'✅' if converged else '⚠️ reached max_iter'}")
            st.markdown("")
            st.markdown("**Why log-linear?**")
            st.markdown(
                "Gauss-Seidel converges geometrically: "
                "‖rₖ‖ ≈ ρᵏ · ‖r₀‖ where ρ ≈ cos²(π/N). "
                "This produces a straight line on a log scale. "
                f"For N={N}: ρ ≈ {np.cos(np.pi/N)**2:.4f}"
            )

    # ── TAB 3 : 3D SURFACE ──
    with tab3:
        st.subheader("3D Surface — LU Solution")
        x_ax = np.linspace(0, 1, N)
        y_ax = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x_ax, y_ax)

        fig = plt.figure(figsize=(10, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax3d.plot_surface(X, Y, T_grid_direct, cmap=colormap, edgecolor='none', alpha=0.92)
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("T (°)")
        ax3d.set_title("3D Surface")
        fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)

        ax2d = fig.add_subplot(1, 2, 2)
        img = ax2d.imshow(T_grid_direct, cmap=colormap, origin='upper')
        ax2d.set_title("2D Heatmap (reference)")
        ax2d.axis('off')
        fig.colorbar(img, ax=ax2d, shrink=0.8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── TAB 4 : ERROR MAP ──
    with tab4:
        st.subheader("Pointwise Error |LU − GS|")
        col_c, col_d = st.columns([2, 1])

        with col_c:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(diff_grid, cmap='YlOrRd', origin='upper')
            ax.set_title(f"|LU − GS|  (max = {error:.2e})")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_d:
            st.markdown("**Error analysis**")
            st.markdown(f"- Max absolute error: `{error:.2e}`")
            st.markdown(f"- Relative error: `{rel_err:.2e}`")
            st.markdown(f"- Tolerance used: `{tol:.0e}`")
            st.markdown("")
            st.markdown(
                "The error is highest near boundary edges where "
                "large temperature gradients make convergence slower. "
                "Reducing tolerance or increasing max_iter reduces this."
            )

else:
    # ── WELCOME STATE ──
    st.info("👈 Set your parameters in the sidebar and press **▶ Solve** to run the simulation.")

    st.subheader("📐 How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Discretization")
        st.markdown(
            "The continuous plate is replaced by an N×N grid. "
            "The Laplace equation ∇²T = 0 is approximated using "
            "finite differences, giving one equation per interior node:"
        )
        st.code("T[i-1,j] + T[i+1,j] +\nT[i,j-1] + T[i,j+1] - 4·T[i,j] = 0")

    with col2:
        st.markdown("### 2️⃣ Linear System")
        st.markdown(
            "All N² equations are stacked into a sparse linear system **Ax = b**, "
            "where **x** contains the unknown temperatures and "
            "**b** encodes the boundary conditions."
        )
        st.code("A  : N²×N² sparse matrix\nx  : unknown temperatures\nb  : boundary conditions")

    with col3:
        st.markdown("### 3️⃣ Two Solvers")
        st.markdown(
            "**LU decomposition** solves exactly via A = P·L·U — fast for small N. "
            "**Gauss-Seidel** iterates from x⁰ = 0 until the residual ‖Ax−b‖ < tol — "
            "better for large N."
        )
        st.code("LU  : O(N⁶)  — exact\nGS  : O(N²)/iter — approx")

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────

st.divider()
st.caption("Heat Equation Solver · LMASD Numerical Python Project · Built with NumPy, Matplotlib & Streamlit")
