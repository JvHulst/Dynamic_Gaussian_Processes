# Estimation of Dynamic Gaussian Processes

Code accompanying the paper

> **Estimation of Dynamic Gaussian Processes**
>
> J. van Hulst, R. van Zuijlen, D. Antunes, W.P.M.H. Heemels
>
> *2023 IEEE 62nd Conference on Decision and Control (CDC)*

**Paper available at:**

- <https://heemels.tue.nl/content/papers/HulZui_CDC23a.pdf>
- <https://arxiv.org/abs/2311.17871>
- <https://doi.org/10.1109/CDC49753.2023.10383256>

Any questions, inquiries or issues can be communicated to j.s.v.hulst@tue.nl

### Citation

```bibtex
@INPROCEEDINGS{Hulst2023,
  AUTHOR    = {Jilles van Hulst and Roy van Zuijlen and Duarte Antunes and W.P.M.H. Heemels},
  TITLE     = {Estimation of Dynamic Gaussian Processes},
  BOOKTITLE = {2023 IEEE 62nd Conference on Decision and Control (CDC)},
  MONTH     = {December},
  YEAR      = {2023},
  DOI       = {10.1109/CDC49753.2023.10383256},
}
```

---

## Repository structure

The code is available in both **MATLAB** and **Python**.
Both versions share the same pipeline and configuration options.

```
MATLAB/                              python/
  Dynamic_Gaussian_Process_Main.m     main.py              <- entry point
  DGP_function_fitting.m              dgp_function_fitting.py
  DGP_estimation.m                    dgp_estimation.py
  DGP_heat_equation.m                 dgp_truth_generators.py
  DGP_wave_equation.m                    (heat, wave, simulation, data)
  DGP_simulation.m
  DGP_data.m
  squexp.m                            squexp.py
  heat_equation_simulation.mat        plotting.py
```

---

## Quick start

### MATLAB

Open `MATLAB/Dynamic_Gaussian_Process_Main.m` in MATLAB, edit the
configuration block at the top, and run the script (press **F5**).

### Python

```bash
cd python
python main.py
```

Requires **Python >= 3.9** with `numpy`, `scipy`, and `matplotlib`.

---

## Configuration options

Both scripts expose the same settings through a configuration struct / dataclass.

| Setting            | Choices                                                          | Description                              |
|--------------------|------------------------------------------------------------------|------------------------------------------|
| `basis`            | `'RBF'`, `'NRBF'`, `'Fourier'`, `'Discrete'`                    | Basis function family                    |
| `kernel`           | `'heat_equation'`, `'wave_equation'`, `'smoothing'`, `'integrator'` | Evolution kernel for the estimator     |
| `initial_mean`     | `'zero'`, `'squexp'`, `'ones'`, `'parabola'`, `'dirac'`          | Prior mean function *m(x)*               |
| `system`           | `'Discrete_approximation'`, `'Heat_equation'`, `'Wave_equation'`, `'Data'` | Ground-truth data source       |

### Kernel types

| Kernel             | States | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `heat_equation`    | 1      | Gaussian Green's function for the 1-D heat equation. Parameters: thermal diffusivity `alpha`, sample time `dt`. |
| `wave_equation`    | 2      | D'Alembert 2x2 kernel for the 1-D wave equation. Two function-valued states [*f*, *g*] where *g* = df/dt. Parameters: wave speed `c`, sample time `dt`. |
| `smoothing`        | 1      | Squared-exponential blur -- each step convolves the function with a normalised Gaussian, causing it to flatten over time. Parameters: length scale `sigma_kf`, amplitude `a_kf`. |
| `integrator`       | 1      | Kronecker-delta / identity -- the function does not evolve deterministically; all change comes from the disturbance (random-walk prior). |

### Truth sources

| System                    | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `Discrete_approximation`  | Simulates the chosen kernel on a fine grid (single-state only).    |
| `Heat_equation`           | Analytical heat-equation solution (Dirac initial condition).       |
| `Wave_equation`           | Analytical D'Alembert solution.                                    |
| `Data`                    | Load measurements from a `.mat` file. Configure any kernel above to match the expected dynamics of your data. |

### Using your own data

1. Set `system = 'Data'` (MATLAB) or `system="Data"` (Python).
2. Choose a `kernel` and `initial_mean` that match your prior belief about the dynamics.
3. Place your `.mat` file (with variables `x`, `y`, and optionally `f_true`, `x_test`) in the MATLAB folder.
4. Adjust the other parameters (`M`, `p`, `N`, `x_min`, `x_max`, `basis`, ...) as needed.