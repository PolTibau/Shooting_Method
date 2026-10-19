# Shooting Method

Numerical Analysis assignment on ordinary differential equations: projectile motion with
quadratic air drag (`R = 0.00132`, initial speed of 100 m/s) and the shooting method to find
the launch angle that makes the projectile land at a given distance (500 m).

The assignment report (in Catalan) is [`Shooting_Method_PolTibau.pdf`](Shooting_Method_PolTibau.pdf).

## Structure

| File | Contents |
| --- | --- |
| `Shooting_Method_PolTibau.py` | Exercises 1-6: comparison of Euler, RK4 and RK45 (`solve_ivp`), error study, impact detection with `events`, shooting method with Newton-Raphson and bisection (with two possible solutions) |
| `metodes.py` | ODE module from the course: `Euler`, `EulerBackward`, `RK2`, `RK4` and `RKF45`. The script only uses `Euler` and `RK4` |

## Running

```
pip install -r requirements.txt
python Shooting_Method_PolTibau.py
```

The script prints the results of each exercise and shows the corresponding plots.
