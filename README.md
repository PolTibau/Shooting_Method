# Shooting Method

Pràctica de Càlcul Numèric sobre equacions diferencials ordinàries: tir d'un
projectil amb fricció quadràtica de l'aire (`R = 0.00132`, velocitat inicial de
100 m/s) i mètode del tir (*shooting method*) per trobar l'angle de llançament
que fa arribar el projectil a una distància donada (500 m).

L'informe de la pràctica és a [`Shooting_Method_PolTibau.pdf`](Shooting_Method_PolTibau.pdf).

## Estructura

| Fitxer | Contingut |
| --- | --- |
| `Shooting_Method_PolTibau.py` | Exercicis 1-6: comparació d'Euler, RK4 i RK45 (`solve_ivp`), estudi de l'error, detecció de l'impacte amb `events`, mètode del tir amb Newton-Raphson i cerca dicotòmica (amb dues solucions possibles) |
| `metodes.py` | Mòdul d'ODEs de l'assignatura: `Euler`, `EulerBackward`, `RK2`, `RK4` i `RKF45`. L'script només n'usa `Euler` i `RK4` |

## Execució

```
pip install -r requirements.txt
python Shooting_Method_PolTibau.py
```

L'script imprimeix els resultats de cada exercici i mostra les gràfiques
corresponents.
