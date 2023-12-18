import numpy as np
from metodes import Euler, RK4, RKF45
import matplotlib.pyplot as plt
import scipy.integrate as sc

def f(t,x):
    v = np.array([x[2],x[3]])
    R = 0.00132
    g = np.array([0,-9.8])
    return np.array([v[0],v[1],-R*np.linalg.norm(v)*v[0]+g[0],-R*np.linalg.norm(v)*v[1]+g[1]])

def angle(phi):
    IC = [0,0,100*np.cos(phi),100*np.sin(phi)]
    return IC 

def diferencia(phi, D, tol):
    return (sc.solve_ivp(f, [0,20], angle(phi), method='RK45',rtol=tol, events=Terra).y_events[0][1][0]-D)

def derivada(phi, D, h=1e-5):
    return (diferencia(phi+h,D,tol)-diferencia(phi,D,tol))/h

def Terra(t,x):
    return x[1]

def tir(distancia, phi_ini, tol, max_it):
    phi = phi_ini
    it = 0
    while (abs(diferencia(phi, distancia,tol)) > tol and it < max_it):
        sol = sc.solve_ivp(f, [0,20], angle(phi), method='RK45', rtol=tol, events=Terra)
        plt.plot(sol.y[0],sol.y[1],'-.')
        phi -= diferencia(phi, distancia,tol) / derivada(phi, distancia)
        it += 1       
    return phi, it

#Terra.terminal = True #Després de que el projectil arribi al Terra aturem la resolució de la edo

Y = [0,0,100*np.cos(np.pi/4),100*np.sin(np.pi/4)]


print("Exercici 1")
Xrk4, Yrk4 =     RK4(f,[0,10], Y, 20)
Xeuler, Yeuler = Euler(f,[0,10], Y, 20)
plt.title("Solució amb el mètode d'Euler i RK4")
plt.plot(Yeuler[0],Yeuler[1], '-o', label = 'Paràbola Euler')
plt.plot(Yrk4[0], Yrk4[1], 'o-', label = 'Paràbola RK4')
plt.legend()
plt.show()
print("A temps t=10 tenim per coordenades amb Euler: ", Yeuler[0][20], Yeuler[1][20])
print("A temps t=10 tenim per coordenades amb RK4: ", Yrk4[0][20], Yrk4[1][20])

print()
print("Exercici 2")
#Veiem com es comporta l'error de les aproximacions.
Xrk4_2, Yrk4_2 = RK4(f,[0,10], Y, 40)
Xeuler_2, Yeuler_2 = Euler(f,[0,10], Y, 40)
rEuler = np.sqrt((Yeuler_2[0][40]-Yeuler[0][20])**2 + (Yeuler_2[1][40]-Yeuler[1][20])**2)/np.sqrt(Yeuler_2[0][40]**2 + Yeuler_2[1][40]**2)
rRK4 = np.sqrt((Yrk4_2[0][40]-Yrk4[0][20])**2 + (Yrk4_2[1][40]-Yrk4[1][20])**2)/np.sqrt(Yrk4_2[0][40]**2 + Yrk4_2[1][40]**2)
print("L'error amb Euler amb 20 subintervals és de: ", rEuler)
print("L'error amb RK4 amb 20 subintervals és de: ", rRK4)

Xrk4_3, Yrk4_3 = RK4(f,[0,10], Y, 200)
Xeuler_3, Yeuler_3 = Euler(f,[0,10], Y, 200)

Xrk4_4, Yrk4_4 = RK4(f,[0,10], Y, 400)
Xeuler_4, Yeuler_4 = Euler(f,[0,10], Y, 400)

rEuler = np.sqrt((Yeuler_4[0][400]-Yeuler_3[0][200])**2 + (Yeuler_4[1][400]-Yeuler_3[1][200])**2)/np.sqrt(Yeuler_4[0][400]**2 + Yeuler_4[1][400]**2)
rRK4 = np.sqrt((Yrk4_4[0][400]-Yrk4_3[0][200])**2 + (Yrk4_4[1][400]-Yrk4_3[1][200])**2)/np.sqrt(Yrk4_4[0][400]**2 + Yrk4_4[1][400]**2)
print("L'error amb Euler amb 200 subintervals és de: ", rEuler)
print("L'error amb RK4 amb 200 subintervals és de: ", rRK4)

print()
print("Exercici 3")
print("Veure evolució de l'error amb ambdós mètodes a la gràfica")
#Iterarem els dos mètodes per observar la funció de l'error
Eeuler = []
Erk4 = []
for i in range(20, 200, 10):
    Yrk4 = RK4(f,[0,10], Y, i)[1]
    Yeuler = Euler(f,[0,10], Y, i)[1]
    Yrk4_e = RK4(f,[0,10], Y, 2*i)[1]
    Yeuler_e = Euler(f,[0,10], Y, 2*i)[1]
    
    rEuler = np.sqrt((Yeuler_e[0][2*i]-Yeuler[0][i])**2 + (Yeuler_e[1][2*i]-Yeuler[1][i])**2)/np.sqrt(Yeuler_e[0][2*i]**2 + Yeuler_e[1][2*i]**2)
    rRK4 = np.sqrt((Yrk4_e[0][2*i]-Yrk4[0][i])**2 + (Yrk4_e[1][2*i]-Yrk4[1][i])**2)/np.sqrt(Yrk4_e[0][2*i]**2 + Yrk4_e[1][2*i]**2)
    
    Eeuler.append(rEuler)
    Erk4.append(rRK4)
    
S = np.arange(20,200,10)
plt.title("Evolució de l'error amb els mètodes d'Euler i RK4")
plt.plot(S,np.log10(Eeuler), '-o', label = 'Error Euler')
plt.plot(S, np.log10(Erk4), '-o', label = 'Error RK4')
plt.legend()
plt.show()

print()
print("Exercici 4")
#Implementem el mètode de RK45 de pas variable i l'executem
SolRK45 = sc.solve_ivp(f, [0,10],Y,method='RK45',rtol=1e-5)
SolRK45exact = sc.solve_ivp(f, [0,10],Y,method='RK45', rtol=1e-8)
error = np.sqrt((SolRK45.y[0][-1]-SolRK45exact.y[0][-1])**2 + (SolRK45.y[1][-1]-SolRK45exact.y[1][-1])**2)/np.sqrt(SolRK45.y[0][-1]**2 + SolRK45exact.y[1][-1]**2)
Xrk4, Yrk4 =     RK4(f,[0,10], Y, 20)
Xeuler, Yeuler = Euler(f,[0,10], Y, 20)

print("Podem estimar que l'error global amb el mètode RK45 és: ", error)

plt.title("Solució amb el mètode d'Euler, RK4 i RK45")
plt.plot(SolRK45.y[0],SolRK45.y[1], '-o', label = 'Paràbola RK45')
plt.plot(Yeuler[0],Yeuler[1], '-o', label = 'Paràbola Euler')
plt.plot(Yrk4[0], Yrk4[1], 'o-', label = 'Paràbola RK4')
plt.legend()
plt.show()
  
print()
print("Exercici 5")
#Usem de nou el mètode RK45 per ara donat un angle
phi = np.pi/4
SolRK45_2 = sc.solve_ivp(f, [0,20], angle(phi), method='RK45', events=Terra)

plt.plot(SolRK45_2.y[0],SolRK45_2.y[1], '-o', label = 'Paràbola RK45')
print("El projectil arriba al terra a t =", SolRK45_2.t_events[0][1])
print("La distància horitzontal recorreguda  és de x =", SolRK45_2.y_events[0][1][0])
plt.title("Implementació de l'argument 'events'")
plt.legend()
plt.show()

print()
print("Exercici 6")
#Vegem ara quin angle cal per arribar a una longitud de 500 metres
#Simularem un problema de zero de funcions de Newton-Raphson
tol=1e-5
bo = tir(500, np.pi/4 ,tol, 1000)
traj_precisa  = sc.solve_ivp(f, [0,20], angle(bo[0]), method='RK45', rtol=tol, events=Terra)
plt.title("Aproximació amb mètode de Newton")
plt.plot(traj_precisa.y[0], traj_precisa.y[1],'-ro')
print("La distància recorreguda és: ",traj_precisa.y_events[0][1][0])
print("L'angle ideal trobat és: ", bo[0]*180/np.pi)
print("Amb N-R s'ha trobat amb: ", bo[1], "iteracions")
plt.show()

print()
print("Exercici 6**")
#Vegem ara quin angle cal per arribar a una longitud de 500 metres
#Simularem una cerca dicotòmica i veurem que podem trobar dos solucions!

tol = 1e-1
phi2 = np.pi/2
phi1 = np.pi/4
E = 100
while(E >= tol):
    SolRK45_2 = sc.solve_ivp(f, [0,20], angle((phi2+phi1)/2), rtol=1e-10, events=Terra)
    plt.plot(SolRK45_2.y[0],SolRK45_2.y[1], '-o')
    E = abs(SolRK45_2.y_events[0][1][0]-500)
    if(E < tol):
        print("L'angle trobat és: ", (phi1+phi2)/2 * 180/np.pi, " (en graus)")
    elif(SolRK45_2.y_events[0][1][0]-500 > 0):
        phi1 = (phi2+phi1)/2
    else:
        phi2 = (phi2+phi1)/2
plt.title("Aproximació amb certa dicotòmica [45,90]")
plt.show()


phi2 = np.pi/2
phi1 = 0
E = 100
while(E >= tol):
    SolRK45_2 = sc.solve_ivp(f, [0,20], angle((phi2+phi1)/2), rtol=1e-10, events=Terra)
    plt.plot(SolRK45_2.y[0],SolRK45_2.y[1], '-o')
    E = abs(SolRK45_2.y_events[0][1][0]-500)
    if(E < tol):
        print("L'angle trobat és: ", (phi1+phi2)/2 * 180/np.pi, " (en graus)")
    elif(SolRK45_2.y_events[0][1][0]-500 > 0):
        phi2 = (phi2+phi1)/2
    else:
        phi1 = (phi2+phi1)/2
plt.title("Aproximació amb certa dicotòmica [0,90]")
plt.show()




