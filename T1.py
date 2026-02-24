import numpy as np
import matplotlib.pyplot as plt
import math, scipy
from scipy.special import jv as J 


resolucion = ((5, 50),(100),(10))  # res cilindro (z,phi), res cargas (theta), res pot

# coords cilindricas r φ z

def cilindro(r,phi,z):  
    return (r*math.cos(phi), r*math.sin(phi), z)

# cilindro 
L = 1 
z = np.linspace(0,L, resolucion[0][0])
Cphi = np.linspace(0,2*math.pi,resolucion[0][1])
R = 1

# cargas 
h = 0.5
zq = h

# puntual
q = 1
xq = 0
yq = 0

# anillo
λ0 = 1 # λ = a2πλ0
a = 0.5
Atheta = np.linspace(0, 2 * np.pi, resolucion[1]) # otro anguo para darle resolucion distinta
x_anillo = a * np.cos(Atheta)
y_anillo = a * np.sin(Atheta)
z_anillo = np.full_like(Atheta, h)


# malla del cilindro
X = np.zeros((len(z),len(Cphi)))
Y = np.zeros((len(z),len(Cphi)))
Z = np.zeros((len(z),len(Cphi)))

for i in range(len(z)):
    for j in range(len(Cphi)):
        X[i,j],Y[i,j],Z[i,j] = cilindro(R,Cphi[j],z[i])

# Potencial

def xnm(n,m): #cero m-esimo de la n-esima funcion de Bessel
    xn = scipy.special.jn_zeros(0,m)
    return xn[n]


def mM(mM,A,b):
    A = np.array(A)
    if mM == "m": 
            return np.minimum(a,b)
    if mM == "M": 
            return np.maximum(a,b)
    if a == b:
        return a

def Φ(x,y,z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    zm = mM("m",z,h)
    zM = mM("M",z,h)
    C = 8*np.pi/(R**2)
    sol = []
    for n in range(1,int(resolucion[2])):  # n > 0
        num = a*λ0*J(0,xnm(0,n)*a/R) + q/(2*np.pi)
        denm =  (xnm(0,n)/R)*(J(1,xnm(0,n))**2)*np.sinh(xnm(0,n)*L/R) 
        An = num / denm
        sol.append(An*J(0,xnm(0,n)*r/R)*np.sinh(xnm(0,n)*zm/R)*np.sinh(xnm(0,n)*(L-zM)/R)) 
    return C*sum(sol)

xz = np.linspace(-2*R, 2*R, resolucion[2])
zx = np.linspace(-2*L, 2*L, resolucion[2])
yy = np.zeros_like(xz)
x,z  = np.meshgrid(xz,zx)


# grafico en color del potencial para y = 0 fijo, plano xz
fig, ax = plt.subplots()
# Inicializar la malla de potenciales
V = np.zeros_like(x)

# Evaluar Φ punto por punto
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        V[i, j] = Φ(x[i, j], 0, z[i, j])  # y=0 en este plano

# Luego graficás V
countour = ax.contourf(x, z, V, levels=100, cmap='viridis')

print(Φ(0.5, 0, 0.3))
plt.colorbar(countour, label='Potencial')
plt.axvline(-R,0,L, color='k', linewidth=0.5)
plt.axvline(R,0,L, color='k', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Z')
plt.xlim(-R*1.2, R*1.2)
plt.ylim(-0.2, L*1.2)
plt.title('Potencial en el plano xz')
plt.show(block = True)


# plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])  # aspect ratio is 1:1:1
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('2019 problema 2')

# Ajustar los ticks en los ejes
ax.set_xticks(np.arange(-R, R + 0.5, 0.5))  # Ticks en el eje X
ax.set_yticks(np.arange(-R, R + 0.5, 0.5))  # Ticks en el eje Y
ax.set_zticks(np.arange(0, L + 0.5, 0.5))   # Ticks en el eje Z
# Ajustar el tamaño de las etiquetas de los ticks
ax.tick_params(axis='x', labelsize=12)  # Tamaño de las etiquetas en el eje X
ax.tick_params(axis='y', labelsize=12)  # Tamaño de las etiquetas en el eje Y
ax.tick_params(axis='z', labelsize=12)  # Tamaño de las etiquetas en el eje Z

# conductor
ax.plot_surface(X,Y,Z, color = "blue", alpha=0.2)
ax.plot_wireframe(X,Y,Z, color='black', alpha=0.2)
# carga puntual
ax.scatter(xq, yq, zq, color='black', s=10, label=f'q = {q}C')
# Anillo
ax.plot3D(x_anillo, y_anillo, z_anillo, color='k', linewidth=2, label= r'$λ = 2\pi a \lambda_0$')

ax.view_init(azim=45, elev=30)
plt.legend(loc = "upper right", facecolor='white', framealpha=0.5, fontsize = 12, edgecolor='grey')
plt.show(block = True)
