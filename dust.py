import numpy as np
import astropy.units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 5


# Setting up the Intial conditions
# depending on what system you are looking at,
# to begin with we look at a basic disk at 1 AU

basis = u.cgs.bases

print("Initializing starting conditions")
T = 196 * u.K
cs = (np.sqrt(const.k_B*T/(2.37*1.008*const.m_p))
      ).decompose(basis)  # cgs change
r = 1 * u.au
msolar = 0.5 * u.solMass
Omega_K = np.sqrt(const.G*msolar/r**3)
Hp = (cs / Omega_K).decompose()
sigma_d = 0.18 * u.g/u.cm**2
rho_d = (sigma_d/(np.sqrt(2*np.pi)*Hp)).decompose(basis)  # cgs change
rho_s = 1.6 * u.g * u.cm**(-3)
a_s = 1*u.micron
m_s = (4/3*np.pi*a_s**3 * rho_s).decompose(basis)  # cgs change
u_f = (1*u.m/u.s).decompose(basis)  # cgs change
trans_width = (0.001*u.m/u.s).decompose(basis)  # cgs change
alpha = 1e-3
Cdt = 0.1
xi = 1.83

dt1 = float(input("dt in yr: "))
dt = dt1 * 60*60*24*365 * u.s
t_end1 = float(input("t_end in yr: "))
t_end = t_end1 * 60*60*24*365 * u.s
save_time = 1 * u.yr
eps_time = 1.10


# Setting up binning

# Here the we define how the resolution of the simulation.
# We split the logspace we want to look at, in terms of the
# different pebble sizes, into bins of specific width.


print("Calculating bins")


# 40 for brown 200 for turb as per birnstiel
amin = float(input("min Dust size in logspace cm (micron is -4): "))
amax = float(input("max Dust size in logspace cm (2 is 100 cm): "))
nbin = int(input("resolution of bins: "))
a = np.logspace(-4, 2, nbin) * u.cm
st = (rho_s * a * np.pi/(2*sigma_d*100)).value
v = np.ones(nbin)
#rho_d = (sigma_d/ (np.sqrt(2*np.pi)* Hp * min(1,np.sqrt(alpha/min(st[0],1/2)*(1+st[0]**2))) )).decompose()

m = (4/3 * a**3 * rho_s * np.pi).decompose(basis)
n = np.zeros(len(m)) * u.m**(-3)
n[0] = (rho_d / m[0]).decompose()
n = n.decompose(basis)  # cgs change
ms = np.add.outer(m, m)
mv = m.value * v
p = n.value * v
q = m.value * p


print("Calculating velocities and sigma_col")
ubm = (np.sqrt(8 * const.k_B*T * (np.add.outer(m, m)) /
               (np.pi*np.outer(m, m)))).decompose()  # cgs change
utm = cs * np.sqrt(2*alpha*st)
utm[st > 1] = cs*np.sqrt(2*alpha/st[st > 1])

utm = np.maximum.outer(utm, utm)

if input("Turbulence on? y/n: ") == "y":
    du = ubm + utm
else:
    du = ubm
sigma_col = np.pi * np.add.outer(a, a)**2


print("Setting up probabilities and fragmentation matrix ")

size = len(a)
mn = np.zeros((size, size))
mm = np.zeros((size, size))
pf = np.zeros((size, size))

crat_ind = (np.divide.outer(m, m) > 10) | (np.divide.outer(m, m) < 0.1)
SS = np.zeros((size, size, size))

for i in range(len(m)):
    for j in range(len(m)):
        kn = (np.where(m < ms[i][j])[0])[-1]

        if crat_ind[i][j]:
            # cratering
            SS[:min(i, j)+1, i, j] = a[:min(i, j)+1] ** (-3*xi)
            Stot = np.sum(m.value * SS[:, i, j]) / (m[min(i, j)].value * 2)
            SS[:min(i, j)+1, i, j] /= Stot
            m_crat = np.abs(m[i].value-m[j].value)
            l = np.where(m.value < m_crat)[0][-1]
            eps = (m_crat - m[l].value)/(m[l+1].value - m[l].value)
            SS[l, i, j] = 1-eps
            SS[l+1, i, j] = eps

        else:
            # fragmentation
            SS[:max(i, j)+1, i, j] = a[:max(i, j)+1] ** (-3*xi)
            Stot = np.sum(m.value * SS[:, i, j]) / (m[i].value + m[j].value)
            SS[:max(i, j)+1, i, j] /= Stot

        if kn+1 < nbin:
            mm[i][j] = m[kn].value
            mn[i][j] = m[kn+1].value
        else:
            sigma_col[i][j] = 0

        if du[i][j] < u_f-trans_width:
            pf[i][j] = 0
        elif du[i][j] > u_f:
            pf[i][j] = 1
        else:
            pf[i][j] = 1-(u_f-du[i][j])/trans_width

pc = 1-pf


ep = (mn - ms.value) / (mn - mm)
ep[sigma_col == 0] = 0

C = np.zeros((size, size, size))
for k in range(size):
    imm = m[k].value == mm
    imn = m[k].value == mn
    C[k][imm] = ep[imm]
    C[k][imn] = 1 - ep[imn]

K = (np.multiply(np.multiply(du, sigma_col), pc)).decompose(basis)  # cgs change
L = (np.multiply(np.multiply(du, sigma_col), pf)).decompose(basis)  # cgs change


print("Time evolving the number density")
print("Starting n: ", (n.to(1/u.cm**3)))
print("Initial velocity: ", p)
n = n.value
K = K.value
L = L.value
dt = dt.value
m = m.value             # cgs change
ms = ms.value           # cgs change
t_end = t_end.value

x = np.array([])
y = np.array([])
y2 = np.array([])
vel = np.array([])

t = 0
n_save = n
i = 0
has_run = False
while t < t_end:
    if np.any(n < 0) or np.abs(np.sum(n*m)/rho_d.value - 1.) > 1e-2:
        if np.any(n < 0):
            print("Negative number particles, aborting", np.argmin(n), np.min(n))
        if np.abs(np.sum(n*m)/rho_d.value - 1.) > 1e-4:
            print("Mass not conserved, aborting",
                  (np.sum(n*m)/rho_d.value - 1.))
        print(f"Time elapsed: {t/(60*60*24*365)} yr")
        break

    dn = np.zeros(size)
    dm = np.zeros(size)
    Sp = np.zeros(size)
    dq = np.zeros(size)
    S = np.zeros(size)
    J = np.zeros((size, size))
    Jp = np.zeros((size, size))
    Jm = np.zeros((size, size))
    JJ = np.zeros((size*2, size*2))
    nn = np.outer(n, n)
    Q = np.multiply(K, nn)
    Q_frag = np.multiply(L, nn)
    mp = m * p
    pn = (np.outer(mp, n) + np.outer(n, mp))/ms

    for k in range(size):
        KC_k = np.multiply(K, C[k])
        LS_k = np.multiply(L, SS[k])

        Kn_k = np.multiply(K[k], n)
        Ln_k = np.multiply(L[k], n)

        Sp[k] = 1/2 * np.sum(np.multiply(KC_k, pn)) - np.sum(Kn_k * p[k]) \
            + 1/2 * np.sum(np.multiply(LS_k, pn)) - np.sum(Ln_k * p[k])

        S[k] = (0.5 * np.sum(np.multiply(Q, C[k])) - np.sum(Q[k])) + \
               (0.5 * np.sum(np.multiply(Q_frag, SS[k])) - np.sum(Q_frag[k]))
        Jp[k] = (np.dot(KC_k/ms, n) + np.dot(LS_k/ms, n)) * m
        Jp[k][k] -= np.sum(Kn_k) + np.sum(Ln_k)
        Jm[k] = (np.dot(KC_k/ms, (p*m)) + np.dot(LS_k/ms, (p*m))) \
            - p[k]*K[k] - p[k]*L[k]
        J[k] = (np.dot(KC_k, n) - n[k]*K[k]) + \
               (np.dot(LS_k, n) - n[k]*L[k])
        J[k][k] -= np.sum(Kn_k) + np.sum(Ln_k)

    JJ[:size, :size] = (np.identity(size)/dt - J)
    JJ[size:, size:] = (np.identity(size)/dt - Jp)
    JJ[:size, size:] = Jm
    SSS = np.concatenate((S, Sp))

    dd = np.linalg.solve(JJ, SSS)
    dn = dd[:size]
    dp = dd[size:]
    n += dn  # (u.s * u.m**3)**(-1)
    p += dp
    t += dt

    if t > save_time.value:
        i += 1
        save_time *= eps_time
        n_save = np.append(n_save, n)
        y = np.append(y, np.argmax(n*m))
        y2 = np.append(y2, np.argmax(n))
        x = np.append(x, t)
        vel = np.append(vel, np.max(p/(n+1e-5)))
        if i % 10 == 0:
            print("Time left:", (t_end-t)/(60*60*24*365),
                  n[0], "Stepping: ", dt/(60*60*24*365))
            print("p", p[0])


print("Lost mass to total ratio : ")
print(((sum(n*m)-rho_d.value)/rho_d.value))
print("Final n: ", n)
print("Final velocity: ", p/(n+1e-30))
print("Momentum conservation: ", np.sum(p*m) -
      np.sum(p*m), (np.sum(p*m)-np.sum(p*m))/np.sum(p*m))

x = (x*u.s).to(u.yr).value
y = a[y.astype(int)].value
y2 = a[y2.astype(int)].value
n_save = np.reshape(n_save, (i+1, len(n)))

plt.loglog(a, n*m)
plt.title("Grain size vs density")
plt.xlabel("grain size in cm")
plt.ylabel("density")
plt.show()
