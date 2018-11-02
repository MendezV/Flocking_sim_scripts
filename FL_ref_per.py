import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.fftpack as fft

def m_rungekutta4per(func, y_0, t, args={}):
	y = np.zeros([len(t), len(y_0)])
	y[0] = y_0
	h = t[1]-t[0]

	for i in range(1,len(y)):

k1 = func(t[i-1],y[i-1],args)

#paso 1
t1 = t[i-1] + (h/2.0)
	y1 = y[i-1] + (h/2.0) * k1
	k2 = func(t1, y1,args)
	
	#paso 2
	t2 = t[i-1] + (h/2.0)
	y2 = y[i-1] + (h/2.0) * k2
	k3 = func(t2, y2,args)
	
	#paso 3
	t3 = t[i-1] + h
	y3 = y[i-1] + (h * k3)
	k4 = func(t3, y3,args)
	
	#paso 4
	pendiente = (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
	
	t[i] = t[i-1] + h
	y[i] = y[i-1] + h * pendiente
	
	y[i][:2*M:2]=y[i][:2*M:2]%(LXf-LX0)
	y[i][2*M:4*M:2]=y[i][2*M:4*M:2]%(LYf-LY0)
		return(y)

def graficar_fase_pendulo(t,y):
	# Grafique el ángulo, diagrame de fase y FFT del péndulo
	
	fig = plt.figure()

	# Plot Angle
	ax_1 = fig.add_subplot(211)
	ax_1.plot(t, y[:,0], c='b')
		ax_1.set_xlabel('Tiempo (s)')
		ax_1.set_ylabel('Angulo (rad)')

		# Plot Phase Diagram
		ax_2 = fig.add_subplot(223)
			ax_2.plot(y[:,0], y[:,1], c='g')
			ax_2.set_xlabel('Angulo (rad)')
			ax_2.set_ylabel('Vel. Angular (rad /s)')
		
			# Calcule la transformada de fourier
			f_fft = fft.fftfreq(len(t), t[1]-t[0])
				y_fft = fft.fft(y[:,0])/np.sqrt(2*len(t))
			
				# Grafique el espectro de potencia (Transformada de Fourier)
				ax_3 = fig.add_subplot(224)
					ax_3.plot(f_fft[:int(N/2)]*2*np.pi, abs(y_fft[:int(N/2)]), c='r')
					ax_3.set_xlim([0, 30])
					ax_3.set_xlabel('Freq. Angular ($2 \pi$ Hz)')
					ax_3.set_ylabel('Potencia')


####################
# Condiciones Iniciales
# angulos pequeños
M=10
D=2 #2 dimensions
c_i_mult = np.zeros(D*2*M)


X_loc=0
X_scale=2
VX_loc=0
VX_scale=2
Y_loc=0
Y_scale=2
VY_loc=0
VY_scale=2
#Positions
#Positions

for i in range(0,2*M,2):
	c_i_mult[i]=np.random.normal(loc = X_loc, scale = X_scale)
for i in range(1,2*M,2):
	c_i_mult[i]=np.random.normal(loc = VX_loc, scale = VX_scale)

for i in range(2*M,2*2*M,2):
	c_i_mult[i]=np.random.normal(loc = Y_loc, scale = Y_scale)
for i in range(2*M+1,2*2*M,2):
	c_i_mult[i]=np.random.normal(loc = VY_loc, scale = VY_scale)


g = 10. # [m /s2]

l_pendulo = 1. # [m]
m_pendulo = 1. # [kg]
fr = 1. # [kg /m /s]

F_ext = 0. # [N]
freq_ext = 0. # [2π /s]

N = 4000 # n_puntos
t = np.linspace(0., 200., N+1) # [s] arreglo de n_puntos en el tiempo



h = t[1]-t[0]
LX0=-10
LXf=10
LY0=-10
LYf=10
LX=LXf-LX0
LY=LYf-LY0

# definir los argumentos
args_sol = {}
'''
	args_sol['alpha'] = g/l_pendulo
	args_sol['beta'] = fr/m_pendulo
	args_sol['gamma'] = F_ext/(m_pendulo*l_pendulo)
	args_sol['omega'] = freq_ext
	'''
args_sol['alpha'] = g/l_pendulo
args_sol['beta'] = fr/m_pendulo
args_sol['gamma'] = 1000
args_sol['omega'] = 0

############################


############################


def forceXper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	#the scale factor is handled as if it was uniform among all agents
	return sin((y1-y2))

def forceYper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	#the scale factor is handled as if it was uniform among all agents
	return -sin((w1-w2))

def force2Xper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	rangei=1
		signs_X=[np.sign(y1-y2),np.sign(y1-y2+LX),np.sign(y1-y2-LX)]
		delta_X=min([abs(y1-y2),abs(y1-y2+LX),abs(y1-y2-LX)])
		delta_Y=min([abs(w1-w2),abs(w1-w2+LY),abs(w1-w2-LY)])
		#the scale factor is handled as if it was uniform among all agents
			return signs_X[np.argmin([abs(y1-y2),abs(y1-y2+LX),abs(y1-y2-LX)])]*(delta_X/rangei)*exp(-0.5*(delta_X/rangei)**2-0.5*(delta_Y/rangei)**2)

def force2Yper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	rangei=1
		signs_Y=[np.sign(w1-w2),np.sign(w1-w2+LX),np.sign(w1-w2-LX)]
		delta_X=min([abs(y1-y2),abs(y1-y2+LX),abs(y1-y2-LX)])
		delta_Y=min([abs(w1-w2),abs(w1-w2+LY),abs(w1-w2-LY)])
		#the scale factor is handled as if it was uniform among all agents
			return signs_Y[np.argmin([abs(w1-w2),abs(w1-w2+LY),abs(w1-w2-LY)])]*(delta_Y/rangei)*exp(-0.5*(delta_X/rangei)**2-0.5*(delta_Y/rangei)**2)


def force3Xper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	rangei=1
		delta_X=min([y1-y2,y1-y2+LX,y1-y2-LX])
		delta_Y=min([w1-w2,w1-w2+LY,w1-w2-LY])
		#the scale factor is handled as if it was uniform among all agents
			return (delta_X)/((delta_Y)**2+(delta_X)**2)

def force3Yper(y1,y2,w1,w2,vy1,vy2,vw1,vw2,t):
	rangei=1
		delta_X=min([y1-y2,y1-y2+LX,y1-y2-LX])
		delta_Y=min([w1-w2,w1-w2+LY,w1-w2-LY])
		#the scale factor is handled as if it was uniform among all agents
			return (delta_Y)/((delta_Y)**2+(delta_X)**2)


def pairFXper(i,y,t):
	F=0
		for j in ([l for l in range(0,i,2)]+[l for l in range(i+2,2*M-1,2)]):
			F+=force2Xper(y[i],y[j],y[i+2*M],y[j+2*M],y[i+1],y[j+1],y[i+1+2*M],y[j+1+2*M],t)
				return F

def pairFYper(i,y,t):
	F=0
		for j in ([l for l in range(0,i,2)]+[l for l in range(i+2,2*M-1,2)]):
			F+=force2Yper(y[i],y[j],y[i+2*M],y[j+2*M],y[i+1],y[j+1],y[i+1+2*M],y[j+1+2*M],t)
				return F

def extFX(yi,wi,t):
	return np.sin(yi-t)

def extFY(yi,wi,t):
	return np.sin(wi-t)

####################

# Condiciones Iniciales
# angulos pequeños

def pendulo7(t,y,args):
	dydt = np.zeros(2*2*M)
	for i in range(0,2*M,2):
		dydt[i] = y[i+1] #primera ecuación
		dydt[i+1] = args['alpha']*pairFXper(i,y,t) + args['omega']*extFX(y[i],y[i+2*M],t) - args['beta']*y[i+1] + args['gamma']*np.random.normal(loc = 0.0, scale = np.sqrt(h))
		
		dydt[i+2*M] = y[i+1+2*M] #primera ecuación
	dydt[i+1+2*M] = args['alpha']*pairFYper(i,y,t) + args['omega']*extFY(y[i],y[i+2*M],t) - args['beta']*y[i+1+2*M] + args['gamma']*np.random.normal(loc = 0.0, scale = np.sqrt(h))
		
		
		return dydt




args_sol['alpha'] = 10
args_sol['beta'] = 0.1
args_sol['gamma'] = 10
args_sol['omega'] =  1


y = m_rungekutta4per(pendulo7, c_i_mult, t, args_sol)
graficar_fase_pendulo(t,y)
plt.show()


xdata=(y[:,:M]).flatten()
vxdata=(y[:,M:2*M]).flatten()
ydata=(y[:,2*M:3*M]).flatten()
vydata=(y[:,3*M:4*M]).flatten()

print("x",np.histogram(xdata))
print("vx",np.histogram(vxdata))
print("y",np.histogram(ydata))
print("vy",np.histogram(vydata))

xedges = arange(0,20,1)
yedges = arange(0,20,1)
H, xedges, yedges = np.histogram2d(y[:,0:2*M:2].flatten(), y[:,2*M:4*M:2].flatten(),bins=(xedges, yedges))
H = H.T
# Let each row list bins with common y range.

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='Square Bins')
plt.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.show()