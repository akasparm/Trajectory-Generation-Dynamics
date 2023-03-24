#-------------------------------Importation of required libraries-------------------------------------
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linspace


#------------------------------Declaration of variables and symbols-----------------------------------
theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7  = symbols('th1, th2, th3, th4, th5, th6, th7', real=True)
theta = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7]
theta[2] = 0

#--------------------------Giving the values of d, a, alpha instead of symbols------------------------------------
d = [0]*7
d[0] = 0.3330
d[1] = 0
d[2] = 0.3160
d[3] = 0
d[4] = 0.3840
d[5] = 0
d[6] = -0.1070 - 0.1

a = [0]*7
a[0] = 0
a[1] = 0
a[2] = 0.0880
a[3] = -0.0880
a[4] = 0
a[5] = 0.0880
a[6] = 0

alpha = [0]*7
alpha[0] = pi/2
alpha[1] = -pi/2
alpha[2] = -pi/2
alpha[3] = pi/2
alpha[4] = pi/2
alpha[5] = -pi/2
alpha[6] = 0


#-----------------------------Calculations of Homogeneous Transformation Matrices---------------------------------
T_calc = []
for i in range(7):
    x = Matrix([[cos(theta[i]), -sin(theta[i])*cos(alpha[i]), sin(theta[i])*sin(alpha[i]), a[i]*cos(theta[i])],
         [sin(theta[i]), cos(theta[i])*cos(alpha[i]), -cos(theta[i])*sin(alpha[i]), a[i]*sin(theta[i])],
         [0, sin(alpha[i]), cos(alpha[i]), d[i]],
         [0, 0, 0, 1]
        ])
    T_calc.append(x)

final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]*T_calc[6]
final_trans = final_trans.evalf()

#---------------------------------T01--------------------------------------
final_trans = T_calc[0]
final_trans = final_trans.evalf()
t01 = final_trans
z1 = final_trans[0:3,2]

#---------------------------------T02--------------------------------------
final_trans = T_calc[0]*T_calc[1]
final_trans = final_trans.evalf()
t02 = final_trans
z2 = final_trans[0:3,2]

#---------------------------------T04--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]
final_trans = final_trans.evalf()
t04 = final_trans
z4 = final_trans[0:3,2]

#---------------------------------T05--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]
final_trans = final_trans.evalf()
t05 = final_trans
z5 = final_trans[0:3,2]

#---------------------------------T06--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]
final_trans = final_trans.evalf()
t06 = final_trans
z6 = final_trans[0:3,2]

#---------------------------------T07--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]*T_calc[6]
final_trans = final_trans.evalf()
t07 = final_trans
z7 = final_trans[0:3,2]
xp = final_trans[0:3, 3]


#-----------------------------------Upper half of jacobian------------------------------
p1 = diff(xp, theta[0])
p2 = diff(xp, theta[1])
p4 = diff(xp, theta[3])
p5 = diff(xp, theta[4])
p6 = diff(xp, theta[5])
p7 = diff(xp, theta[6])

#--------------------------------------Jacobian Matrix----------------------------------
jac = Matrix([[p1[0], p2[0], p4[0], p5[0], p6[0], p7[0]], [p1[1], p2[1], p4[1], p5[1], p6[1], p7[1]], [p1[2], p2[2], p4[2], p5[2], p6[2], p7[2]], 
                [z1[0], z2[0], z4[0], z5[0], z6[0], z7[0]], [z1[1], z2[1], z4[1], z5[1], z6[1], z7[1]], [z1[2], z2[2], z4[2], z5[2], z6[2], z7[2]]])


###########################################################################################
######################################## HW 5 #############################################
###########################################################################################


m = [3.06, 4.97, 0.64, 3.22, 3.58, 1.22, 1.66, 0.93]            #Mass vector
g = -9.8
h1 = 0.07
h2 = 0.236
h3 = 0.333 + 0.096*cos(theta[1])
h4 = 0.333 + 0.254*cos(theta[1])
h5 = h4 + 0.082*cos(theta[3])
h6 = h5
h7 = h6 - 0.088*cos(theta[5])

pe = -(m[0]*h1 + m[1]*h2 + m[2]*h3 + m[3]*h4 + m[4]*h5 + m[5]*h6 + m[6]*h7 + m[7]*h7)*g

gm = []
gm.append(0)
gm.append(diff(pe, theta[1]))
gm.append(diff(pe, theta[3]))
gm.append(0)
gm.append(diff(pe, theta[5]))
gm.append(0)
gm = Matrix(gm)

print("==============================Gravity Matrix=================================")
pprint(gm)

ext_force = Matrix([[-5], [0], [0], [0], [0], [0]])


#--------------------------Iteration of Jacobian and Velocity Matrix--------------------
theta_inst = [0, 0, pi/2, 0, pi, 0]
w = 2*pi/200

plt.title('Joint Torque vs Time')
jt1 = []
jt2 = []
jt3 = []
jt4 = []
jt5 = []
jt6 = []



for t in range(200):
    p = jac.subs(theta_1, theta_inst[0]).subs(theta_2, theta_inst[1]).subs(theta_4, theta_inst[2]).subs(theta_5, theta_inst[3]).subs(theta_6, theta_inst[4]).subs(theta_7, theta_inst[5])
    jac_inv = p.inv()
    x_dot = Matrix([[0], [0.1*w*math.sin(w*t+pi/2)], [0.1*w*math.cos(w*t+pi/2)], [0], [0], [0]])
    q_dot = ((jac_inv*x_dot)% (2*pi)).evalf()  

    for i in range(len(q_dot)):
        theta_inst[i] += q_dot[i]

    gravity_matrix = gm.subs(theta_1, theta_inst[0]).subs(theta_2, theta_inst[1]).subs(theta_3, 0).subs(theta_4, theta_inst[2]).subs(theta_5, theta_inst[3]).subs(theta_6, theta_inst[4]).subs(theta_7, theta_inst[5])
    joint_torque = (gravity_matrix - p.T*ext_force).evalf()

    jt1.append(joint_torque[0])
    jt2.append(joint_torque[1])
    jt3.append(joint_torque[2])
    jt4.append(joint_torque[3])
    jt5.append(joint_torque[4])
    jt6.append(joint_torque[5])

    inst_trans = final_trans.subs(theta_1, theta_inst[0]).subs(theta_2, theta_inst[1]).subs(theta_3, 0).subs(theta_4, theta_inst[2]).subs(theta_5, theta_inst[3]).subs(theta_6, theta_inst[4]).subs(theta_7, theta_inst[5])

plt.title('Time vs tau(1-6) graph')

plt.subplot(3, 2, 1)
plt.scatter(range(0,200), jt1)

plt.subplot(3, 2, 2)
plt.scatter(range(0,200), jt2)

plt.subplot(3, 2, 3)
plt.scatter(range(0,200), jt3)

plt.subplot(3, 2, 4)
plt.scatter(range(0,200), jt4)

plt.subplot(3, 2, 5)
plt.scatter(range(0,200), jt5)

plt.subplot(3, 2, 6)
plt.scatter(range(0,200), jt6)

plt.show()
