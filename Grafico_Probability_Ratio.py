import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from scipy.signal import find_peaks

# Graph settings function
def multicolor_label(ax, x, y, texts,size):
    for dx, (text, color) in enumerate(texts):
        ax.text(x + dx * 0.07, y, text, color=color,
                fontsize=size, ha='center', va='top')
                
# Define common parameters
def params_evaluation(phi):
    G = 1.0
    delta_0 = 0.0
    epsilon = 1e-15
    # Define cosine/sine trigonometric terms
    cos_phi = np.cos(phi)
    cos_2phi = np.cos(2*phi)
    cos2_phi = np.cos(phi)**2
    sin_phi = np.sin(phi)
    sin_2phi = np.sin(2*phi)
    sin2_phi = np.sin(phi)**2
    # Define Delta Terms
    const = 4 * np.pi * G**2
    delta = 1/3*const * sin_2phi/2
    delta_s = 4/3*const * sin_phi/2
    # Define decay rates
    gamma_minus = const * (1 - cos_2phi)
    gamma_1plus = const * (1 + (4/3) * cos_phi + (1/3) * cos_2phi)
    gamma_2plus = const * (1 - (4/3) * cos_phi + (2/3) * cos_2phi)
    gamma = const
    # Define Lambda terms
    X = 9*delta**2 + 9/2*delta_s**2-(gamma_1plus+gamma_2plus)**2/4
    Y = (delta-2*delta_s)*(gamma_1plus-gamma_2plus)-(np.sqrt(2))*(delta_s+4*delta)*np.sign(cos_phi+cos_2phi)*np.sqrt(gamma_1plus*gamma_2plus)
    sqrt_inner1 = (np.sqrt(X**2 + Y**2) - X)/2
    sqrt_inner2 = (np.sqrt(X**2 + Y**2) + X)/2
    # Protect against negative values in sqrt due to numerical errors
    sqrt_inner1[sqrt_inner1 < 0] = 0
    sqrt_inner2[sqrt_inner2 < 0] = 0
    v = np.sign(Y) * np.sqrt(sqrt_inner1)
    u = np.sqrt(sqrt_inner2)
    # Define Lambda_plus terms
    a = delta_0+3/2*delta+1/2*u
    b = v/2 - (gamma_1plus+gamma_2plus)/4
    lambda_plus = a+b*1j
    # Define Lambda_minus terms
    c = delta_0+3/2*delta-1/2*u
    d = v/2 + (gamma_1plus+gamma_2plus)/4
    lambda_minus = c-d*1j
    X_R = ( (np.sqrt(2)*delta_s/4) + (np.sqrt(2)*delta) - (1j/2)*np.sign(cos_phi+cos_2phi)*np.sqrt(gamma_1plus*gamma_2plus) )
    Y_R_plus = ( lambda_plus - (delta_0 + delta_s + delta - (1j/2)*gamma_1plus) )
    Y_R_minus = ( lambda_minus - (delta_0 + delta_s + delta - (1j/2)*gamma_1plus) )
    N_plus = np.sqrt(X_R**2+Y_R_plus**2)
    N_minus = np.sqrt(X_R**2+Y_R_minus**2)
    N_plus = np.where(N_plus == 0, epsilon, N_plus)
    N_minus = np.where(N_minus == 0, epsilon, N_minus)
    # The A, B, C, and D amplitudes represent the components of the normalized left eigenvectors of the self-energy matrix
    A = X_R / N_plus
    B = Y_R_plus / N_plus
    C = X_R / N_minus
    D = Y_R_minus / N_minus
    # Integrals
    integrals = {
    'integral_F_minus_msq' : (4*math.pi**2)/(gamma*gamma_minus + epsilon),
    'integral_F_1plus_msq' : (4*math.pi**2)/(gamma*((gamma_1plus+gamma_2plus)/2-v) + epsilon),
    'integral_F_2plus_msq' : (4*math.pi**2)/(gamma*((gamma_1plus+gamma_2plus)/2+v) + epsilon),
    'integral_F_minus_F_1plus' : (4*math.pi**2*1j)/(gamma*((a+3*delta)+1j*(gamma_minus/2-b)) + epsilon),
    'integral_F_minus_F_2plus' : (4*math.pi**2*1j)/(gamma*((c+3*delta)+1j*(gamma_minus/2+d)) + epsilon),
    'integral_F_1plus_F_2plus' : (4*math.pi**2*1j)/(gamma*((c-a)+1j*(d-b)) + epsilon)}
    
    return G, A, B, C, D, integrals
    
    
# Define bunching_probability evaluation function
def bunching_probability(params):
    G, A, B, C, D, integrals = params
    # Define exponentials terms
    exp_i_phi = np.exp(1j * phi)
    exp_2i_phi = np.exp(2j * phi)
    
    # Define Amplitude A (Antisymmetric)
    Amplitude_A = -0.5 * (1 - exp_2i_phi)**2
    
    # Define Amplitude S1 (Symmetric 1)
    term1_symmetric1 = (A**2 / 6) * (1 + exp_i_phi)**2 * (1 + exp_2i_phi)
    term2_symmetric1 = (A * B) / (3 * np.sqrt(2)) * (2 + exp_i_phi + 2 * exp_2i_phi) * (1 + exp_2i_phi)
    term3_symmetric1 = (B**2 / 3) * (1 - exp_i_phi + exp_2i_phi) * (1 + exp_2i_phi)
    Amplitude_S1 = term1_symmetric1 + term2_symmetric1 + term3_symmetric1
    
    #Define Amplitude S2 (Symmetric 2)
    term1_symmetric2 = (C**2 / 6) * (1 + exp_i_phi)**2 * (1 + exp_2i_phi)
    term2_symmetric2 = (C * D )/ (3 * np.sqrt(2)) * (2 + exp_i_phi + 2 * exp_2i_phi) * (1 + exp_2i_phi)
    term3_symmetric2 = (D**2 / 3) * (1 - exp_i_phi + exp_2i_phi) * (1 + exp_2i_phi)
    Amplitude_S2 = term1_symmetric2 + term2_symmetric2 + term3_symmetric2
   
    # Define Pure terms
    term_A_pure = np.abs(Amplitude_A)**2 * integrals ['integral_F_minus_msq']
    term_S1_pure = np.abs(Amplitude_S1)**2 * integrals ['integral_F_1plus_msq']
    term_S2_pure = np.abs(Amplitude_S2)**2 * integrals ['integral_F_2plus_msq']
    
    # Define Interference terms
    interference_A_S1 = 2 * np.real(Amplitude_A * np.conj(Amplitude_S1) * integrals ['integral_F_minus_F_1plus'])
    interference_A_S2 = 2 * np.real(Amplitude_A * np.conj(Amplitude_S2) * integrals ['integral_F_minus_F_2plus'])
    interference_S1_S2 = 2 * np.real(Amplitude_S1 * np.conj(Amplitude_S2) * integrals ['integral_F_1plus_F_2plus'])
    
    # Define Bunching Probability
    P_RR_total = (term_A_pure + term_S1_pure + term_S2_pure + interference_A_S1 + interference_A_S2 + interference_S1_S2)
    
    return G**4 * P_RR_total
    

# Define antibunching_probability evaluation function
def antibunching_probability(params):
    G, A, B, C, D, integrals = params
    # Define Exponential terms
    exp_i_phi = np.exp(1j * phi)
    exp_2i_phi = np.exp(2j * phi)
    exp_2i_phi_m = np.exp(-2j * phi)
    
    # Define Amplitude A (Antisymmetric)
    Amplitude_A = -0.5 * (1 - exp_2i_phi)**2
    
    # Define Amplitude S1 (Symmetric 1)
    term1_symmetric1 = (A**2 / 6) * (1 + exp_i_phi)**2 * (1 + exp_2i_phi_m)
    term2_symmetric1 = (A * B) / (3 * np.sqrt(2)) * (2 + exp_i_phi + 2 * exp_2i_phi) * (1 + exp_2i_phi_m)
    term3_symmetric1 = (B**2 / 3) * (1 - exp_i_phi + exp_2i_phi) * (1 + exp_2i_phi_m)
    Amplitude_S1 = term1_symmetric1 + term2_symmetric1 + term3_symmetric1
    
    #Define Amplitude S2 (Symmetric 2)
    term1_symmetric2 = (C**2 / 6) * (1 + exp_i_phi)**2 * (1 + exp_2i_phi_m)
    term2_symmetric2 = (C * D )/ (3 * np.sqrt(2)) * (2 + exp_i_phi + 2 * exp_2i_phi) * (1 + exp_2i_phi_m)
    term3_symmetric2 = (D**2 / 3) * (1 - exp_i_phi + exp_2i_phi) * (1 + exp_2i_phi_m)
    Amplitude_S2 = term1_symmetric2 + term2_symmetric2 + term3_symmetric2
   
    # Define Pure terms
    term_A_pure = np.abs(Amplitude_A)**2 * integrals ['integral_F_minus_msq']
    term_S1_pure = np.abs(Amplitude_S1)**2 * integrals ['integral_F_1plus_msq']
    term_S2_pure = np.abs(Amplitude_S2)**2 * integrals ['integral_F_2plus_msq']
    
    # Define Interference terms
    interference_A_S1 = 2 * np.real(Amplitude_A * np.conj(Amplitude_S1) * integrals ['integral_F_minus_F_1plus'])
    interference_A_S2 = 2 * np.real(Amplitude_A * np.conj(Amplitude_S2) * integrals ['integral_F_minus_F_2plus'])
    interference_S1_S2 = 2 * np.real(Amplitude_S1 * np.conj(Amplitude_S2) * integrals ['integral_F_1plus_F_2plus'])
    
    # Define Bunching Probability
    P_RL_total = (term_A_pure + term_S1_pure + term_S2_pure + interference_A_S1 + interference_A_S2 + interference_S1_S2)
    
    return G**4 * P_RL_total
    

# Define Probability_ratio evaluation function
def probability_ratio(A_RR,A_RL):
    beta=A_RR/A_RL
    return beta

# Evaluation
phi = np.linspace(0, 2 * np.pi, 700)
params = params_evaluation(phi)
P_bunch  = bunching_probability(params)
P_anti = antibunching_probability(params)
beta = probability_ratio(P_bunch,P_anti)



P_anti[-1]=P_anti[-2]
P_anti[0]=P_anti[1]
P_bunch[-1]=P_bunch[-2]
P_bunch[0]=P_bunch[1]




# Define RGB colors
Slate_Blue = (80/255, 90/255, 110/255)
Medium_Blue = (0.12, 0.43, 0.86)
Periwinkle_Blue = (0.0, 0.27, 0.88)
Mint_Green = (0.04, 0.47, 0.08)
Dark_Teal = (0.43, 0.63, 0.20)
Turquoise = (0.43, 0.90, 0.08)



# Firs Plot creation
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(phi, beta, label=r'$\beta$',color='red')
ax.axhline(1, color='black', linestyle='--', linewidth=1, label=r'$\beta=1$')

ax.legend(fontsize=12)

#Axs settings
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_clip_on(False)
ax.spines['bottom'].set_clip_on(False)
ax.xaxis.set_label_coords(1.05, 0.5)
ax.set_ylim(-0.0, 3.5)

ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
 
multicolor_label(ax, 6.9, 0.15, [(r'$\varphi(\omega_0)$','black')],16)
multicolor_label(ax, 0., 3.7, [(r'$\beta$', 'black')], size=16)

tick_positions = [0, np.pi/2, np.pi, 3*np.pi /2, 2*np.pi]
tick_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
ax.set_xticks(tick_positions, tick_labels)
ax.xaxis.set_minor_locator(MultipleLocator(0.25*3.14))
major_ticks_y = [1,2,3]
ax.set_yticks(major_ticks_y)
ax.yaxis.set_minor_locator(MultipleLocator(0.5*1))

# Second plot creation
fig1, bx = plt.subplots(figsize=(12, 7))
bx.plot(phi, P_bunch, label=r'$P_{bunch}$', color = Medium_Blue)
bx.plot(phi, P_anti, label=r'$P_{anti}$', color = Dark_Teal)
bx.plot(phi,P_bunch+P_anti,label =r'$P_{bunch}+P_{anti}$', linestyle='--', color = 'black')
bx.legend(fontsize=12)

# Axs settings
bx.spines['left'].set_position('zero')
bx.spines['bottom'].set_position('zero')
bx.spines['right'].set_color('none')
bx.spines['top'].set_color('none')
bx.spines['left'].set_clip_on(False)
bx.spines['bottom'].set_clip_on(False)
bx.xaxis.set_label_coords(1.05, 0.5)
bx.set_xlim(0.,2*np.pi)
bx.set_ylim(0.0, 1.15)
bx.plot(0, 1, "^k", transform=bx.get_xaxis_transform(), clip_on=False)
bx.plot(1, 0, ">k", transform=bx.get_yaxis_transform(), clip_on=False)

multicolor_label(bx, 6.6, 0.04, [(r'$\varphi(\omega_0)$','black')],16)
multicolor_label(bx, 0., 1.25, [(r'$P_{bunch}\left(P_{anti}\right)$', 'black')], size=16)

tick_positions = [0, np.pi/2, np.pi, 3*np.pi /2, 2*np.pi]
tick_labels = [ r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
bx.set_xticks(tick_positions, tick_labels)
bx.xaxis.set_minor_locator(MultipleLocator(0.25*3.14))
major_ticks_y = [0.5,1]
bx.set_yticks(major_ticks_y)
bx.yaxis.set_minor_locator(MultipleLocator(0.1*1))

# Plots show
plt.show()

