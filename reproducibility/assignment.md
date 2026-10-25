# Laboratory Assignment


## 1. Objective

Investigate how the dynamics of a model neuron change as the external input varies, and compare three approaches: numerical bifurcation analysis, estimation of attractor dimensions from a time series, and the entropy–complexity plane method.

**Research question:** can a membrane potential recording distinguish between regular activity, deterministic chaos, and regular activity with measurement noise?

The required part uses the Hindmarsh–Rose model. The optional part compares the Hodgkin–Huxley and Izhikevich models.

## 2. Model and Experimental Setup

Consider the Hindmarsh–Rose model [1, 2]:

$$
\begin{aligned}
\dot x &= y-ax^3+bx^2-z+I,\\
\dot y &= c-dx^2-y,\\
\dot z &= r\bigl(s(x-x_R)-z\bigr).
\end{aligned}
$$

Here, $x$ is the membrane potential variable, $y$ is the fast recovery variable, $z$ is the slow adaptation variable, and $I$ is a constant external input. Time and all variables are expressed in model units.

Fix the parameters:

$$
a=1,\qquad b=3,\qquad c=1,\qquad d=5,
$$

$$
r=0.006,\qquad s=4,\qquad x_R=-1.6.
$$

Investigate the range

$$
I\in[0,4].
$$

Initial conditions for independent runs:

$$
(x(0),y(0),z(0))=(-1.6,-12,0),
$$

$$
(x(0),y(0),z(0))=(1,0,0).
$$

Choose the integration method, numerical accuracy, parameter grid, transient duration, and observation duration independently. Justify their adequacy through convergence checks. Save all model variables and computational settings; for time-series analysis, use uniformly sampled recordings of $x(t)$ after the transient.

## 3. Tasks

### Part A. Numerical Bifurcation Analysis

#### A1. Equilibria

Find all equilibria of the system over the specified range of $I$. Justify their number and plot the equilibrium branch or branches. Provide the equations used to locate equilibria, describe the numerical method, and report residual checks.

#### A2. Stability and Critical Points

Investigate the stability of the equilibria. Locate and refine at least one critical parameter value to an accuracy of $10^{-6}$ or better in $I$. State the suspected bifurcation type and the conditions verified numerically.

Distinguish stable and unstable segments in the branch diagram. Show how the spectrum of the linearization changes near the identified critical point.

#### A3. Oscillatory Regimes

Plot diagrams of local membrane potential maxima and interspike intervals as functions of $I$. Register a spike when $x$ crosses zero in the positive direction.

Perform forward and backward parameter sweeps, as well as independent runs from the specified initial conditions. Investigate the transitions you identify and compare them with the results of A1–A2.

#### A4. Classification of Dynamics

Select at least three time-dependent regimes: regular spiking, periodic bursting, and suspected chaotic activity. Also retain a stable steady-state regime.

For the time-dependent regimes, present time series, phase portraits, and interspike-interval return plots. Compute the largest Lyapunov exponent from the model equations and investigate the convergence of its estimate. Justify the classification of the regimes independently of the entropy–complexity method.

Use these same regimes in Parts B and C.

### Part B. Attractor Reconstruction and Dimensions

In this part, use only the scalar time series $x(t)$ from the selected time-dependent regimes as input data.

#### B1. Reconstruction

Reconstruct the attractors using time-delay vectors. Compare delay selection based on the first zero of the autocorrelation function with selection based on the first local minimum of mutual information [7]. Investigate the embedding dimension using the false nearest neighbors method [6].

Justify the selected parameters and the data preprocessing procedure. Present diagnostic plots and projections of the reconstructed attractors.

#### B2. Correlation Dimension

Estimate the correlation dimension $D_2$ using the Grassberger–Procaccia method [3]. Account for temporal dependence between points using a Theiler window [5].

Present correlation sums, local slopes, and the selected scaling regions. Investigate the sensitivity of the estimates to the embedding dimension, delay, record length, and Theiler window width.

#### B3. Capacity Dimension

For one periodic regime and one suspected chaotic regime, estimate the capacity dimension $D_0$ using box-counting [4].

Show how the number of occupied boxes depends on scale and justify the estimation ranges. Investigate the effects of record length, embedding dimension, and grid position. Compare the results with the estimates of $D_2$.

#### B4. Interpretation

Compare the geometry of the reconstructed attractors and the estimated dimensions with the classification from Part A. State which estimates were obtained reliably and justify this assessment. Characterize the limit set of the steady-state regime separately.

### Part C. The Entropy–Complexity Plane

#### C1. Control Signals

Supplement the three time-dependent neuronal signals with control recordings of the same length and sampling interval:

- Gaussian white noise;
- a randomly shuffled copy of one neuronal time series;
- the signal

$$
s_{\mathrm{qp}}(t)=
\sin\left(\frac{2\pi t}{T_0}\right)
+0.5\sin\left(\frac{2\pi\sqrt2\,t}{T_0}\right),
$$

where $T_0$ is the full period of one of the selected regular regimes.

#### C2. Entropy and Statistical Complexity

Construct the Bandt–Pompe ordinal pattern distribution [8]. Denote the pattern length by $\ell$, the delay between its elements by $q$, the pattern distribution by $P$, and the number of possible patterns by $K=\ell!$.

For the entropy–complexity plane, use [9]:

$$
H(P)=\frac{S(P)}{\ln K},
\qquad
C(P)=H(P)\frac{J(P,P_e)}{J_{\max}(K)},
$$

where

$$
S(P)=-\sum_{k=1}^{K}p_k\ln p_k,
\qquad
P_e=(1/K,\ldots,1/K),
$$

$$
J(P,P_e)=
S\left(\frac{P+P_e}{2}\right)
-\frac12S(P)-\frac12S(P_e),
$$

and $J_{\max}(K)$ is the maximum of $J(P,P_e)$ over distributions $P$ for fixed $K$.

Implement the computation of $H$ and $C$, and justify the normalization and the handling of tied values. Verify the implementation using tests of your own choice.

#### C3. Comparison of Regimes

Plot the neuronal and control signals on the $(H,C)$ plane using the same analysis parameters. Assess the variability of the results across consecutive segments of the recordings. Compare the positions of the points with the classification from Part A.

Select a region of the parameter diagram containing a change of regime. For at least 15 values of $I$, plot $H(I)$, $C(I)$, and the corresponding path in the $(H,C)$ plane. Compare these with the bifurcation results.

#### C4. Analysis Parameters

Investigate the effects of pattern length $\ell\in\{4,5\}$, delay $q$, sampling interval, and time-series length [10]. Justify the adequacy of the number of observed patterns.

State which settings distinguish the regimes under study and how robust this result is.

#### C5. Measurement Noise

After solving the ODEs, add noise to one regular and one chaotic signal:

$$
s_n^{\mathrm{obs}}=s_n+\eta\sigma_s\xi_n,
\qquad \xi_n\sim\mathcal N(0,1),
$$

$$
\eta\in\{0,0.01,0.05,0.1,0.2\},
$$

where $\sigma_s$ is the standard deviation of the noise-free signal. Use at least five noise realizations for each nonzero noise level.

Investigate how the positions of the signals in the $(H,C)$ plane change as $\eta$ increases, using at least two sets of ordinal analysis parameters. For one nonzero noise level, repeat the estimation of $D_2$ for the regular signal.

Assess the effect of measurement noise on distinguishing the regimes.

### Part D. Overall Conclusion

Prepare a summary table including signal origin, the parameter $I$, the independent classification, the largest Lyapunov exponent, reconstruction parameters, estimates of $D_2$ and $D_0$, and values of $H$ and $C$. Indicate where a characteristic is not applicable or an estimate is unreliable.

Answer the research question. Compare the conclusions drawn from the three methods, explain any discrepancies, and state the limitations of the results.

## 4. Submission Requirements

Submit a reproducible computational study and a report. A single Jupyter Notebook containing code, equations, results, and conclusions is acceptable. For separate submissions, include the source code and a report in Markdown or PDF.

Include experimental settings, dependency versions, key tables, and figures. Specify the numerical methods and data processing rules used. Randomized experiments must be reproducible from saved initial states of the random number generator.

Label plot axes, units, and relevant parameters. Support your conclusions with computational results and numerical reliability checks.


## 5. Optional Study: Comparison of Neuron Models

**Research question:** which characteristics of neuronal activity are preserved when the Hodgkin–Huxley model is replaced by the Izhikevich model?

Use the Hodgkin–Huxley model as a synthetic reference for the comparison.

### 5.1. Hodgkin–Huxley Model

Use the spatially homogeneous model [11, 15]:

$$
C_m\dot V=I_{\mathrm{HH}}(t)
-\overline g_{\mathrm{Na}}m_g^3h_g(V-E_{\mathrm{Na}})
-\overline g_{\mathrm K}n_g^4(V-E_{\mathrm K})
-g_L(V-E_L),
$$

$$
\dot q_g=\alpha_q(V)(1-q_g)-\beta_q(V)q_g,
\qquad q_g\in\{m_g,h_g,n_g\}.
$$

Transition rates:

$$
\begin{aligned}
\alpha_m(V)&=\frac{0.1(V+40)}{1-e^{-(V+40)/10}},
&\beta_m(V)&=4e^{-(V+65)/18},\\
\alpha_h(V)&=0.07e^{-(V+65)/20},
&\beta_h(V)&=\frac{1}{1+e^{-(V+35)/10}},\\
\alpha_n(V)&=\frac{0.01(V+55)}{1-e^{-(V+55)/10}},
&\beta_n(V)&=0.125e^{-(V+65)/80}.
\end{aligned}
$$

Parameters:

$$
C_m=1,\quad
\overline g_{\mathrm{Na}}=120,\quad
\overline g_{\mathrm K}=36,\quad
g_L=0.3,
$$

$$
E_{\mathrm{Na}}=50,\quad E_{\mathrm K}=-77,\quad E_L=-54.387.
$$

Time is measured in ms, potential in mV, capacitance per unit area in µF/cm$^2$, conductances in mS/cm$^2$, and external current density in µA/cm$^2$. No temperature factor is introduced.

Initial conditions:

$$
V(0)=-65,\qquad
q_g(0)=\frac{\alpha_q(-65)}{\alpha_q(-65)+\beta_q(-65)}.
$$

### 5.2. Izhikevich Model

Use the system [12]:

$$
\dot v=0.04v^2+5v+140-u+I_{\mathrm{Izh}}(t),
$$

$$
\dot u=a_I(b_Iv-u),
$$

with a reset when $v$ reaches $30$ mV:

$$
v\leftarrow c_I,\qquad u\leftarrow u+d_I.
$$

Parameters and initial conditions:

$$
a_I=0.02,\qquad b_I=0.2,\qquad c_I=-65,\qquad d_I=8,
$$

$$
v(0)=-65,\qquad u(0)=b_Iv(0).
$$

Time is measured in ms. Matching the numerical scales of the inputs is part of the assignment.

### 5.3. Tasks

**1. Calibration with Constant Input.** Introduce a common signal $J$ and set

$$
I_{\mathrm{HH}}=J,\qquad
I_{\mathrm{Izh}}=\kappa J+\beta,\qquad \kappa>0.
$$

Fit $\kappa$ and $\beta$ using the firing rate–current characteristics at

$$
J\in\{0,5,8,10,15,20\}.
$$

Use $J\in\{7,9,12,18\}$ only for validation. Choose the matching criterion independently. Evaluate the errors in reproducing the firing rate and the threshold for spike generation. After calibration, keep the model parameters and the input transformation fixed.

**2. Periodic Input.** Investigate responses to the signal

$$
J(t)=J_0+A\sin\left(\frac{2\pi t}{T}\right),
$$

$$
I_{\mathrm{HH}}(t)=J(t),\qquad
I_{\mathrm{Izh}}(t)=\kappa J(t)+\beta.
$$

Initial exploration range:

$$
J_0=10,\qquad A\in[0,15],\qquad T\in\{10,20,40\}\ \text{ms}.
$$

For one forcing period, carry out a detailed analysis as the amplitude varies. Compare interspike intervals, stroboscopic diagrams, synchronization regimes, and transitions between them [13]. Refine at least one periodic solution and investigate its stability. Describe how the reset in the Izhikevich model is handled during integration and stability analysis [14].

**3. Time-Series Analysis.** For selected responses of both models, repeat the reconstruction, dimension estimation, and entropy–complexity plane analysis. Additionally, investigate the ordinal structure of the interspike intervals. Justify the comparability of the observations and analysis parameters.

**4. Conclusion.** Determine which properties of the detailed model are reproduced by the Izhikevich model within the investigated range. State the domain of applicability of the simplification and identify the characteristics that reveal differences.
