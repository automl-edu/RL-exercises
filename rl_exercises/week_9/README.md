# Week 9: Model-Based RL

This week you will combine model-based imagination with on-policy updates by completing the implementation in **dyna_ppo.py** and performing multiple experiments using it.

Throughout, use [RLiable](https://github.com/google-research/rliable) for plotting performance curves and statistical comparisons. Feel free to leverage [HyperSweeper](https://github.com/automl/hypersweeper) for hyperparameter tuning if you wish.

---

## Level 1: Core Dyna-PPO vs PPO

Complete the TODOs in **dyna_ppo.py** and then verify whether adding a learned model improves sample efficiency. Then try to understand when does the model helps and when it hurts.

### 1.1 Sample Efficiency
- **Experiment:**  
  - Train with `use_model=True` (Dyna-PPO) and `use_model=False` (plain PPO).  
  - Keep the same **real env steps** budget (e.g. 15 k).  
  - Plot avg return vs real steps  
  - **Questions:**  
    1. How many steps sooner does Dyna-PPO reach 80 % of the final PPO return?  
    2. Is there an initial “model learning penalty” (Dyna underperforms early)?

### 1.2 Model Prediction Accuracy
- **Experiment:**  
  - At checkpoints (e.g. every 10 k steps), evaluate your dynamics model on held-out transitions.  
  - Measure **one-step MSE** and **multi-step error** $Eₖ$ for $k=1…20$.  
  - Plot one-step MSE vs real steps.  
  - Plot $Eₖ$ curves at early & late stages.  
  - **Questions:**  
    1. Around what MSE threshold does Dyna-PPO begin to outperform PPO?  
    2. How fast does $Eₖ$ grow, and how does that inform your choice of `imag_horizon`?

---

## Level 2: Advanced Analyses & Ablations

In **Level 2**, you will dig deeper into the design decisions: hyperparameters, distribution shift, and environment effects.

### 2.1 Hyperparameter Sensitivity
- **A. Imagination Horizon**  
  Test `imag_horizon ∈ {1,3,5,10,20}`.  
  - Plot final return vs horizon.  
  - Try to charecterize the trade-off between planning depth and compounding model error.
- **B. Model/Imagination Budget**  
  Compare three regimes using return vs real-steps curves
  
  | Regime       | model_epochs | imag_batches |
  |--------------|--------------|--------------|
  | Conservative | 1            | 5            |
  | Balanced     | 3            | 10           |
  | Aggressive   | 5            | 20           |
  
  - Question: which regime gives the best sample efficiency?
- **C. Buffer Size**  
  Sweep `max_buffer_size ∈ {1 000, 5 000, 10 000, 50 000}`.  
  - Plot model MSE and policy return vs buffer size.  
  - Question: does too much stale data hurt?

### 2.2 Distribution Shift
- **Experiment:**  
  - Track state-visit histograms at early, mid, late training.  
  - Compute model error on “old” vs “new” states.  
- **Questions:**  
  1. Does model accuracy degrade on fresh states?  
  2. When does model cost outweigh benefit?

### 2.3 Failure-Mode Analysis
- **Experiment:**  
  - Introduce artificial noise to your dynamics model:
    ```python
    def corrupt_model(delta, r, σ):
        return delta + torch.randn_like(delta)*σ, r + torch.randn_like(r)*σ
    ```
    - Plot the fial perofrmance against the introduced noise
    - How does the model perfromance change with this added noise?

## Level 3: Uncertainty‐Aware Ensembles & Advanced Planning (Stretch)

You’ll build on your Dyna-PPO work to incorporate **model uncertainty** and even switch to planning with your learned models.

### 3.1 Ensemble Dynamics & Epistemic Uncertainty  
- **Build an ensemble** of $M$ dynamics models (e.g. $M\in\{3,5,10\}$), each trained on the same replay buffer but with different random inits.  
- **At each imagined rollout step**, compute the **mean** and **variance** across the $M$ predicted $\Delta s$ and $\hat r$.  
- **Plot** the variance as a function of real-env steps, and correlate high-variance regions with model errors.

### 3.2 Uncertainty‐Gated Imagination  
- **Design a gating rule**: only use imagined transitions whose **predicted variance** is below a threshold $\tau$.  
- **Sweep** $\tau\in\{0.01,0.05,0.1,0.2\}$:  
  - How does gating affect sample efficiency?  
  - Does it stabilize learning when your single-model Dyna begins to diverge?

### 3.3 CEM‐Based Model Predictive Control  
- **Implement CEM‐MPC** using your ensemble:  
  1. Sample $K$ action sequences of length $H$.  
  2. Roll out in each ensemble member, compute **worst-case** cumulative reward (or penalize high-variance trajectories).  
  3. Select the first action of the best (robust) sequence and repeat.  
- **Compare**:  
  - **Dyna-PPO** (policy updates via imagined rollouts)  
  - **Ensemble MPC** (online planning every real step)  
  on your chosen env.
