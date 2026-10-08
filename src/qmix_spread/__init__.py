"""QMIX (Rashid et al., 2018) for the Simple Spread environment from mpe2.

Modules, in the order data flows through them:
    config      - all settings of a run
    env         - Simple Spread as numpy arrays (obs [n, O], state [S], one team reward)
    networks    - AgentNetwork (computes Q_i) and QMixer (mixes Q_i into Q_tot)
    rollout     - playing episodes with the QMIX policy or a random policy
    buffer      - replay buffer of whole episodes
    learner     - TD target, TD error, loss and the gradient step
    train       - the training loop and saving of all outputs
    checkpoint  - saving / loading a trained model
    td_example  - the step-by-step numeric TD example
    plotting    - figures
"""
