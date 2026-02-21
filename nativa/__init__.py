"""
NATIVA — Core Library
======================
Réseau neuromorphique pour la détection d'anomalies vibratoires.

Composants :
    - nativa_network : réseau unifié (mode industriel)
    - neuron : neurones LIF avec seuils adaptatifs
    - stdp : apprentissage STDP modulé par Free Energy
    - free_energy : inférence Active (ELBO)
    - kuramoto : synchronisation de phase Kuramoto-Sakaguchi
    - event_encoder : encodeurs spike (rate, temporal, delta)
    - integrators : intégrateurs numériques (RK4, Euler)
    - delay_buffer : buffers temporels circulaires
"""
