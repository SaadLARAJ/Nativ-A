# NATIVA

Un réseau de neurones à spikes pour la détection d'anomalies vibratoires en maintenance prédictive.

![Résultats NATIVA sur CWRU — Distribution des scores de surprise](results/cwru_report.png)

> **AUC-ROC 0.997** en détection non-supervisée sur le dataset standard CWRU.
> Aucun exemple de défaut vu pendant l'entraînement.

---

## Ce que c'est

NATIVA est un réseau neuromorphique qui apprend à quoi ressemble un moteur "sain" — sans jamais voir un défaut — puis détecte toute déviation. Pas de labels, pas de GPU, juste des spikes et de la physique.

L'architecture combine quatre briques :
- **Neurones LIF** (Leaky Integrate-and-Fire) avec seuils adaptatifs
- **STDP** (Spike-Timing-Dependent Plasticity) pour l'apprentissage non-supervisé
- **Oscillateurs de Kuramoto** pour la synchronisation de phase entre neurones
- **Inférence Active** (Free Energy) pour scorer l'anomalie

Le principe : un signal de vibration sain produit un motif de spikes stable. Quand le roulement se dégrade, le motif change, la "surprise" du réseau augmente, et on déclenche une alerte.

## Comment ça marche

```
  Signal vibratoire (accéléromètre)
         │
         ▼
  ┌──────────────────────┐
  │  STFT → 8 bandes de  │   Encodage multi-bande :
  │  fréquence → spikes  │   le spectre est découpé en sous-bandes,
  │  (norm. globale)      │   normalisé sur la baseline "saine"
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │  100 neurones LIF    │   Le réseau traite les spikes
  │  + STDP + Kuramoto   │   pas de temps par pas de temps
  │  + Free Energy       │
  └──────────┬───────────┘
             ▼
  Score de surprise (Free Energy)
  Si surprise > seuil → ALERTE
```

## Résultats

Benchmarké sur le dataset CWRU (Case Western Reserve University) — le standard industriel pour les roulements.

### Single-condition (0 HP, 0.007")

| Métrique | NATIVA (non-supervisé) | Random Forest (supervisé) |
|----------|:----------------------:|:-------------------------:|
| AUC-ROC | 0.997 | 1.000 |
| F1 | 0.992 | 1.000 |
| Recall | 0.989 | 1.000 |
| Faux Négatifs | 8 | 0 |

Le RF est supervisé (il voit les labels). NATIVA n'a **jamais vu un défaut**. 99.7% d'AUC en non-supervisé, c'est un résultat solide.

### Multi-condition (36 conditions)

3 types de défaut × 3 tailles × 4 charges moteur :

| Charge moteur | AUC moyen |
|:-------------:|:---------:|
| 0 HP | 0.999 |
| 1 HP | 0.823 |
| 2 HP | 0.985 |
| 3 HP | 0.999 |

**Inner Race** est le plus fiable (AUC 0.985 ± 0.05). **Ball** et **Outer Race** sont solides sauf à 1HP.

![Heatmap AUC-ROC par type de défaut, taille et charge moteur](results/cwru_multi_report.png)

### Limitation connue : 1 HP

À 1HP (1772 RPM), 3 des 9 conditions sont faibles (AUC < 0.77). L'encodeur multi-bande utilise des bandes linéaires de 750 Hz : les fréquences de défaut (BPFO = 105 Hz, BPFI = 160 Hz) tombent dans la même bande que l'énergie normale de fonctionnement. L'encodeur est "harmonic-dependent" (il détecte via les résonances haute fréquence) et non "kinematic-dependent" (il ne résout pas les fréquences cinématiques individuelles). Des bandes logarithmiques (Mel-scale) corrigeraient ce point.

Le AUC moyen sur les 36 conditions est **0.95**. En excluant le 1HP, il monte à **0.99**.

### Ce qui explique le gap : l'encodeur, pas le SNN

On a testé NATIVA avec 3 baselines non-supervisées (Autoencoder, Isolation Forest, OC-SVM) sur les mêmes conditions. Résultat :

| Méthode | Input | Mean AUC |
|:--------|:------|:--------:|
| Autoencoder / IF / OC-SVM | 22 features FFT (float) | **1.000** |
| NATIVA (8 bandes → spikes) | 8 bandes STFT (binaire) | 0.951 |
| **NATIVA (22 features → spikes)** | **22 features FFT (rate-coded)** | **0.995** |

![Comparaison encodage vs apprentissage](results/feature_spike_report.png)

**Le SNN n'est pas le problème.** Quand on donne les mêmes features (en spikes) à NATIVA, il fait 0.995 — le gap passe de 5% à 0.5%. Le 1HP passe de 0.82 à 0.995. Le bottleneck, c'est l'encodeur 8-bandes linéaires, pas l'algorithme d'apprentissage.

Les 0.5% restants = le coût du spike : rate coding ajoute du bruit stochastique que le floating-point n'a pas. C'est le **trade-off SWaP** (Size, Weight, and Power) : 0.5% de précision contre une architecture compatible avec des puces neuromorphiques en microwatts.

## Journal d'Architecture : Le chemin de l'échec à la rupture

L'Open Source ne doit pas cacher les impasses. NATIVA ne s'est pas fait en un jour. Voici comment nous sommes passés d'un AUC catastrophique de 0.50 à une architecture neuromorphique capable d'assassiner l'analyse spectrale classique FFT.

### 1. La naïveté initiale (L'échec de la normalisation)
Les premiers essais sur CWRU donnaient un score de **0.50 (le hasard pur)**. Le SNN apprenait, la Free Energy fonctionnait, mais le réseau était aveugle. Pourquoi ?
Parce que nous appliquions une normalisation Min-Max standard *sur chaque fenêtre temporelle*. Cela écrasait complètement les variations d'amplitude entre une machine saine et une machine dont le roulement s'écaille. 
**La solution :** La *Normalisation Globale*. Nous calons désormais l'encodeur sur le 99ème centile du dataset sain complet. Résultat immédiat : l'AUC a bondi à **0.997**.

### 2. Le mur Paderborn et le piège des algorithmes Magiques
Forts d'un score parfait sur CWRU (chocs francs, haute fréquence), nous avons testé NATIVA sur le dataset Paderborn (usure abrasive, frottements basse fréquence). 
**Résultat : L'effondrement (AUC 0.50).**
Nous avons alors essayé l'échelle Mel (utilisée en reconnaissance vocale) pour tenter de sauver les basses fréquences. **Erreur stratégique :** L'échelle Mel écrase les hautes fréquences, détruisant nos scores sur CWRU (chute à 0.69) tout en échouant sur Paderborn. 
*La Leçon : Il n'y a pas d'encodeur universel.* L'IA vibratoire doit respecter la physique de la panne.

### 3. La Famine Compétitive (L'échec de la fusion naïve)
Pour contrer cela, nous avons tenté un *Dual-Encoder* : injecter à la fois le bruit brut (Paderborn) et l'Enveloppe temporelle HF (CWRU) dans le même réseau SNN.
**Le crash :** Le bruit continu a noyé le signal impulsif. La règle STDP du réseau s'est concentrée sur le bruit ambiant continu et a ignoré les chocs rares. C'est ce que nous avons théorisé comme la **"Famine Compétitive"**. Le réseau est devenu sourd aux deux types de pannes (AUC 0.42). 

### 4. Le Hack Neuromorphique final : Le Modèle de Jeffress
Face aux pannes périodiques sourdes de Paderborn, l'industrie classique utilise la Transforms de Fourier (FFT), qui est lourde pour l'Edge AI sur batterie (TinyML). 
Nous avons décidé d'utiliser l'astuce neuronale des chouettes pour localiser le son : le **Modèle de Coïncidence Temporelle de Jeffress**.
En utilisant de simples registres à décalage binaires (Virtual Shift Registers) en C, nous avons construit des "Delay-Lines". Le réseau cherche une autocorrélation temporelle pure sans aucune opération en virgule flottante. 
**Résultat :** Le score sur la panne Paderborn KA03 a bondi de **0.256 à 0.987**, pour une empreinte mémoire inferieure à 12 Ko.

### Conclusion R&D et Taxonomie des pannes

**Le SNN est prouvé bon** (feature-spike 0.995, envelope V2 1.000). Le grand enseignement du projet est la création d'une **Taxonomie de l'observabilité physique** :
1. **Pannes Impulsives** (CWRU) : Détectables par Enveloppe temporelle (résolu à 100%).
2. **Périodiques sourdes** (Paderborn KA03) : Détectables par Lignes de Retard de Jeffress (résolu).
3. **Apériodiques / Frottement continu** (Paderborn KA05) : **Le Mur Physique Absolu**. Un accéléromètre seul ne peut pas discerner une usure abrasive d'un changement de régime moteur (variation RMS) sans risquer de fausses alarmes.

### Edge AI : L'implémentation C Bare-Metal (Propriétaire)

Toute la théorie validée en Python dans ce dépôt public a été portée et optimisée en langage **C Bare-Metal**. Cette implémentation industrielle est le moteur de déploiement réel de NATIVA pour les puces très basse consommation (ex: Cortex-M4, ou hardware neuromorphique synchrone).

| Métrique du Firmware C | Valeur (Validation laboratoire - TRL 4)|
|---------|--------|
| RAM totale (Allocation Statique) | **4.7 KB** |
| Latence d'inférence (Cortex-M4) | **~0.2 ms** |
| FPU (Floating Point Unit) requis | **Non** pour le modèle SNN de base |
| Dépendances externes | **Zéro (ni RTOS, ni malloc)** |

*(Note : Le code firmware C (`edge/`) est sous licence propriétaire exclusive et n'est pas inclus dans ce dépôt public. Me contacter pour partenariats industriels / licences matérielles).*

## Positionnement : Pourquoi NATIVA intéresse l'Industrie (STMicroelectronics, SKF, Siemens)

### 1. Le Mur Physique est une garantie Anti-Bullshit
Face à la prolifération de start-ups promettant de "détecter 100% des anomalies avec une IA Deep Learning", NATIVA apporte une approche d'ingénieur. Le système assume ses limites (ex: 0.45 sur Paderborn KA05) en prouvant que **le signal mono-capteur ne contient pas l'information**. Dans l'industrie lourde (aéronautique, nucléaire), une IA qui avoue "je ne peux pas voir cette panne avec un seul capteur" a paradoxalement plus de valeur qu'une boîte noire sujette aux hallucinations et aux fausses alarmes sur le moindre changement de charge (Fausses variations RMS).

### 2. Le trade-off SWaP (Size, Weight, Power)
NATIVA n'est pas conçu pour battre la précision du ML classique sur serveur. C'est le **Wake-Up Sensor ultime** pour l'Edge :
- Tourne en continu sur l'edge (ALU classique, pas de réseau flottant complexe, pas de FFT).
- Détecte *quelque chose d'anormal* (Dérive de l'Energie Libre).
- Réveille le système de diagnostic lourd (CNN/RF ou transmission radio) uniquement pour confirmation.

> *"Un Isolation Forest fait 100% de précision, mais il requiert un CPU puissant pour calculer 22 features FFT complexes en continu. NATIVA atteint le même 100% sur CWRU avec 120 opérations/sample, occupant 4.7 KB de RAM. Échanger des calculs Floating-Point gourmands contre un traitement en impulsions binaires (Spikes), c'est la définition de l'Edge AI."*

### 3. La Frontière (NATIVA 2.0) : L'Inférence Active Distribuée (IoT Mesh)
Le "mur physique" mono-capteur dicte la suite : la **Fusion Multi-modale**. L'Inférence Active de Karl Friston est bâtie sur les **Markov Blankets** (couvertures de Markov). 
La vision industrielle ultime de NATIVA est un réseau Mesh où chaque capteur (un Nœud Vibration, un Nœud Micro, un Nœud Température) fait tourner sa boucle NATIVA (4.7 KB) indépendamment. Au lieu d'émettre de grosses données temporelles en radio (très coûteux en énergie), ils échangent simplement des "Spikes de Surprise" binaires entre eux. S'ils synchronisent leurs erreurs de prédiction, ils isolent les pannes d'usure continue (comme KA05) sans nécessiter de serveur central.

**NATIVA est le framework mathématique idéal pour faire communiquer les futures puces neuromorphiques et microcontrôleurs en environnement contraint.**

## Reproduire les résultats

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Benchmark single-condition (~2 min)
python experiments/benchmark_cwru.py

# Benchmark multi-condition, 36 conditions (~10 min)
python experiments/benchmark_cwru_multi.py
```

Le dataset CWRU est téléchargé automatiquement (~30 MB).



## Références

- Dataset CWRU : [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- LIF + STDP : Diehl & Cook, "Unsupervised learning of digit recognition using STDP", 2015
- Kuramoto : Fell & Axmacher, "The role of phase synchronization in memory processes", 2011
- Active Inference : Friston, "The free-energy principle", 2010
- Multi-band encoding : inspiré de MRA-SNN (2024) et ISO 10816

## Auteur

Saad LARAJ
