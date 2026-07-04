
# Fiabilité d'une poutre fléchie – Code Python

Ce dépôt contient le code Python complet qui implémente la méthodologie de fiabilité structurale présentée dans l'article  
**"Influence de l'incertitude des propriétés mécaniques sur la fiabilité d'une poutre"**.

Le code réalise une analyse de fiabilité avancée pour une poutre isostatique de section rectangulaire soumise à deux états limites :
- **État Limite de Service (ELS)** – critère de flèche
- **État Limite Ultime (ELU)** – critère de résistance en flexion

Il intègre :
- des lois de probabilité non‑normales (Log‑normale pour le module d’Young et la résistance, Gumbel pour la charge),
- la transformation de Rosenblatt,
- l’algorithme HL‑RF robustifié par une recherche linéaire d’Armijo,
- une optimisation basée sur la fiabilité (RBDO) par dichotomie,
- une validation par simulation de Monte Carlo avec intervalles de confiance de Wilson,
- la génération automatique des graphiques de résultats.

---

## 🚀 Utilisation

Le programme effectue les étapes suivantes :

1. **HL‑RF** pour une hauteur initiale \(h = 0{,}30\) m :
   - calcule les indices de fiabilité \(\beta_1\) (flèche) et \(\beta_2\) (résistance),
   - affiche les facteurs de sensibilité \(\alpha\) (cosinus directeurs).

2. **Monte Carlo** (\(N = 10^6\) tirages) pour valider les résultats FORM.

3. **RBDO** : recherche par dichotomie des hauteurs critiques \(h_{\text{ELS}}\) et \(h_{\text{ELU}}\), puis détermine \(h_{\min}\).

4. **Validation Monte Carlo** (\(N = 500\,000\) tirages) sur la hauteur optimale \(h_{\min}\).

5. **Génération de deux graphiques** :
   - `beta_vs_h.png` : évolution des indices de fiabilité \(\beta_1(h)\) et \(\beta_2(h)\),
   - `histogrammes_double_etat.png` : distributions des marges de sécurité \(g_1\) et \(g_2\) à \(h=0{,}30\) m.

### Modifier les paramètres d’étude

Les paramètres géométriques et probabilistes sont définis en tête du script :

```python
# Paramètres géométriques
L = 6.0          # portée (m)
b = 0.20         # largeur (m)
w_lim = L / 250  # flèche limite (m)
h_initial = 0.30 # hauteur initiale (m)

# Paramètres probabilistes
mu_E = 30e9      # module d'Young moyen (Pa)
delta_E = 0.10   # coefficient de variation de E
mu_q = 15e3      # charge moyenne (N/m)
delta_q = 0.10   # coefficient de variation de q
mu_fy = 250e6    # contrainte d'élasticité moyenne (Pa)
delta_fy = 0.10  # coefficient de variation de fy
```

**Pour étudier un autre cas** (par exemple une poutre en béton ou une portée différente), modifiez ces valeurs et ajustez si nécessaire les bornes de recherche `h_min` et `h_max` dans la section RBDO.

---

## 📁 Structure du code

Le code est organisé en plusieurs sections fonctionnelles :

| Section / Fonction | Description |
|---------------------|-------------|
| **Paramètres** | Définition des données géométriques et probabilistes. |
| `U_to_X(u)` | Transformation inverse de Rosenblatt : espace normal standard \(\mathbf{U}\) → espace physique \((E, q, f_y)\). |
| `dX_dU(u)`  | Dérivées des variables physiques par rapport aux variables normales (utiles pour le gradient). |
| `g1(h, u)`, `grad_g1(h, u)` | Fonction de performance ELS et son gradient. |
| `g2(h, u)`, `grad_g2(h, u)` | Fonction de performance ELU et son gradient. |
| `compute_beta(h, g_func, grad_func, ...)` | Algorithme HL‑RF avec recherche linéaire d’Armijo (backtracking). Retourne \(\beta\), le point de conception et un booléen de convergence. |
| `monte_carlo_pf(h, N, seed)` | Simulation Monte Carlo. Retourne les probabilités de défaillance, les indices \(\beta\), les intervalles de Wilson, ainsi que les échantillons de \(g_1\) et \(g_2\) pour traçage. |
| `find_h_target(target_beta, beta_func, h_min, h_max)` | Résolution par dichotomie de l’équation \(\beta(h) = \text{cible}\). |
| **Programme principal** | Orchestre l’exécution complète (HL‑RF, Monte Carlo, RBDO, graphiques). |

---

## 📊 Résultats typiques

Pour les paramètres par défaut (\(L=6\) m, \(b=0{,}20\) m, acier), vous obtiendrez des résultats proches de :

| Paramètre | Valeur |
|-----------|--------|
| \(\beta_1\) (FORM) à \(h=0{,}30\) m | ≈ 1,7451 |
| \(\beta_1\) (Monte Carlo) à \(h=0{,}30\) m | ≈ 1,7079 |
| \(P_{f,1}\) (Monte Carlo) à \(h=0{,}30\) m | ≈ \(4{,}38 \times 10^{-2}\) |
| \(h_{\text{ELS}}\) (pour \(\beta=1{,}5\)) | ≈ 0,296 m |
| \(h_{\text{ELU}}\) (pour \(\beta=3{,}8\)) | ≈ 0,123 m |
| \(h_{\min}\) (FORM corrigé) | ≈ 0,297 m |
| Section finale recommandée | **0,20 × 0,30 m²** |
| \(\beta_1\) (Monte Carlo) à \(h_{\min}\) | ≈ 1,545 (\(> 1{,}5\), validation OK) |
| Écart FORM / Monte Carlo | ≈ \(2{,}2\,\%\) |

Les graphiques générés permettent d’apprécier visuellement la sensibilité des indices de fiabilité à la hauteur de la poutre.


## 📧 Contact

**Mohamed Ayoub Balkhouar**  
École Hassania des Travaux Publics, Casablanca  
[mohamedayoubbalkha@gmail.com]

**Bonnes simulations !** 🚀
```
