# Rapport TP PixelCNN

Pour ce TD, onai travaillé sur la génération de texte avec le modèle GPT. J'ai trouvé que son fonctionnement est assez intuitif : il prédit chaque token l'un après l'autre en se basant uniquement sur ceux qu'il a déjà vus. Ça permet de générer des textes qui ont l'air cohérents.

## Modele autorégressif

Pour faire simple, ce modèle décompose une distribution complexe en plusieurs morceaux plus faciles à gérer. La formule ressemble à ça :

$$
p(x) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1,x_2) \dots p(x_n|x_1, \dots, x_{n-1})
$$

En gros, chaque token $x_t$ dépend juste de ceux d'avant. Ça garantit que le modèle ne "triche" pas en regardant des tokens futurs.

## Preparation des données

La preparation des données se font en plusieurs etapes :

1. Nettoyer les chaines de caracteres : mettre en minuscule, ajouter des espaces autour de la ponctuation, virer les caractères bizarres
2. Ensuite tokeniser
3. Limiter le vocabulaire aux 10 000 mots les plus fréquents (generalement on prends 90-95% du vocabulaire du dataset)

## Dataset

Pour l'entraînement, on créé des exemples avec :

- Une entrée : $(x_1, x_2, \dots, x_{n-1})$
- Et sa cible correspondante : $(x_2, x_3, \dots, x_n)$

Comme ça, le modèle apprend à deviner le prochain mot

## Architecture

Mon modèle GPT est composé de plusieurs parties :

### Les embeddings et l'encodage de position

- Pour les embeddings, on transforme chaque token $x_i$ en un vecteur $e_i \in \mathbb{R}^{d_{model}}$
- On ajoute aussi un embedding de position $p_i$ pour que le modèle sache dans quel ordre sont les mots :

$$
h_i^{(0)} = e_i + p_i
$$

où $h_i^{(0)}$ représente le token $i$ au début du réseau.

### Les blocs Transformer

Chaque bloc contient :

#### De l'attention masquée (multi-head)

On calcule l'attention comme ça :

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$, $K$, et $V$ sont des proonctions linéaires des embeddings
- $d_k$ c'est la dimension des clés
- Le masque empêche de regarder les mots futurs : on considère que les positions $j \leq i$

#### Un réseau feed-forward (MLP)

Cette transformation non-linéaire enrichit les représentations :

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

- $W_1$, $W_2$ sont les poids et $b_1$, $b_2$ les biais

#### Des connexions résiduelles et normalisations

Ça aide à stabiliser l'entraînement :

$$
x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

## La couche finale

on  mis une projection linéaire suivie d'un softmax pour prédire le prochain token :

$$
p(x_t|x_{<t}) = \text{softmax}(W_{vocab} h_t + b_{vocab})
$$

- $W_{vocab}$ et $b_{vocab}$ sont les paramètres de cette dernière couche

## Comment on  entraîné ça

on  minimisé la Cross-Entropy Loss :

$$
\text{Loss} = -\frac{1}{N}\sum_{t=1}^{N}\log p(x_t | x_{<t})
$$

Cette loss mesure si le modèle arrive à prédire correctement le prochain token.

## La génération de texte

Pour générer du texte, j'échantillonne token par token selon :

$$
p(x_{t+1}|x_{\leq t}) = \frac{\exp(z_{t+1}/T)}{\sum_{i}\exp(z_i/T)}
$$

- $z$ représente les logits en sortie du modèle
- $T$ c'est la température qui change la diversité :
  - Quand $T < 1$, le texte est plus cohérent mais moins créatif
  - Quand $T > 1$, le texte est plus varié mais risque d'être moins sensé
