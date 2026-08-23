# Introduction aux réseaux de neurones artificiels

CSI4506 Introduction à l’intelligence artificielle

> **IMPORTANT:**
>
> **Échéance** : Le devoir 2 doit être soumis au plus tard le 26 octobre 2026 à 23 heures. Veuillez consulter l’énoncé du devoir disponible sur Brightspace.

- Russell et Norvig ([2020](#ref-Russell:2020aa)), pages 705–708, 750–788

### Regarder ces vidéos de 3Blue1Brown

- [**But what is a Neural Network?**](https://youtu.be/aircAruvnKk) (19 minutes)
- [**Gradient descent, how neural networks learn**](https://youtu.be/IHZwWFHWa-w) (21 minutes)
- [**What is backpropagation really doing?**](https://youtu.be/Ilg3gGewQ5U) (14 minutes)
- [**Backpropagation calculus**](https://youtu.be/tIeHLnjs5U8) (10 minutes)

### Documentaire sur le développement de PyTorch

- [PyTorch (2024) Official PyTorch Documentary - Powering the AI Revolution](https://www.youtube.com/watch?v=rgP_LBtaUEc) (36 minutes)

## Participer

- [présentation](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## Practiquer

- [Séparabilité Circulaire](CircularSeparability) : Pour compléter votre exploration de TensorFlow Playground (ci-dessous), j’ai développé un cahier se concentrant sur l’ingénierie des attributs.

- [TensorFlow Playground](https://playground.tensorflow.org/)

  - **Options de jeu de données** : Les utilisateurs peuvent choisir parmi quatre types de jeux de données : circulaire, XOR, gaussien et spirale.
  - **Ingénierie des attributs** : Permet la création de nouveaux attributs pour améliorer la performance du modèle.
  - **Architecture du modèle** : Permet la personnalisation de l’architecture du réseau de neurones, y compris la variation du nombre de couches et de neurones par couche.
  - **Réglage des hyperparamètres** : Offre des options pour ajuster le taux d’apprentissage, les fonctions d’activation, les techniques de régularisation et les spécifications des tâches pour observer leurs effets sur l’entraînement du modèle.
  - **Suggestion 1** : Pour le jeu de données gaussien, qui est linéairement séparable, configurez un réseau sans couches cachées et un seul neurone de sortie utilisant la fonction d’activation sigmoïde. Cette configuration construit efficacement un modèle de régression logistique.
  - **Suggestion 2** : Le jeu de données circulaire n’est pas linéairement séparable en utilisant seulement les attributs originaux \\x_1\\ et \\x_2\\. Cependant, en créant de neauveaux attributs, \\x_1^2\\ et \\x_2^2\\, le problème devient linéairement séparable dans l’espace des attributs transformé. Un réseau sans couches cachées et un seul nœud de sortie est suffisant pour cette tâche.

## Les références

Russell, Stuart, et Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4ᵉ éd. Pearson. <http://aima.cs.berkeley.edu/>.
