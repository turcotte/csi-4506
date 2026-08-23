# Régression logistique

CSI4506 Introduction à l’intelligence artificielle

> **IMPORTANT:**
>
> **Échéance** : Le devoir 1 doit être soumis au plus tard le 5 octobre 2026 à 23 heures. Veuillez consulter l’énoncé du devoir disponible sur Brightspace.

## Préparer

- Régression logistique: Russell et Norvig ([2020](#ref-Russell:2020aa)), pages 684-686
- Regarder [Machine Learning and Logistic Regression](https://youtu.be/AX-ZEC-71DI), IBM Technology.

## Participer

- [présentation](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## Pratiquer

En classe, nous avons développé un modèle de régression logistique pour la reconnaissance des chiffres manuscrits à partir d’un jeu de données de l’UCI ML. Ce jeu de données comprend 1797 images de taille \\8 \times 8\\. Le jeu de données [MNIST](https://www.openml.org/search?type=data&sort=runs&id=554&status=active) (`mnist_784`) contient 70 000 images de taille \\28 \times 28\\. L’[exemple](https://scikit-learn.org/dev/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html) suivant tiré du site `sklearn` utilise ce jeu de données et présente graphiquement les coefficients (\\\theta\\) pour chacun des 10 modèles. Vous pouvez charger ce modèle de la manière suivante :

``` python
from sklearn.datasets import fetch_openml  

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

## Les références

Russell, Stuart, et Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4ᵉ éd. Pearson. <http://aima.cs.berkeley.edu/>.
