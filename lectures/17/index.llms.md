# Recherche informée

CSI4506 Introduction à l’intelligence artificielle

> **IMPORTANT:**
>
> **Échéance** : Le devoir 3 doit être soumis au plus tard le 16 novembre 2026 à 23 heures. Veuillez consulter l’énoncé du devoir disponible sur Brightspace.
>
> **Échéance** : Le Quiz 2 est prévu pour le mercredi 18 novembre 2026, pendant le cours. Des détails supplémentaires sont disponibles dans la [FAQ](../../course-faq).

## Se préparer

- Russell et Norvig ([2020](#ref-Russell:2020aa)), pages 85–92
- Révisez la [présentation 16](../../lectures/16/index.llms.md), notamment la représentation des états, la génération des successeurs, la frontière et la recherche en largeur.

## Participer

- [présentation](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## S’exercer

Utilisez le carnet Jupyter de la présentation pour :

- Comparer les heuristiques du nombre de tuiles mal placées et de la distance de Manhattan sur les mêmes plateaux. Comparez les développements de nœuds et validez les chemins et les longueurs de solution par rapport à BFS sur un petit échantillon.
- Poser h(s)=0 et expliquer la relation entre A\*, la recherche à coût uniforme et BFS, y compris l’effet du départage.

Pour une vérification empirique plus étendue, téléchargez le [script de validation BFS–A\* (en anglais)](validate_astar.py). Exécutez `python3 validate_astar.py 4106 25000` pour enregistrer les résultats de 25 000 plateaux résolubles distincts ; le carnet de la présentation lit le CSV enregistré pour afficher les histogrammes et les statistiques.

## Les références

Russell, Stuart, et Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4ᵉ éd. Pearson. <http://aima.cs.berkeley.edu/>.
