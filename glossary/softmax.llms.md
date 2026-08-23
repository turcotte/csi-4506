# Glossaire

Softmax

Auteur·rice

Affiliations

Marcel Turcotte [](mailto:marcel.turcotte@uottawa.ca)

École de science informatique et de génie électrique

Université d’Ottawa

Date de publication

28 août 2025

La fonction **softmax** standard (unitaire) est une fonction mathématique qui convertit un vecteur de nombres réels en une distribution de probabilité, où les composantes du vecteur sont exponentiées puis normalisées en divisant par la somme de toutes les composantes exponentiées. Cela garantit que les valeurs de sortie sont dans l’intervalle \\(0, 1)\\ et s’additionnent à 1, ce qui les rend adaptées pour représenter des probabilités. Plus précisément, \\\sigma: \mathbb{R}^K \rightarrow(0,1)^K\\, pour \\K \> 1\\ et \\\mathbf{z} = (z_1,\ldots,z_K) \in \mathbb{R}^K\\.

\\ \sigma(\mathbf{z}) = \frac{e^{z_i}}{\sum\_{j=1}^K e^{z_j}} \\

``` python
import numpy as np

def softmax(z):
  return np.exp(z) / np.sum(np.exp(z))

s = softmax([1.47, -0.39, 0.22])
print(s, sum(s))
```

    [0.69339596 0.10794277 0.19866127] 1.0
