# Construire des fonctions avec ReLU

Code

- [Montrer tout le code](javascript:void(0))

- [Cacher tout le code](javascript:void(0))

- 

  ------------------------------------------------------------------------

- [Voir les sources](javascript:void(0))

Un complément interactif à la présentation 11

Auteur·rice

Marcel Turcotte

Date de publication

Version : 18 oct. 2026

Modifiez les sept paramètres d’un réseau comportant **une entrée, deux unités cachées ReLU et une sortie linéaire**. Commencez par modifier un seul paramètre à la fois.

``` math
f(x)=w_3\varphi(w_1x+b_1)+w_4\varphi(w_2x+b_2)+b_3.
```

Ici, $`\varphi(z)=\max(0,z)`$ est la fonction ReLU. Les courbes de couleur montrent les deux **contributions pondérées**. La courbe plus épaisse représente leur somme à laquelle s’ajoute le biais de sortie $`b_3`$.

Afficher le code Observable

``` js
viewof parameters = {
  const initial = {w1: 1, b1: 1, w3: 1.5, w2: 1, b2: -1, w4: -2, b3: 0};
  const labels = {w1: "w₁", b1: "b₁", w3: "w₃", w2: "w₂", b2: "b₂", w4: "w₄", b3: "b₃"};
  const inputs = Object.fromEntries(
    Object.entries(initial).map(([name, value]) => {
      const input = Inputs.range([-3, 3], {label: labels[name], value, step: 0.1, width: 190});
      input.querySelector('input[type="range"]').setAttribute("aria-label", labels[name]);
      return [name, input];
    })
  );

  const form = Inputs.form(inputs, {
    template: fields => html`<div class="relu-controls">
      <fieldset style="--unit-colour: #c26817">
        <legend>Unité cachée 1</legend>
        <p class="unit-caption">Poids d'entrée, biais, poids de sortie</p>
        ${fields.w1}${fields.b1}${fields.w3}
      </fieldset>
      <fieldset style="--unit-colour: #3679b0">
        <legend>Unité cachée 2</legend>
        <p class="unit-caption">Poids d'entrée, biais, poids de sortie</p>
        ${fields.w2}${fields.b2}${fields.w4}
      </fieldset>
      <fieldset style="--unit-colour: currentColor">
        <legend>Sortie</legend>
        <p class="unit-caption">Déplacement vertical de la courbe complète</p>
        ${fields.b3}
      </fieldset>
      <div><button type="button" class="btn btn-outline-secondary btn-sm">Réinitialiser les paramètres</button></div>
    </div>`
  });

  form.querySelector("button").addEventListener("click", () => {
    form.value = initial;
    form.dispatchEvent(new Event("input", {bubbles: true}));
  });

  return form;
}
```

Afficher le code Observable

``` js
viewof yExtent = Inputs.select([8, 16, 40, 80], {
  value: 8,
  label: "Échelle verticale",
  format: extent => `−${extent} à +${extent}`
})
```

Afficher le code Observable

``` js
Plot.plot({
  width: Math.min(width, 850),
  height: 410,
  marginLeft: 50,
  marginRight: 20,
  x: {domain: [-3, 3], label: "x", grid: true},
  y: {domain: [-yExtent, yExtent], label: "Sortie", grid: true},
  color: {
    domain: ["Contribution de l'unité 1", "Contribution de l'unité 2", "Sortie du réseau"],
    range: ["#c26817", "#3679b0", "currentColor"],
    legend: true
  },
  marks: [
    Plot.ruleY([0], {strokeOpacity: 0.35}),
    Plot.ruleX(breakpoints, {
      x: "x", stroke: "series", strokeDasharray: "4,4", strokeOpacity: 0.45,
      clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "unit1", stroke: () => "Contribution de l'unité 1",
      strokeWidth: 2, clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "unit2", stroke: () => "Contribution de l'unité 2",
      strokeWidth: 2, clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "output", stroke: () => "Sortie du réseau",
      strokeWidth: 3.5, clip: true
    })
  ]
})
```

Les axes restent fixes lorsque les paramètres changent. Utilisez le sélecteur d’échelle verticale si une courbe dépasse les limites actuelles. Les lignes verticales en pointillés indiquent les positions des changements de pente des unités cachées.

Afficher le code Observable

``` js
html`<div class="relu-breakpoints" aria-live="polite">
  <p><strong>Unité 1 :</strong> ${describeBreakpoint(parameters.w1, parameters.b1)}</p>
  <p><strong>Unité 2 :</strong> ${describeBreakpoint(parameters.w2, parameters.b2)}</p>
</div>`
```

## Au-delà de deux unités

Chaque unité cachée apporte une courbe ReLU décalée et mise à l’échelle. Leur somme peut changer de pente partout où l’une des unités change de pente. Les poids de sortie peuvent renforcer les contributions ou les faire s’opposer, tandis que $`b_3`$ déplace l’ensemble du résultat verticalement.

Imaginez l’ajout d’autres unités cachées. Où pourriez-vous placer de nouveaux changements de pente, et comment choisiriez-vous les poids de sortie pour façonner les segments entre ces points? Avec davantage d’unités, nous disposons de plus de composantes pour construire des fonctions complexes. Cette exploration rejoint la discussion de la présentation sur l’approximation avec des couches cachées plus larges.

Essayez de fixer $`w_3=0`$ pour supprimer la contribution de la première unité, ou changez le signe de $`w_4`$ pour inverser la contribution de la seconde. Utilisez **Réinitialiser les paramètres** pour rétablir le réseau de départ.

Pour expérimenter avec davantage d’unités cachées, explorez le [Neural Network Visualizer](https://kickereb.github.io/nn-visualizer/experiment-2d.html). Ajoutez des unités et observez l’évolution de la courbe de régression du réseau. Ses unités cachées utilisent la fonction d’activation $`\tanh`$.

[Retour à la présentation 11](slides.llms.md#approximation-universelle).

## Comment les courbes sont calculées

Le navigateur évalue le réseau en différents points de l’intervalle $`[-3,3]`$. Les positions des changements de pente situées dans cet intervalle sont incluses explicitement pour que la courbe tracée passe par chaque point de changement de pente.

Afficher le code Observable

``` js
relu = z => Math.max(0, z)

breakpoints = [
  {w: parameters.w1, b: parameters.b1, series: "Contribution de l'unité 1"},
  {w: parameters.w2, b: parameters.b2, series: "Contribution de l'unité 2"}
].filter(d => d.w !== 0)
  .map(d => ({x: -d.b / d.w, series: d.series}))
  .filter(d => d.x >= -3 && d.x <= 3)

samples = {
  const xs = Array.from({length: 601}, (_, i) => -3 + i / 100);
  xs.push(...breakpoints.map(d => d.x));
  return [...new Set(xs)].sort((a, b) => a - b).map(x => {
    const unit1 = parameters.w3 * relu(parameters.w1 * x + parameters.b1);
    const unit2 = parameters.w4 * relu(parameters.w2 * x + parameters.b2);
    return {x, unit1, unit2, output: unit1 + unit2 + parameters.b3};
  });
}

describeBreakpoint = (w, b) => {
  if (w === 0) return "Activation constante : aucun changement de pente.";
  const x = -b / w;
  return `x = ${x.toFixed(2).replace(".", ",")}${x < -3 || x > 3 ? " (hors de l'intervalle représenté)" : ""}`;
}
```
