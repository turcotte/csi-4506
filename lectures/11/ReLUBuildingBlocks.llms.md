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

Version : 20 oct. 2026

Modifiez les sept paramètres d’un réseau comportant **une entrée, deux unités cachées ReLU et une sortie linéaire**. Commencez par modifier un seul paramètre à la fois.

``` math
f(x)=w_3\varphi(w_1x+b_1)+w_4\varphi(w_2x+b_2)+b_3.
```

Ici, $`\varphi(z)=\max(0,z)`$ est la fonction ReLU. Les courbes de couleur montrent les deux **activations cachées**. La courbe plus épaisse les combine à l’aide des poids de sortie $`w_3`$ et $`w_4`$, puis ajoute le biais de sortie $`b_3`$.

Afficher le code Observable

``` js
networkDiagram = {
  const nodes = {
    input: {x: 100, y: 260, r: 36, type: "input", fill: "#afffff"},
    inputBias: {x: 100, y: 65, r: 24, type: "constant", fill: "#eeeeee"},
    hidden1: {x: 470, y: 170, r: 44, type: "relu", fill: "#bfbfff", accent: "#c26817", caption: "Unité 1"},
    hidden2: {x: 470, y: 350, r: 44, type: "relu", fill: "#bfbfff", accent: "#3679b0", caption: "Unité 2"},
    hiddenBias: {x: 470, y: 65, r: 24, type: "constant", fill: "#eeeeee"},
    output: {x: 850, y: 260, r: 44, type: "linear", fill: "#80ff80", caption: "f(x)"}
  };
  const edges = [
    {from: "input", to: "hidden1", name: "w1", symbol: "w₁", lx: 285, ly: 206},
    {from: "input", to: "hidden2", name: "w2", symbol: "w₂", lx: 285, ly: 323},
    {from: "hidden1", to: "output", name: "w3", symbol: "w₃", lx: 650, ly: 190},
    {from: "hidden2", to: "output", name: "w4", symbol: "w₄", lx: 650, ly: 325},
    {from: "inputBias", to: "hidden1", name: "b1", symbol: "b₁", lx: 275, ly: 102},
    {from: "inputBias", to: "hidden2", name: "b2", symbol: "b₂", lx: 225, ly: 170},
    {from: "hiddenBias", to: "output", name: "b3", symbol: "b₃", lx: 680, ly: 155}
  ];
  const formatValue = value => value.toLocaleString("fr-CA", {maximumFractionDigits: 1});

  const drawEdge = edge => {
    const start = nodes[edge.from], end = nodes[edge.to];
    const dx = end.x - start.x, dy = end.y - start.y;
    const length = Math.hypot(dx, dy);
    return svg`<line data-parameter=${edge.name} data-from=${edge.from} data-to=${edge.to}
      x1=${start.x + dx * start.r / length} y1=${start.y + dy * start.r / length}
      x2=${end.x - dx * (end.r + 3) / length} y2=${end.y - dy * (end.r + 3) / length}
      stroke="currentColor" stroke-width="2" marker-end="url(#relu-network-arrow)"></line>`;
  };

  const drawNode = ([name, node]) => svg`<g data-node=${name} transform="translate(${node.x},${node.y})">
    <circle r=${node.r} fill=${node.fill} stroke=${node.accent || "#111"} stroke-width="2.5"></circle>
    ${node.type === "input" || node.type === "constant"
      ? svg`<text text-anchor="middle" y="10" font-size=${node.type === "input" ? 36 : 28}
          font-style=${node.type === "input" ? "italic" : "normal"} fill="#111">${node.type === "input" ? "x" : "1"}</text>`
      : svg`<g fill="#111" stroke="#111">
          <line y1=${-node.r} y2=${node.r} stroke-width="2"></line>
          <text x="-20" y="12" text-anchor="middle" font-size="39" stroke="none">∑</text>
          <path d="${node.type === "relu" ? "M 6 15 H 16 L 34 -15" : "M 7 17 L 34 -17"}"
            fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
        </g>`}
    ${node.caption ? svg`<text text-anchor="middle" y=${node.r + 25}
      font-family="${node.type === "linear" ? "Georgia, serif" : "system-ui, sans-serif"}"
      font-size="18" fill=${node.accent || "currentColor"}>${node.caption}</text>` : null}
  </g>`;

  return html`<figure class="relu-network-figure">
    <div class="relu-network-scroll">
      ${svg`<svg class="relu-network-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440"
        role="img" aria-labelledby="relu-network-title relu-network-description" font-family="Georgia, serif">
        <title id="relu-network-title">Une entrée, deux unités cachées ReLU et une sortie linéaire</title>
        <desc id="relu-network-description">L'entrée x est reliée aux deux unités cachées par w₁ et w₂.
          Les unités cachées sont reliées à la sortie par w₃ et w₄.
          Un nœud constant au-dessus de x fournit les biais b₁ et b₂; un second nœud constant fournit b₃ à la sortie.
          Les deux nœuds constants n'ont aucune arête entrante.</desc>
        <defs>
          <marker id="relu-network-arrow" viewBox="0 -4 8 8" refX="8" refY="0"
            markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
            <path d="M 0 -4 L 8 0 L 0 4 Z" fill="currentColor"></path>
          </marker>
        </defs>
        <g text-anchor="middle" font-family="system-ui, sans-serif" font-size="17" fill="currentColor">
          <text x="100" y="20">Entrée</text>
          <text x="470" y="20">Couche cachée (ReLU)</text>
          <text x="850" y="20">Sortie linéaire</text>
        </g>
        ${edges.map(drawEdge)}
        ${Object.entries(nodes).map(drawNode)}
        ${edges.map(edge => svg`<text class="parameter-label" data-label=${edge.name}
          x=${edge.lx} y=${edge.ly} text-anchor="middle">${edge.symbol} = ${formatValue(parameters[edge.name])}</text>`)}
      </svg>`}
    </div>
    <figcaption>Les petits nœuds émettent toujours 1 et n'ont aucune arête entrante. Les poids de leurs arêtes sortantes sont les biais.
      Les étiquettes des paramètres suivent les curseurs; les contours orange et bleu identifient les deux activations cachées.</figcaption>
  </figure>`;
}
```

Afficher le code Observable

``` js
viewof parameters = {
  const initial = {w1: 2, b1: 3, w3: 1, w2: 2, b2: 0, w4: -2, b3: 3};
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
    domain: ["Activation de l'unité 1", "Activation de l'unité 2", "Sortie du réseau"],
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
      x: "x", y: "activation1", stroke: () => "Activation de l'unité 1",
      strokeWidth: 2, clip: true
    }),
    Plot.lineY(samples, {
      x: "x", y: "activation2", stroke: () => "Activation de l'unité 2",
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

La courbe de sortie combine les deux activations cachées. Sa pente peut changer partout où l’une des activations change de pente. Les poids de sortie $`w_3`$ et $`w_4`$ mettent les activations à l’échelle et peuvent renforcer leurs contributions ou les faire s’opposer, tandis que $`b_3`$ déplace l’ensemble du résultat verticalement.

Imaginez l’ajout d’autres unités cachées. Où pourriez-vous placer de nouveaux changements de pente, et comment choisiriez-vous les poids de sortie pour façonner les segments entre ces points? Avec davantage d’unités, nous disposons de plus de composantes pour construire des fonctions complexes. Cette exploration rejoint la discussion de la présentation sur l’approximation avec des couches cachées plus larges.

Essayez de fixer $`w_3=0`$ : la première activation cachée reste visible, mais ne contribue plus à la sortie du réseau. Changer le signe de $`w_4`$ inverse la contribution de la seconde unité à la sortie sans modifier sa courbe d’activation bleue. Utilisez **Réinitialiser les paramètres** pour rétablir le réseau de départ.

Pour expérimenter avec davantage d’unités cachées, explorez le [Neural Network Visualizer](https://kickereb.github.io/nn-visualizer/experiment-2d.html). Ajoutez des unités et observez l’évolution de la courbe de régression du réseau. Ses unités cachées utilisent la fonction d’activation $`\tanh`$.

[Retour à la présentation 11](slides.llms.md#approximation-universelle).

## Comment les courbes sont calculées

Le navigateur évalue le réseau en différents points de l’intervalle $`[-3,3]`$. Les positions des changements de pente situées dans cet intervalle sont incluses explicitement pour que la courbe tracée passe par chaque point de changement de pente.

Afficher le code Observable

``` js
relu = z => Math.max(0, z)

breakpoints = [
  {w: parameters.w1, b: parameters.b1, series: "Activation de l'unité 1"},
  {w: parameters.w2, b: parameters.b2, series: "Activation de l'unité 2"}
].filter(d => d.w !== 0)
  .map(d => ({x: -d.b / d.w, series: d.series}))
  .filter(d => d.x >= -3 && d.x <= 3)

samples = {
  const xs = Array.from({length: 601}, (_, i) => -3 + i / 100);
  xs.push(...breakpoints.map(d => d.x));
  return [...new Set(xs)].sort((a, b) => a - b).map(x => {
    const activation1 = relu(parameters.w1 * x + parameters.b1);
    const activation2 = relu(parameters.w2 * x + parameters.b2);
    const output = parameters.w3 * activation1 + parameters.w4 * activation2 + parameters.b3;
    return {x, activation1, activation2, output};
  });
}

describeBreakpoint = (w, b) => {
  if (w === 0) return "Activation constante : aucun changement de pente.";
  const x = -b / w;
  return `x = ${x.toFixed(2).replace(".", ",")}${x < -3 || x > 3 ? " (hors de l'intervalle représenté)" : ""}`;
}
```
