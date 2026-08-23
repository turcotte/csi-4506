# Jupyter Notebook - Capitalisation boursière

CSI4506 Introduction à l’intelligence artificielle

Auteur·rice

Affiliations

Marcel Turcotte [](mailto:marcel.turcotte@uottawa.ca)

École de science informatique et de génie électrique

University of Ottawa

Date de publication

5 septembre 2025

# Capitalisation boursière de NVIDIA et Intel (2012 - 2024)

La bibliothèque Python `yfinance` est souvent utilisée pour télécharger des données du marché boursier.

``` python
import yfinance as yf
import matplotlib.pyplot as plt
```

Définissons les actions qui nous intéressent pour cette analyse.

``` python
# Définir les tickers pour NVIDIA et Intel
tickers = ['NVDA', 'INTC']
```

Téléchargeons maintenant les données sur notre instance Colab ou notre ordinateur local.

``` python
# Télécharger les données historiques du marché depuis 2012
data = yf.download(tickers, start='2012-01-01', end='2024-01-01', group_by='ticker')
```

En se concentrant sur les prix de clôture.

``` python
# Extraire les prix de clôture ajustés
nvda_data = data['NVDA']['Close']
intc_data = data['INTC']['Close']
```

Dessinons.

``` python
# Tracer les données des prix des actions
plt.figure(figsize=(12, 6))
plt.plot(nvda_data.index, nvda_data, label='NVIDIA')
plt.plot(intc_data.index, intc_data, label='Intel')
plt.title('Prix des actions de NVIDIA et Intel (2012 - 2024)')
plt.xlabel('Date')
plt.ylabel('Prix des actions (USD)')
plt.legend()
plt.grid(True)
plt.show()
```

Calculons maintenant la capitalisation boursière.

``` python
# Récupérer le nombre d'actions en circulation (cela donne la valeur la plus récente)
nvda_shares = yf.Ticker('NVDA').info['sharesOutstanding']
intc_shares = yf.Ticker('INTC').info['sharesOutstanding']

# Calculer la capitalisation boursière (Close ajusté * actions en circulation)
nvda_market_cap = data['NVDA']['Close'] * nvda_shares
intc_market_cap = data['INTC']['Close'] * intc_shares
```

Bien que les prix des actions de NVIDIA et Intel soient comparables, la capitalisation boursière de NVIDIA a connu une augmentation significative depuis 2020, contrairement à la capitalisation boursière plus stable d’Intel.

``` python
# Tracer les données de la capitalisation boursière
plt.figure(figsize=(12, 6))
plt.plot(nvda_market_cap.index, nvda_market_cap, label='NVIDIA')
plt.plot(intc_market_cap.index, intc_market_cap, label='Intel')
plt.title('Capitalisation boursière de NVIDIA et Intel (2012 - 2024)')
plt.xlabel('Date')
plt.ylabel('Capitalisation boursière (USD)')
plt.legend()
plt.grid(True)
plt.show()
```
