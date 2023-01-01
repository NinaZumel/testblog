---
title: Python test
date: 2022-12-31 6:38 PM
categories:
- Data Science
tags:
- blogging
- html
---

This is a test of markdown code generated from a jupyter notebook using nbconvert.
The images are separate, and I suspect I may need to load them first.


```python
import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestRegressor


import seaborn as sns
import matplotlib.pyplot as plt

```


```python
rng = np.random.default_rng(31122022)
```


```python
N = 900 # data rows
nvars = 5

X = pd.DataFrame({
    f'x{i}': rng.normal(size=N) for i in range(nvars)
})

X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.227147</td>
      <td>1.085968</td>
      <td>1.094263</td>
      <td>0.026900</td>
      <td>1.516278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.006074</td>
      <td>-0.760099</td>
      <td>0.101677</td>
      <td>-0.751571</td>
      <td>1.050778</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.404066</td>
      <td>0.423160</td>
      <td>-0.541542</td>
      <td>-0.441431</td>
      <td>0.754013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.736771</td>
      <td>-0.990351</td>
      <td>0.499307</td>
      <td>1.286138</td>
      <td>-1.028488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.586533</td>
      <td>-0.252939</td>
      <td>1.414710</td>
      <td>0.194825</td>
      <td>0.717326</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>895</th>
      <td>-0.227380</td>
      <td>-0.686395</td>
      <td>-0.037671</td>
      <td>-0.143922</td>
      <td>-1.305936</td>
    </tr>
    <tr>
      <th>896</th>
      <td>-1.804874</td>
      <td>-1.437549</td>
      <td>-0.382849</td>
      <td>-0.225181</td>
      <td>0.464108</td>
    </tr>
    <tr>
      <th>897</th>
      <td>1.261462</td>
      <td>0.287481</td>
      <td>-0.521777</td>
      <td>-0.380030</td>
      <td>-0.487183</td>
    </tr>
    <tr>
      <th>898</th>
      <td>-1.293631</td>
      <td>0.648808</td>
      <td>-0.688767</td>
      <td>-0.202352</td>
      <td>0.255296</td>
    </tr>
    <tr>
      <th>899</th>
      <td>0.039766</td>
      <td>1.410734</td>
      <td>-1.215591</td>
      <td>1.460892</td>
      <td>2.340056</td>
    </tr>
  </tbody>
</table>
<p>900 rows × 5 columns</p>
</div>








```python
y = (X['x0'] + X['x1']**2 + X['x2'] * (2*X['x3'] - X['x4']))
```


I've dropped showing the model here, since there's nothing to show.


```python
model_rf = RandomForestRegressor()
model_rf.fit(X, y)
```

```python
pframe = pd.DataFrame({
    'actual': y,
    'pred': model_rf.predict(X)
})

r2 = sklearn.metrics.r2_score(y_true=y, y_pred = pframe['pred'])

sns.scatterplot(
    data=pframe,
    x='pred',
    y='actual'
)
plt.plot([np.min(y),np.max(y)], [np.min(y),np.max(y)])

plt.title(f'actual presented as a function of prediction, r-squared={r2:.2f}')
plt.show()
```

    
![png](https://ninazumel.com/testblog/images/myth1_regression_example_files/myth1_regression_example_10_0.png)

The link to the image has to be hand-tweaked.

