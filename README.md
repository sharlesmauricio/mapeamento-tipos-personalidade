# Mapeamento de Tipos de Personalidade com Machine Learning

Este repositório contém um projeto de clustering utilizando o algoritmo KMeans. O objetivo é agrupar dados baseados em suas características e analisar os grupos formados.

## Requisitos


1. [Pandas](https://pandas.pydata.org/docs/)
2. [Numpy](https://numpy.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/)
6. [Yellowbrick](https://www.scikit-yb.org/en/latest/)

Você pode instalar essas bibliotecas usando o seguinte comando:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


## Carregamento e Tratamento dos Dados

O dataset utilizado neste projeto é carregado de um arquivo CSV e processado para remover colunas desnecessárias e registros com valores nulos.



# Configurações
pd.options.display.max_columns = 150
pd.options.display.float_format = "{:.2f}".format

# Carregamento dos dados
data = pd.read_csv('/content/data-final.csv', sep='\t')

# Remoção de colunas desnecessárias
data.drop(data.columns[50:110], axis=1, inplace=True)

# Descrição dos dados
data.describe()
```

## Contagem de Registros por Valor

Contamos os registros por valor para identificar a quantidade de não respondentes.

```python
# Contagem dos registros por valor
data['EXT1'].value_counts()
```

## Limpeza dos Dados

Selecionamos apenas os registros com valores maiores que zero.

```python
# Seleção de registros maiores que zero
data = data[(data > 0.00).all(axis=1)]
```

## Elbow Method para Determinação do Número de Clusters

Utilizamos o método do cotovelo (Elbow Method) para determinar o número ideal de clusters.

```python
# Visualizador Elbow
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,10))

# Amostra de dados
data_sample = data.sample(5000, random_state=1)

# Início do teste
visualizer.fit(data_sample)
visualizer.poof()
```

## Agrupamento com KMeans

Instanciamos o KMeans com 5 clusters e rotulamos os dados.

```python
# Instanciação do KMeans com 5 clusters
kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data)

# Predições
predicoes = k_fit.labels_
data['Clusters'] = predicoes

# Contagem de registros por cluster
data["Clusters"].value_counts()

# Média dos clusters
data.groupby('Clusters').mean()
```

## Resultados

Os dados foram agrupados em 5 clusters, e a média de cada cluster foi calculada.

```python
# Resultados
data["Clusters"].value_counts()
data.groupby('Clusters').mean()
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
