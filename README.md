# TI0175-stellar-classification

Equipe: 

- **Carlos Vinícius - 558171** 
- **Francisco Vinícius - 553889** 

Este repositório contém a implementação do Homework 2 da disciplina TI0175, cujo objetivo é avaliar a capacidade de diferentes métodos de regressão em prever o redshift a partir de atributos fotométricos e estruturais. Para isso, foram utilizados quatro modelos principais: Regressão Linear Ordinária (OLS), Regressão Ridge, Regressão por Componentes Principais (PCR) e uma Rede Neural do tipo MLP. O trabalho inclui também a comparação entre implementações feitas do zero e as versões equivalentes da biblioteca scikit‑learn.

O projeto segue uma estrutura padronizada: leitura dos dados de treino e teste, remoção das variáveis não numéricas, padronização dos atributos e execução dos modelos. Cada método é avaliado tanto em termos de desempenho nos conjuntos de treino e teste quanto por meio de validação cruzada. Os resultados incluem métricas como RMSE e R², gráficos de desempenho, seleção do melhor parâmetro para Ridge e PCR e uma análise comparativa final entre todos os modelos.

O código está dividido em arquivos independentes para cada regressão e se concentram na pasta nomeada como **regressão** (dentro dessa pasta tambem há um aviso importante), além de uma pasta contendo os dados utilizados. O relatório associado discute o comportamento de cada modelo, seus limites, vantagens e o nível de generalização alcançado com os dados disponíveis. Há tambem muitos arquivos relacionados ao homework 1, já que compatilha o dataset com o homework 2.

Fonte dos dados:
fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. Retrieved October 2025 from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17.
