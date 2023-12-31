﻿# Trabalho-Machine-Learning-IESB

Este projeto demonstra um fluxo de trabalho de análise de dados e aprendizado de máquina usando dados financeiros. O foco é carregar, processar e mesclar diversos conjuntos de dados financeiros, seguido pela criação e avaliação de um modelo de regressão linear para prever os preços das ações, todos os arquivos já foram tratados anteriormente.

## Descrição
O projeto envolve os seguintes passos principais:

Carregamento e Pré-processamento de Dados:

Carregar conjuntos de dados financeiros para moedas, commodities e ações.

Mesclar os conjuntos de dados com base em uma coluna comum 'Data'.
Renomear as colunas para maior clareza e consistência.
Seleção de Recursos e Definição do Alvo:

Selecionar recursos relevantes (variáveis independentes) e definir a variável alvo (variável dependente) para o modelo de aprendizado de máquina.
Divisão de Treino e Teste:

Dividir os dados em conjuntos de treino e teste para avaliar o desempenho do modelo.
Modelagem de Regressão Linear:

Inicializar e treinar um modelo de regressão linear usando scikit-learn.
Avaliação do Modelo:

Fazer previsões no conjunto de teste.
Avaliar o desempenho do modelo usando métricas como Erro Quadrático Médio (MSE) e Raiz do Erro Quadrático Médio (RMSE).

## Uso
Baixe o projeto
```
git clone https://github.com/iamedusm/Trabalho-Machine-Learning-IESB.git
cd Trabalho-Machine-Learning-IESB
pip install -r requirements.txt
python main.py
```


## Dependências
pandas
scikit-learn
