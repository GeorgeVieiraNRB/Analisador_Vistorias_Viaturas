# Vistória de Viatura

Descrição dos requisitos necessários para executar o programa.

![](header.png)

## Instalações necessárias


```sh
pip install pandas 
pip install nltk
pip install spacy 
pip install gensim 
pip install numpy 
pip install seaborn 
pip install datetime
pip install sklearn
pip install fastapi 
pip install uvicorn
pip install -U scikit-learn
```

## Como executar o arquivo

```sh
1. É necessário importar os dados de treino e os dados que serão classificados. Um exemplo desses dados se encontram na pasta deste projeto.

2. Treinar o modelo: A classe Training() é responsável por treinar o modelo de classificação. No arquivo training.py, existe um exemplo de como executar essa classe. UM modelo persistente do classificação é salvo. 

3. Classificação:  A classe Classificador(): é responsável por classificar em problemas urgentes e não urgentes dos dados do arquivo de entrada. No arquivo classification.py, existe um exemplo de como executar essa classe.

Observação: Ao desenvolver o código, executamos a classe Classificador(). 
```


## Integração com a FastAPI
```sh
Na arquivo main.py existe um tamplate para a integração com a FastAPI. 
```