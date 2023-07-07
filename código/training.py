import pandas as pd
from pre_processing import Processing 
from word2vec import Representation
from sklearn.svm import SVC
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load
from sklearn.metrics import confusion_matrix


class Training():
    ''''''
    def __init__(self, file_csv, dimensoes=100):
        self.df = pd.read_csv(file_csv)
        self.new_data = self.df.dropna(axis = 0, how ='any') 
        self.df_treino, self.df_teste = train_test_split(self.new_data, test_size=0.3, stratify=self.new_data['target'])    
        self.dimensoes = dimensoes
    
    def _gerar_representacao(self, r:Representation):
        return r.gerar_representacao()
        

    def gerar_classificador(self):
        '''Cria o modelo de classificação.'''
        #Aplicar o pré-processamento aos dados de treino
        Processing().processing(self.df_treino)
        #Criar a representação de treino
        representacao_treino = Representation(self.dimensoes, self.df_treino['dados_tokenizados']) 
        repr_word2vec = self._gerar_representacao(representacao_treino)
        #Construindo o modelo classificador
        classificador = SVC(kernel='linear', C=2.5)
        classificador.fit(repr_word2vec, self.df_treino['target'])
        #Salavando o modelo de classificação
        dump(classificador, 'classificador.joblib')
        

    def _estatistic(self):
        '''Mede a accuracia da classificação.'''
        #Aplicar o pré-processamento aos dados de teste
        Processing().processing(self.df_teste) 
        #Criar a representação de teste
        representacao_teste = Representation(self.dimensoes, self.df_teste['dados_tokenizados']) 
        repr_word2vec = self._gerar_representacao(representacao_teste )
        #Lendo o classificador salvo
        classificador = load('classificador.joblib')
        predicoes = classificador.predict(repr_word2vec)
        accu = accuracy_score(self.df_teste['target'], predicoes)
        return  predicoes, accu


    def accuracy_and_matrix_confusao(self):
        predicoes, accuracy= self._estatistic()
        confusion =  confusion_matrix(self.df_teste['target'], predicoes)
        print(f"Accuracy: {accuracy}")
        self._tostring_confusion_matrix(confusion)
    

    def _tostring_confusion_matrix(self, confusion_matrix):
        print(f"Verdadeiro Positivo: {confusion_matrix[0][0]}")
        print(f"Falso Positivo: {confusion_matrix[0][1]}")
        print(f"Falso Negativo: {confusion_matrix[1][0]}")
        print(f"Verdadeiro Negativo: {confusion_matrix[1][1]}")


#####################################################################
#Exemplo de como executar essa classe
#####################################################################
#Cria a classe de treinamento
treinamento = Training('dados_treino.csv', 175)
#Gera o classificador
treinamento.gerar_classificador()
#Mede a acurácia da classificação
treinamento.accuracy_and_matrix_confusao()