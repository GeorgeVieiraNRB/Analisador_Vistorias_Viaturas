import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords



#pacote necessario para fazer a remoção de stopwords
nltk.download('stopwords')

#lista com stopwords
stopwords = stopwords.words('portuguese')


class Processing():
    '''Realiza o pré-processamento dos dados de entrada.'''

    def __init__(self):
        pass 
       
    
    def _tokenization(self, alteracoes:str):
        '''Retira os tokens de uma texto. Os tokens podem ser pontuação ou numeral.
        
        Args:
            info (string): Frase que será tokenizada.
        
        return:
            dados_tokenizados (lista): Dados tokenizados.   
        '''
        #Aqui aplica-se expressões regulares(Detecção de pontuação e numeral)
        alteracoes=alteracoes.lower()
        dados_tokenizados = list()
        tokenizer = RegexpTokenizer(r'[A-z]\w*') 
        # Criar a variável com os tokens sem pontuação e numeral oriundos do texto.
        dados_tokenizados = tokenizer.tokenize(alteracoes)
        return dados_tokenizados


    def _removeStopwords(self, tokens:list()):
        '''Remoção de stopwords, ou seja, remoção de palavras que podem ser consideradas irrelevantes para algumas análises, como artigos e preposições).
        
        Args:
            tokens (lista): Dados tokenizados.

        return:
            dados_sem_stopwords (lista): Dados com as stopwords removidas.
        
        '''
        #Lista com palavras que não devem serem removidas
        exception_list = ["sem"]
        # Criação da variável contendo os tokens sem stopwords
        dados_sem_stopword = [word for word in tokens if word not in stopwords or word in exception_list]
        return dados_sem_stopword
    

    def processing(self, file_pd:pd.DataFrame):
        '''Executa o pré-processamento e adiciona as linhas ao dataFrame.
          
        Args:
            file_pd (DataFrame): 

        return:
            file_pd (DataFrame): 
            
        '''
        #Adicionando a coluna 'dados_tokenizados' ao dataFrame (Arquivo .csv)
        file_pd['dados_tokenizados'] = None
        #Adicionando a coluna 'dados_sem_stopwords' ao dataFrame (Arquivo .csv)
        file_pd['dados_sem_stopwords'] = None
        #Percorrendo o arquivo e adicionando as linhas
        for index, value in file_pd.iterrows():
            #Aplicando a tokenização
            tokenizacao = self._tokenization(value['alteracoes'])
            #Armazenando os dados tokenizados na base de dados
            file_pd.at[index,'dados_tokenizados'] = tokenizacao
            #Aplicando a remoção de stopwords
            sem_stopwords = self._removeStopwords(tokenizacao)
            #Armazenando os dados sem stopwords na base de dados
            file_pd.at[index,'dados_sem_stopwords'] = sem_stopwords
