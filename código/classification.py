from pre_processing import Processing 
from joblib import load
import pandas as pd
from word2vec import Representation

class Classificador():
    ''''''
    def __init__(self):
        pass


    def processar(self, arquivo_csv, placa:str = 'null'):
        '''Realiza a classificação das urgencias.'''
        #Lendo o arquivo .csv e transformando-o para o fomato de dataFrame
        df = pd.read_csv(arquivo_csv)
        #Validando a placa
        df = self._validar_placa(df, placa)
        if len(df.index) != 0:
            #Criando a coluna de target(Classificação)
            df['target'] = self._classificar(df)
            #CRiando a coluna de resolvido(Saber se o problema já foi resolvido)
            df['resolvido'] = 0
            #Chamando a função para mudar o valor da linha
            self._mudar_resolvido(df, 3)
            print(df)
        else:
             print(f"A placa {placa} não existe na base de dados!!")
    
    

    def _validar_placa(self, df:pd.DataFrame, placa):
        '''Verifica se a placa existe na base de dados'''
        #Verificando se foi passado uma placa que existe na base de dados
        if placa != 'null':
            #Vericar se a placa existe na base de dados
            if len(df.index) != 0:
                #Selecionando as linhas que possuem a placa requisitada
                df = self._select_placa(df, placa)
                return df
            else:
                #A placa não existe
                return None


    def _classificar(self, df, coluna='dados_tokenizados'):
        #Aplicando o pré-processamneto a base de dados
        Processing().processing(df)
        #Criando a representação dos dados
        representacao = Representation(175, df[coluna])
        repr_word2vec = representacao.gerar_representacao()
        #Lendo o classificador
        clf = load('classificador.joblib')
        #Executando as predições 
        predicoes = clf.predict(repr_word2vec)
        return predicoes


    def _select_placa(self, df:pd.DataFrame, placa):
        '''Seleciona uma placa especifica da base de dados.'''
        return  df.loc[df['placa'] == placa]



    def _mudar_resolvido(self, df, index):
        '''Muda o status do problema para resolvido/não-resolvido'''
        if index in df.index:
            if df.iloc[index]['resolvido'] == 0:
                df.at[index, 'resolvido'] = 1
            else:
                df.at[index, 'resolvido'] = 0
        else:
            print(f"O {index} não existe na seleção atual")


#####################################################################
#Exemplo de como executar essa classe.
#####################################################################
#Iniciando a classe de classificação.
teste = Classificador()
#classificar as informações de entrada.
teste.processar('dados_viaturas_pdi.csv', "AQW1020")



        
        