'''
 * @author George Vieira and Rodrigo Luiz
 * 
 * * Created:   22.04.2022
 * 
 * (c) Copyright by Poli-UPE
 * 
'''

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return "wello Word"


@app.get("/treinar")
def treino():
    return "Treinamneto"


@app.get("/classificar")
def classificar():
    return "classificar"