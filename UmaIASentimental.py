"""
Flávio Lao. Uma IA Sentimental.
Este é meu primeiro estudo com IA. Esta, é treinada com uma lista de frases, onde cada uma recebe um valor positivo ou negativo.
Quando a IA é executada, ela pedirá um frase, e ira correlacionar ela com sua base de dados, e tentar inferir se a frase é positiva ou negativa, avaliando cada token das frases e o seu valor definido.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer  # Mais inteligente que CountVectorizer
from sklearn.naive_bayes import MultinomialNB

humamResult = ["Negativo", "Positivo"] # Resultados possiveis
vetorizador = TfidfVectorizer(min_df=1)  # Considera palavras mesmo que apareçam só 1 vez
modelo = MultinomialNB() #Modelo

def inicio(): #
    print("Digite uma frase simples, uma afirmação positiva ou negativa (Ex: Sorvete é uma delicia!)")
    return str(input("Sua frase : "))

def carregaJSON():
    texto = []
    classes = []
    with open('promptOriginal.json', encoding='utf-8') as f:
        result = json.load(f)
    
    texto = (result.get("prompts")) # Frases do banco de dados
    classes = (result.get("valores")) # Valores de cada frase

    # Esta ia tem a capacidade de aprender novas frases, as frases novas são guardadas em um arquivo separado
    with open('promptExtra.json', encoding='utf-8') as f:
        result = json.load(f)
    
    texto += (result.get("prompts")) # Conjunto de todas as frases
    classes += (result.get("valores")) # Conjunto de todos os valores

    return texto, classes

def preProcess(texto):
    return vetorizador.fit_transform(texto)

def treinamento():
    # PASSO 1: Dados brutos (arquivo promptOriginal e promptExtra)
    frases, valores = carregaJSON()
    # PASSO 2: Pré-processamento
    X = preProcess(frases)
    # PASSO 3: Treinamento
    modelo.fit(X, valores)

    return frases, valores 


if "__main__" == __name__ :
    print('...digite "q ou quit" para sair...')

    # Treinamento da IA
    textoOri, classesOri = treinamento()

    while True:
        txtInput = inicio().strip().lower()
        if txtInput in ('q', 'quit', 'exit'): #Sair
            break
        else: #Se NÃO SAIR
            #Caso a frase não exista no banco de dados atual, tenta incluir a frase primeiro
            if not any(word in vetorizador.vocabulary_ for word in txtInput.split()):
                print("⚠️ Frase com palavras não vistas no treino!")
                adicionar = input("Deseja adicionar esta frase ao banco de dados? (s/n)")
                if (adicionar.lower() in 's'):
                    novoValor = int(input(f"Qual o valor desta frase: {txtInput} ? (0,1) : "))

                    #Adiciona a nova frase na lista de frases
                    textoOri.append(txtInput)
                    classesOri.append(novoValor)

                    with open('promptExtra.json', encoding='utf-8') as f:
                        result = json.load(f)

                    frasesPromptExtra = (result.get("prompts")) # Conjunto das frases em promptExtra
                    valoresPromptExtra = (result.get("valores")) # Conjunto dos valores em promptExtra
                    frasesPromptExtra.append(txtInput) 
                    valoresPromptExtra.append(novoValor) 

                    #Todos os dados de promptExtra, mais a frase e valor atual
                    data = {"prompts" : frasesPromptExtra, "valores" : valoresPromptExtra}

                    #Atualiza o arquivo promptExtra
                    with open('promptExtra.json', 'w', encoding='utf-8') as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=4)

                    #Treinar a IA novamente
                    textoOri, classesOri = treinamento()

            # PASSO 4: Predição 
            resultado = modelo.predict(vetorizador.transform([txtInput]))
            if (resultado[0] == 1):
                print('\033[42;30m  Meus Transistores Emocionais sentem que esta é uma frase POSITIVA 󰇵 \033[0m')
            else:
                print('\033[41;30m  Meus Transistores Emocionais sentem que esta é uma frase NEGATIVA  \033[0m')

            probas = modelo.predict_proba(vetorizador.transform([txtInput]))[0]
            print(f"Positivo: {probas[1]*100:.2f}% | Negativo: {probas[0]*100:.2f}%")

            print("---#--- Digite 'q' para sair, ou ...")
