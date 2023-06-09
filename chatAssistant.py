#testing the use of openAi for a more automated tool
#first let get the basic interactions down.
import os, openai, pickle, pyperclip, time, pyllamacpp
import gpt4all
#import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms import GPT4All
from langchain.llms import LlamaCpp
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from duckpy import Client
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pdf2image import convert_from_path
#from langchain.prompts.prompt import PromptTemplate

#db = Chroma

#model = GPT4All('')

def duckSearch(message):
    client = Client()
    results = client.search(message)
    try:
        if len(results)>0:
            for result in results:
                print(result)
                keep = input('keep y/n/q?:')
                if keep.lower() == 'y':
                    newResult = result['title'] +','+ result['url']
                    pyperclip.copy(newResult)
                    print('will wait here..\nthe result is on your clipboard\nhit any key when ready to proceed')
                elif keep.lower() =='q':
                    break
                else:
                    next
    except Exception as ex:
        print(f'error {ex}')
    return results

def addTextFile():
    filePath = input('file name with path: ')
    print(f"adding file {filePath}")
    loader = TextLoader(filePath)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    db = Chroma.from_documents(texts, embeddings, persist_directory='db')
    db.persist()
    print(f"file done...")
    #documents = loader.load()
    #with open(filePath) as f:
    #    myText = f.read()
    #texts = chunkItDown(documents)
    #updateDb(texts,embeddings,filePath)
    #updateDb(docs, embeddings,filePath)
    return docs

def importPDF():
    filePath = input('file name with path: ')
    print(f'processing file {filePath}, be patient..\n')
    loader = PyPDFLoader(filePath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    db = Chroma.from_documents(texts, embeddings, persist_directory='db')
    db.persist()
    print(f"file done...")
    '''
    texts = ''
    for i in range(len(documents)):
        texts = texts+ (documents[i].page_content)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_text(texts)
    texts = chunkItDown(docs)
    updateDb(texts, embeddings, filePath)
    '''
    #adb.afrom_texts(texts,embeddings)
    #adb.from_texts(texts,embeddings)
    #text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    #docs = text_splitter.split_documents(documents)
    #db1 = Chroma(persist_directory='newdb')
    #db1.persist()
    #adb.from_documents(docs, embeddings)


    #db.from_documents(docs, embeddings)

    '''
    documents = loader.load_and_split()
    reader = PyPDF2.PdfReader(filePath)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text+=text
    '''
    print('pdf file is done...\n')
    return texts

def chunkItDown(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    texts = text_splitter.split_text(documents)
    #texts = []
    #for text in documents:
    #    texts.append(text_splitter.split_text(text.page_content))

    '''
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap= 200,
        length_function= len
    )
    texts = text_splitter.split_text(raw_text)
    '''
    return texts


def runLocalModel(messages, llm):
    #GPT4All
    # Run model on prompt
    response = qa.run(messages)
    #response = llm.chat_completion(messages,default_prompt_footer=False, verbose =False, streaming=False, default_prompt_header=True )
    reply = response["choices"][0]["message"]["content"]
    return reply, response["choices"][0]["message"]

def gptModel(messages):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    #messages.append({"role":"assistant","content":reply})
    return reply, messages



def createChain():
    #qa = LLMChain(
    docSearch = FAISS.load_local('docIndex',embeddings)
    model_path = "ggml-gpt4all-j-v1.3-groovy.bin"
    llm = GPT4All(model=model_path, n_ctx=1500, backend='gptj', verbose=False)
    #retriever = docSearch.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docSearch.as_retriever(search_kwargs={"k": 3}),
        return_source_documents = True,
        verbose = False
    )
    return qa

def updateDb(texts,embeddings,filePath):
    docsearch = Chroma(persist_directory='db',embedding_function=embeddings)
    docsearch = docsearch.from_texts(texts, embeddings, metadatas=[{"source": f"{filePath} {i} of {len(texts)}"} for i in range(len(texts))], persist_directory="db")
    docsearch.persist()
    docsearch = None


def importFile(fileName):
    if fileName != None:
        if fileName[-4:].lower() == '.pdf':
            loader = PyPDFLoader('UnconventionalWarfare.pdf')
        elif fileName[-4:].lower() == '.txt':
            loader = TextLoader(fileName)

        documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print(len(texts))
        db = Chroma.from_documents(texts, embeddings, persist_directory='db')
        db.persist()
        print('file done..')
        return

def newModels(curModel, llm):
    mymodels = gpt4all.GPT4All.list_models()
    modelNames = []
    selection = input('Try new Model y/n?')
    if selection.lower() == 'y':
        for model in mymodels:
            print(f"Model {model['filename']} \nDescription{model['description']}")
            tryMe = input('try this one? y/n')
            if tryMe.lower() == 'y':
                modelName = model['filename']
                dlModel(modelName)
                break
    else:
        modelName = curModel
        dlModel(modelName)

    llm = createLLM(modelName, llm)
    return llm

def dlModel(filename):
    modelName = filename
    if os.path.isfile(path=modelName) == False:
        gpt4all.GPT4All.download_model(modelName, model_path='')
        print(f"your model: {modelName} has been downloaded")
    return


def createLLM(modelName, llm):
    streaming = False
    verbose = False
    echo = False
    temp = .31
    n_threads = 10
    top_p = .995
    top_k = 40
    n_ctx = 1000

    repeat_penalty = 1.1
    if '-j' in modelName:
        backend = 'gptj'
    elif '-mpt' in modelName:
        backend = 'mpt'

    llm = GPT4All(model=modelName,
                  backend=backend,
                  streaming=streaming,
                  echo=echo,
                  temp=temp,
                  n_ctx=n_ctx,
                  )
    print(f"using a backend of {backend}")
    return llm

    '''
    streaming = streaming,
    echo = echo,
    temp=temp,
    top_p = top_p,
    top_k = top_k,
    repeat_penalty = repeat_penalty,
    n_threads = n_threads
    '''



##################
#   start main   #
##################

stt = time.time()
response = []
messages = []
docs = None
qa_chain = None
embeddings = None
llm = None
fileName = None

modelName = 'ggml-gpt4all-j-v1.3-groovy.bin'
llm =newModels(modelName,llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#fileName = 'test.txt'
#fileName = 'UnconventionalWarfare.pdf'


db = Chroma(persist_directory='db',embedding_function=embeddings)
'''
llm = GPT4All(
    model="ggml-gpt4all-j-v1.3-groovy.bin",
    n_ctx=1000,
    backend="gptj",
    verbose=False,
    streaming = False,
    echo = False,
    temp = .31,
)
'''
#llm = newModels(modelName, llm)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)
'''
res = qa('from the document tell me about Roz, who she works with and what she knows.')

while True:
    question = input('Enter Question: ')
    res = qa(question)

print(res["result"])

res = qa(f"who was the invoice from")
res = qa('in the document, can you tell me how much the total invoice amount is?')

#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
'''
#os.environ['OPENAI_API_KEY'] = openai.api_key
readFlag = False

#db = Chroma(persist_directory='db',embedding_function=embeddings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#modelName = 'ggml-gpt4all-j-v1.3-groovy.bin'
#modelName = 'ggml-mpt-7b-chat.bin'
#llm = createLLM(modelName)
print(f"Currently running model: {modelName}")

#resp = qa('tell me about this invoice.')
#print(resp)
#embeddings = HuggingFaceEmbeddings()

docsearch = Chroma(persist_directory='db',embedding_function=embeddings)
#db = Chroma(persist_directory='db',embedding_function=embeddings)
#llm = GPT4All(model='ggml-gpt4all-l13b-snoozy.bin',n_ctx=1024,  verbose=False)

initPrompt = "Your name is Roz, you work for me, George Wilken we work together in my office. You are my assistant and you will answer my questions as concise as possible unless instructed otherwise. You are knowledgable in python programming and industrial automation sales. You are not very chatty but you have a good sense of humor. You also like to like to provide quotes relevant to events and questions. If you don't know an answer you will say I don't know. Is that clear?"

conversation = ConversationChain(
    llm=llm,
    verbose = False,
)
'''
#conversation.prompt = prompt
PROMPT = PromptTemplate(
    template=prompt
)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Roz", human_prefix = 'George')
)
'''
print('Roz:')
#reply = conversation.run(initPrompt)


readFlag = True

try:
    with open("conversation_history.pkl", "rb") as f:
        messages = pickle.load(f)

except Exception as ex:
    print('we had an issue loading history so starting fresh')
    messages = []
    print('# Responses from your assistant will be automatically copied to your clipboard.\n Chat history will be reloaded autoamaitcally.\n')
    myprompt = input("What type of chatbot would you like to create?\n")
    messages.append({"role": "system", "content": myprompt})
    print('Say hello to your new assistant')

#if messages[-1]['content'] != None:
#    pyperclip.copy(messages[-1]['content'])

reply = None
print('Enter just M for menu:')

while input !="quit()":
    while len(messages)>10:
        del messages[1]

    message = input("George: ")
    stt = time.time()
    if message.lower() == 'm':
        print("-"*20 + " Menu "+"-"*20)
        print("\t>t to try new model\n\t>d to doc search\n\t>s to search via duck\n\t>4 for ChatGPT\n\t1 for load a file\n" + "-"*46)
        next
    elif message.lower() == '>t':
        llm = newModels(modelName,llm)

    elif message == '>1':
        readFlag = False
        if readFlag == False:
            try:
                fileName = input('Enter file name with path: ')
                print('importing file\n')
                texts = importFile(fileName)
                reply = None
                readFlag = True
            except Exception as ex:
                print(f'we had an error with the import {ex}')

    elif message == 'quit()':
        break
    elif '>s' in message[0:3].lower():
        print('lets search for it.')
        myMessage = message[3:]
        #docsearch(myMessage)
        duckSearch(myMessage)
    elif '>4' in message[0:3]:
        model = 'gpt-3.5-turbo'
        message = message[3:]
        messages.append({"role": "user", "content": message})
        reply, messages = gptModel(messages)
    elif readFlag == True and '>d' in message[0:3]:
        #if readFlag == True and message:
        message = message[3:]
        query = message


        docsearch = Chroma(persist_directory="db", embedding_function=embeddings)
        docsearch.persist()


        if len(message)==0:
            user_input = input("What's your question: ")
            #print(user_input)
            myInput = user_input.strip()
        else:
            myInput = message.strip()
        llm.temp = 0
        try:
            reply = qa(message)
        except Exception as ex:
            print(f"we had an error: {ex}")

        docsearch.persist()

    else:
        if len(message) >2:
            print('Roz:\t')
            docsearch = Chroma(persist_directory='db', embedding_function=embeddings)
            docsearch.persist()

            reply = conversation.run(message)


    if reply == None:
        next
    elif reply == '':
        print('got no answer from local model, answering with chatGPT')
        #if local model fails to generate content, use chatGPT
        reply, messages = gptModel(messages)

    if reply != '' and reply != None:
        #print(f'Roz: {reply}')
        messages.append({"role": "assistant", "content": reply})

        try:
            with open("conversation_history.pkl", "wb") as f:
                pickle.dump(messages, f)
        except Exception as ex:
            print(f'error occurred in writing pickle {ex}')


        if isinstance(reply,dict):
            if 'answer' in dict.keys(reply):
                reply = reply['answer']
            if 'result' in dict.keys(reply):
                reply = reply['result']
        if reply !='':
            pyperclip.copy(reply)
        elapsed=time.time()-stt
        print(f'elapsed time: {round(elapsed,2)} seconds.')
        #print("\nRoz: " + reply + "\n")
    #print('OUT')