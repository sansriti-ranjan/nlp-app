from fastapi import FastAPI # main fastapi functionality
# CORS allows frontend to communicate with backend 
# (frontend and backend have different origins (ports))
from fastapi.middleware.cors import CORSMiddleware 
from NMT import translate

# App object
app = FastAPI()

# Set up permissions between frontend (react) and backend (fastapi)
origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# root endpoint with a hello message
@app.get('/')
def root():
    return {'message': 'Hey'}


# endpoint to trigger the translation
@app.post('/translate')
def read_item(text_message):
    print('hi')
    print(text_message)

    translated_sent = translate(text_message, 'cpu')


    return {'input': text_message, 'target': translated_sent}