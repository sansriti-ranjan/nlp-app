"""FastAPI main file"""
from fastapi import Request, FastAPI # main fastapi functionality
# CORS allows frontend to communicate with backend 
# (frontend and backend have different origins (ports))
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class Sentence(BaseModel):
    """pydantic class to accept the post request"""
    text_message: str

# root endpoint with a hello message
@app.get('/')
def root():
    return {'message': 'Hey'}


@app.post('/translate')
def read_item(sentence: Sentence):
    """Endpoint to recieve user input and call our translation model"""

    translated_sent = translate(sentence.text_message, 'cpu')

    return {'input': sentence.text_message, 'target': translated_sent}