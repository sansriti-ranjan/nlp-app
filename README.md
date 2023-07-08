# Neural Machine Translation Web App

## Running the Project Locally
Clone the repo, install the node.js dependencies, start the frontend and the backend.
```bash
git clone https://github.com/sansriti-ranjan/nlp-app.git
cd nlp-app
```

### Start the React frontend
In a terminal run:
```bash
cd frontend
npm install
npm start
```

### Start the FastAPI backend
Open another terminal and run:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
