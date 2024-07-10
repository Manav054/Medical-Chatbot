# End to end Medical Chatbot using Gemini

## steps to run the project

### Clone the repository
```bash
Project repo: http://github.com/
```
### Create environment
```bash
conda create -n mchatbot python=3.9 -y
```
### Activate Environment
```bash
source activate mchatbot
```
### Install required dependencies
```bash
pip install -r requirements.txt
```
### Create a `.env` file in the root directory and add your Pinecone and Gemini Credentials in it.
```ini
PINECONE_API_KEY=""
GOOGLE_API_KEY=""
```

### Run the following code to create embeddings in your pinecone vectorstore
```bash
python store_index.py
```

### To run the medical-chatbot
```bash
flask run
```

## Techstack used
- Python
- Langchain
- Flask
- Pinecone
- Google Gemini