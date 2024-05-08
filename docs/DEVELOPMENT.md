### High-Level Overview of Development Plan

**Objective:**
The goal is to develop "OTheraPy Voice Chatbot," a sophisticated voice-enabled chatbot specialized in Occupational Therapy for Asperger Adults. This system will utilize advanced natural language processing (NLP) models to understand and respond to user queries effectively, transforming voice input into meaningful responses.

**Main Components:**
1. **Speech-to-Text (STT)**: Converts user's spoken language into text.
2. **Chatbot Engine**: Consists of a document retriever and response generator.
3. **Text-to-Speech (TTS)**: Converts chatbot's text responses back into spoken words.
4. **Web User Interface (UI)**: Provides an interactive interface for users to interact with the chatbot.

**Technical Architecture:**
- **Local Machine**: Main processing including NLP model operations.
- **Google Colab**: For intensive model training and initial document encoding.
- **Render.com**: To host lightweight services and static content.
- **Weaviate**: To handle vector-based data storage and retrieval for dynamic content.

### Development Environment & Tools

1. **Docker & Docker Compose**: For containerization of the application, ensuring consistent environments across different stages of development and deployment.
2. **Python & Virtual Environments**: Core programming language and isolated environments for managing package dependencies.

### Core Application Logic Development

1. **STT Integration**:
   - Utilize `coqui_stt` for real-time voice to text conversion.
   
2. **Document Retriever**:
   - Implement using `sentence-transformers` for efficient semantic search.
   - Store and retrieve data from Weaviate.

3. **Response Generator**:
   - Employ `unsloth/Phi-3-mini-4k-instruct` for generating responses based on the retrieved context.

4. **TTS Integration**:
   - Implement using `coqui_tts` to convert text responses into speech.

5. **Web UI Development**:
   - Develop using Streamlit for an interactive user interface.
   - Deploy locally for testing and use Render.com for production.

### Containerization and Deployment Strategy

- **Dockerfiles and Docker Compose**: For defining the application's containers.
- **Testing**: Conduct thorough testing in Docker environments.
- **Deployment**: Utilize Heroku for deployment using Docker containers.

### Implementation Details

#### Environment Setup
**Requirements:**
- Docker and Docker Compose
- Python 3.8+
- Google Colab for model training

**1. Installing Docker and Docker Compose:**
```bash
brew install docker docker-compose
```
Verify the installation with:
```bash
docker --version
docker-compose --version
```

**2. Setting Up Python Virtual Environment:**
Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Installing Python Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Google Colab Setup:**
Use Google Colab for training models with high-performance GPUs. Mount Google Drive for easy data transfer between your local machine and the cloud:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Configure and run notebooks directly in Google Colab to utilize its computational resources effectively.

#### Project Structure
```
OTheraPy Voice Chatbot/
|-- docker-compose.yml
|-- Dockerfile
|-- .env
|-- requirements.txt
|-- app/
    |-- __init__.py
    |-- main.py
    |-- stt.py
    |-- tts.py
    |-- chatbot/
        |-- __init__.py
        |-- retriever.py
        |-- generator.py
    |-- ui/
        |-- __init__.py
        |-- streamlit_app.py
|-- data/
    |-- processed/
|-- models/
    |-- stt/
        |-- model.tflite
        |-- scorer.scorer
    |-- tts/
        |-- config.json
        |-- tts_model.pth
    |-- nlp/
        |-- roberta/
        |-- unsloth/
|-- tests/
    |-- __init__.py
    |-- test_main.py
|-- docs/
    |-- DEVELOPMENT.md
```

#### Configuration Files and Environment Variables
**.env File Setup:**
Create a `.env` file in the root directory to manage environment variables securely:
```plaintext
# Environment Configuration
API_KEY=your_api_key_here
DATABASE_URL=your_database_url_here
```
Use the `python-dotenv` package to load these variables in your application.

#### Dependencies List
Ensure all dependencies are listed in your `requirements.txt`:
```plaintext
flask==1.1.2
transformers==4.5.1
sentence-transformers==2.0.0
coqui-stt==0.9.3
```

#### Database Setup
**Weaviate Configuration:**
Set up Weaviate to manage vector storage for the chatbot. Define the schema and initialize the database:
```bash
weaviate-cli schema apply -f schema.json
```

#### Integration Details
**Connecting Components:**
Ensure all components are integrated smoothly:
- STT outputs feed directly into the chatbot engine.
- The chatbot engine uses retriever and generator models for processing and responding.
- Responses are then passed to TTS for vocal output.

**Error Handling and Logging:**
Implement robust error handling and logging to track and manage runtime issues effectively.

#### Development
**Speech-to-Text (STT) Integration:**
Integrate `coqui_stt` for real-time voice to text conversion in `app/stt.py`.

**Document Retriever Setup:**
Use `sentence-transformers` for embedding and retrieval in `app/chatbot/retriever.py`.

**Response Generator Integration:**
Set up response generation with `unsloth/Phi-3-mini-4k-instruct` in `app/chatbot/generator.py`.

**Text-to-Speech (TTS) Setup:**
Integrate `coqui_tts` for text-to-speech functionality in `app/tts.py`.

**Streamlit Web Interface:**
Develop a Streamlit UI for interaction in `app/ui/streamlit_app.py`.

#### Continuous Integration/Continuous Deployment (CI/CD)
**CI/CD Pipeline Setup:**
Configure GitHub Actions or GitLab CI for automated testing and deployment:
```yaml
name: Python application

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Test with unittest
      run: |
        python -m unittest discover -s tests
    -

 name: Deploy to Heroku
      run: bash deploy_heroku.sh
```

#### Deployment
Deploy to Heroku using Docker. Use the Heroku CLI to manage the deployment process:
```bash
heroku login
heroku container:login
heroku create your-app-name
heroku container:push web --app your-app-name
heroku container:release web --app your-app-name
```
