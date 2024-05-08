# OTheraPy Voice Chatbot
## Goal
To build a domain expert voice chatbot that acts as a world-class Doctor of Occupational Therapy for Asperger Adults.

## Resources
**Google Colab**
A Google Colab T4 instance will be used to train and run the initial document encoding with a more advanced model on specific medical data, enhancing the semantic depth and accuracy of the encoded documents.

**Local machine**
The development and deployment will be done on a 2017 15-inch MacBook Pro with 16GB RAM and a 2.8 GHz Quad-Core Intel Core i7 processor. This machine will be used for:

1. Running the retriever model to perform nearest neighbor search in the vector database.
2. Running the generator model to generate responses.
3. Performing real-time speech-to-text (STT) and text-to-speech (TTS) processing using vocode-python as part of the chatbot application.

**Render.com**
Render.com's free tier instances (512MB RAM, 0.1 CPU) will be used for:
1. Hosting static content.
2. Running background tasks for log processing and data analysis.
3. Handling asynchronous batch processes that don't require instant execution.

**Weaviate**
Weaviate, a cloud vector database, will be used to store and retrieve previous user-bot interactions and other relevant dynamic data efficiently. This will enable the chatbot to quickly pull relevant information for generating contextual responses.

## Models
### Prioritization of Semantic Richness
A key emphasis is placed on selecting models that can provide rich semantic representations while being computationally feasible on the local machine.

### Document Encoding Model
Using the computational power of Google Colab's T4 instance, a high-capacity model will be employed for initial document encoding to ensure deep semantic understanding:
- `roberta-large` - Known for its robust performance in capturing complex language nuances, ideal for processing medical and therapy-related texts.

### Retriever Model
The retriever model used on the local machine:
- `sentence-transformers/all-mpnet-base-v2` (420 MB) - It efficiently converts queries into vector representations and performs accurate nearest neighbor searches, essential for real-time retrieval and response generation.

### Generator Model
For generating coherent and contextually relevant responses based on the retrieved information:
- `unsloth/Phi-3-mini-4k-instruct` - Optimized for generating accurate responses in the context of occupational therapy, making it a perfect fit for the chatbot's role.

## Environment Management 
1. **Containerize the Application**: Define a `Dockerfile` that specifies the environment, including the operating system, programming language runtime, and libraries needed for the chatbot. This file will also outline the process to build the application image.
2. **Manage Dependencies**: Specify all dependencies explicitly in the `Dockerfile`. Use Docker's layer caching to speed up builds by reusing layers that haven't changed.
3. **Environment Consistency**: Use Docker Compose to define and run multi-container Docker applications. With Compose, you can configure application services in a `docker-compose.yml` file, ensuring that all services use the same Docker environment and network.
4. **Development and Testing**: Use Docker to replicate the production environment locally. This allows developers and testers to work in an environment that matches production, reducing the "it works on my machine" problem.
5. **Deployment**: Deploy the Docker containers on both development and production environments. Use orchestration tools like Kubernetes or Docker Swarm to manage container deployment, scaling, and management across clusters of machines.

## Docker for Dependency and Environment Management
Using Docker to manage dependencies and ensure environmental consistency is vital for the seamless deployment and operation of the OTheraPy Voice Chatbot across different platforms.
