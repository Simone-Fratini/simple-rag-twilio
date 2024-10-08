# Simple RAG Twilio



## Overview



This project integrates several technologies to create a Flask-based web application that answers user queries through WhatsApp. The application extracts content from PDF documents, indexes it using FAISS, and uses OpenAI's model to generate responses to queries based on the indexed content.



## Features



- **PDF Text Extraction**: Extracts text from PDF files for processing.

- **Text Indexing**: Uses FAISS to index the extracted text for efficient retrieval.

- **Query Answering**: Uses OpenAI's models to answer queries based on the indexed text.

- **WhatsApp Integration**: Allows users to interact with the system via WhatsApp using Twilio.



## System Requirements



- Python 3.8 or higher

- Operating System: Compatible with Windows, MacOS, Linux



## Dependencies



- Flask

- Twilio

- PyMuPDF (fitz)

- FAISS

- OpenAI

- LangChain (and related packages)

- ngrok



## Setup Instructions



### 1. Clone the Repository



Clone the repository to your local machine:

\`\`\`bash

git clone <repository-url>

cd <repository-directory>

\`\`\`



### 2. Install Dependencies



Install the necessary Python libraries:

\`\`\`bash

pip install -r requirements.txt

\`\`\`



### 3. Environment Variables



Set up the required environment variables or replace the placeholders in the code:

- TWILIO_PHONE_NUMBER

- TWILIO_ACCOUNT_SID

- TWILIO_AUTH_TOKEN

- OPENAI_API_KEY

- PDF_PATH

- INDEX_PATH



### 4. Run the Application



Start the Flask application:

\`\`\`bash

python app.py

\`\`\`



### 5. Open a New Terminal



Start ngrok to expose your local server to the internet:

\`\`\`bash

ngrok http 5000

\`\`\`



### 6. Configure Twilio



Copy the ngrok webhook URL to your Twilio WhatsApp sender's configuration to handle incoming messages.



## Usage



Once the application is running, you can interact with it via WhatsApp:

- Send queries related to the content of the loaded PDF to the Twilio WhatsApp number configured in the application.



## Contributing



Contributions to the project are welcome. Please ensure to update tests as appropriate.


