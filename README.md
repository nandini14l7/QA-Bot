# QA Bot

**QA Bot** is an intelligent conversational assistant designed to help users retrieve answers from uploaded documents, including PDFs, DOCX, and TXT files. Powered by advanced AI models like Llama3 and enhanced by **Retrieval-Augmented Generation (RAG)**, QA Bot provides fast, accurate, and context-aware answers to your queries. 

In addition to supporting text-based queries, the bot also offers voice-based interactions, enabling users to speak their questions and receive spoken responses.

## Features

- **Document-based Q&A**: Upload documents in formats such as PDF, DOCX, or TXT, and ask questions based on their content.
- **Voice Interaction**: Ask questions using voice, and hear the answers through speech synthesis.
- **Efficient Document Processing**: Documents are split into smaller chunks and indexed for quick retrieval.
- **Retrieval-Augmented Generation (RAG)**: The bot leverages RAG to ensure accurate and contextually relevant answers by retrieving and processing the most relevant portions of documents.
- **User-friendly Interface**: Built with **Streamlit**, making it easy to upload files, ask questions, and interact with the bot.

## Requirements

To run this project, you will need:

- Python 3.7+
- Streamlit
- Langchain (for Llama3 and RAG integration)
- SpeechRecognition (for voice input)
- gTTS (Google Text-to-Speech for voice output)
- FAISS (for efficient document indexing)

## Installation

Follow these steps to get your project running locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/nandini14l7/QA-Bot.git
    cd QA-Bot
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## How It Works

1. **Upload Documents**: You can upload PDFs, DOCX, and TXT files through the **Streamlit** interface.
2. **Ask Questions**: Type or speak your questions related to the documents you’ve uploaded. The bot will respond with the most relevant information based on the document content.
3. **Voice Interaction**: Press the microphone button to speak your query. The bot will process the speech, convert it to text, and generate a spoken response.
4. **Document Processing**: The bot processes the document content, splits it into smaller chunks, and indexes these chunks using **FAISS** for efficient retrieval.
5. **RAG Integration**: **Retrieval-Augmented Generation (RAG)** improves the bot’s ability to generate accurate answers by retrieving the most relevant document sections before providing a response.

## Usage

Once the app is running:

1. **Upload a Document**: Click on the file uploader in the sidebar to upload your document(s).
2. **Ask a Question**: Type your query in the input box or click the microphone button to speak your question.
3. **Get an Answer**: The bot will retrieve the relevant portions of the document and provide a response.
4. **Voice Responses**: The answer will be read aloud, providing a hands-free experience.

## Technologies Used

- **Llama3**: A state-of-the-art language model for generating text-based responses.
- **Retrieval-Augmented Generation (RAG)**: A technique for enhancing the bot’s ability to retrieve relevant information from documents before generating a response.
- **Streamlit**: A framework for building web applications with Python, used for creating the bot's user interface.
- **FAISS**: A library for efficient similarity search, used for indexing and searching document chunks.
- **gTTS (Google Text-to-Speech)**: Used for converting text-based responses into speech.
- **SpeechRecognition**: Used for converting voice input into text.

## Contributing

If you would like to contribute to this project:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch and open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Llama3** and **Langchain** for the advanced language model and tools that power the bot’s Q&A capabilities.
- **FAISS** for enabling fast document retrieval and indexing.
- **Streamlit** for providing an easy-to-use framework for creating the interactive web app.
- **gTTS** and **SpeechRecognition** for enabling the voice interaction feature.

## Contact

If you have any questions, feel free to open an issue or contact me directly at nandinishukla1407@gmail.com.

