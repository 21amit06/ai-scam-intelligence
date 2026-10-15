import joblib
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

scam_vectorizer = joblib.load("models/scam_vectorizer.pkl")
scam_model = joblib.load("models/scam_model.pkl")

fake_job_vectorizer = joblib.load("models/fake_job_vectorizer.pkl")

fake_job_model = joblib.load("models/fake_job_model.pkl")

url_vectorizer = joblib.load("models/url_vectorizer.pkl")

url_model = joblib.load("models/url_model.pkl")

print("All ML models loaded successfully!")


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"<.?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def get_risk_level(probability):
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def predict_scam_message(text):
    cleaned = clean_text(text)
    vector = scam_vectorizer.transform([cleaned])
    probability = (scam_model.predict_proba(vector)[0][1])
    risk = get_risk_level(probability)
    return {
        "risk_score": round(float(probability * 100), 2),
        "risk_level": risk
    }


def predict_fake_job(text):
    cleaned = clean_text(text)
    vector = (fake_job_vectorizer.transform([cleaned]))
    probability = (fake_job_model.predict_proba(vector)[0][1])
    risk = get_risk_level(
        probability
    )
    return {
        "risk_score": round(float(probability * 100), 2),
        "risk_level": risk
    }


# URL PREDICTION
def predict_url(url):
    vector = (url_vectorizer.transform([url]))
    probability = (url_model.predict_proba(vector)[0][1])
    risk = get_risk_level(probability)

    return {
        "risk_score": round(float(probability * 100), 2),
        "risk_level": risk
    }


from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"C:\Users\kumar\PycharmProjects\Ai fraud detection\knowledge_base\scam_knowledge.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = (text_splitter.split_documents(documents))
print(len(chunks))

from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="scam_knowledge"
)

# CREATE RETRIEVER
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3})

# LOAD LLAMA
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# FINAL SCAM ANALYSIS
def analyze_scam(input_type, user_input):
    # ML PREDICTION
    if input_type == "message":
        risk_result = (
            predict_scam_message(
                user_input
            )
        )
    elif input_type == "job":
        risk_result = (
            predict_fake_job(
                user_input
            )
        )

    elif input_type == "url":
        risk_result = (
            predict_url(
                user_input
            )
        )
    else:
        return "Invalid input type.Use message "

    results = retriever.invoke(
        user_input)
    context = "\n\n".join(
        doc.page_content
        for doc in results)

    prompt = f"""
AI scam risk assessment assistant.
Analyze the following user input.
User Input:
{user_input}

ML Risk Score:
{risk_result["risk_score"]}%

Retrieved Evidence:
{context}

"""
    # LLM RESPONSE
    response = llm.invoke(prompt)
    return response.content


