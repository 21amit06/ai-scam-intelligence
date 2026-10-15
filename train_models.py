import re
from urllib.parse import urlparse
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

scamshield_train=pd.read_csv(r"C:\Users\kumar\OneDrive\Desktop\Ai Scam_intelligence\csam_shield_scam_detection\train.csv")
scamshield_val = pd.read_csv(
    r"C:\Users\kumar\OneDrive\Desktop\Ai Scam_intelligence\csam_shield_scam_detection\validation.csv"
)

scamshield_test = pd.read_csv(
    r"C:\Users\kumar\OneDrive\Desktop\Ai Scam_intelligence\csam_shield_scam_detection\test.csv"
)
#print("Train Shape",scamshield_train.shape)
#print(scamshield_train["label"].value_counts())


def clean_text(text):
    text=str(text)
    text=text.lower()
    text=re.sub(r"<.?>"," ",text)
    text=re.sub(r"\s+"," ",text)
    text=text.strip()
    return text

#CLEAN SCAMSHIELD TEXT
scamshield_train["clean_text"]=(scamshield_train["text"].apply(clean_text))
scamshield_val["clean_text"]=(scamshield_val["text"].apply(clean_text))
scamshield_test["clean_text"]=(scamshield_test["text"].apply(clean_text))

#PREPARE SCAMSHIELD DATA
scam_X_train_text=scamshield_train["clean_text"]
scam_y_train=scamshield_train["label"]

scam_X_val_text = scamshield_val["clean_text"]
scam_y_val = scamshield_val["label"]

scam_X_test_text = scamshield_test["clean_text"]
scam_y_test = scamshield_test["label"]
#print(scam_y_test)

#SCAMSHIELD VECTOR
from sklearn.feature_extraction.text import TfidfVectorizer
scam_vectorizer=TfidfVectorizer(
    max_features=1000,
    ngram_range=(1,2)
)
scam_X_train=scam_vectorizer.fit_transform(scam_X_train_text)
scam_X_val=scam_vectorizer.transform(scam_X_val_text)
scam_X_test=scam_vectorizer.transform(scam_X_test_text)
#print(scam_X_train.shape)

#SCAMSHIELD MODEL
scam_model=LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
scam_model.fit(scam_X_train,scam_y_train)

scam_y_val_pred=scam_model.predict(scam_X_val)
print("Accuracy:",
      accuracy_score(scam_y_val,scam_y_val_pred))
#print("\nClassification Report:")
#print(
#    classification_report(
#       scam_y_val,
#        scam_y_val_pred
#    )
#)

scam_y_test_pred=scam_model.predict(scam_X_test)
print(
    "Accuracy:",
    accuracy_score(
        scam_y_test,
        scam_y_test_pred
    ))

#  LOAD FAKE JOB DATASET
print("FAKE JOB DATASET")
fake_job = pd.read_csv(
    r"C:\Users\kumar\OneDrive\Desktop\Ai Scam_intelligence\fake_job\fake_job_postings.csv"
)
# PREPARE FAKE JOB TEXT
fake_job_text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in fake_job_text_columns:
    fake_job[column] = fake_job[column].fillna("")

fake_job["combined_text"] = (
    fake_job["title"] + " " +
    fake_job["company_profile"] + " " +
    fake_job["description"] + " " +
    fake_job["requirements"] + " " +
    fake_job["benefits"]
)

fake_job["clean_text"] = (
    fake_job["combined_text"].apply(clean_text)
)


#  SPLIT FAKE JOB DATA
fake_job_X = fake_job["clean_text"]
fake_job_y = fake_job["fraudulent"]


(fake_job_X_train,fake_job_X_temp,fake_job_y_train,fake_job_y_temp) = train_test_split(
    fake_job_X,
    fake_job_y,
    test_size=0.30,
    random_state=42,
    stratify=fake_job_y
)


(fake_job_X_val,fake_job_X_test,fake_job_y_val,fake_job_y_test) = train_test_split(
    fake_job_X_temp,
    fake_job_y_temp,
    test_size=0.50,
    random_state=42,
    stratify=fake_job_y_temp
)

# FAKE JOB TF-IDF
fake_job_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

fake_job_X_train_tfidf = (fake_job_vectorizer.fit_transform(fake_job_X_train))

fake_job_X_val_tfidf = (fake_job_vectorizer.transform(fake_job_X_val))

fake_job_X_test_tfidf = (fake_job_vectorizer.transform(fake_job_X_test))

# FAKE JOB MODEL

fake_job_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

fake_job_model.fit(fake_job_X_train_tfidf,fake_job_y_train)

# FAKE JOB VALIDATION
fake_job_y_val_pred = (fake_job_model.predict(fake_job_X_val_tfidf))

# 15. FAKE JOB FINAL TEST
fake_job_y_test_pred = (fake_job_model.predict(fake_job_X_test_tfidf))

print(
    "Accuracy:",
    accuracy_score(
        fake_job_y_test,
        fake_job_y_test_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        fake_job_y_test,
        fake_job_y_test_pred
    )
)

print("\nConfusion Matrix:")
# print(
#     confusion_matrix(
#         fake_job_y_test,
#         fake_job_y_test_pred
#     )
# )

# LOAD URL DATASET
url_df = pd.read_json(
    r"C:\Users\kumar\OneDrive\Desktop\Ai Scam_intelligence\url_scam\urls.json"
)

url_df = url_df.drop_duplicates()

url_df = url_df.reset_index(drop=True)

# URL FEATURE EXTRACTION
def extract_url_features(url):
    url = str(url)
    parse_url = url
    if not parse_url.startswith(
        ("http://", "https://")
    ):
        parse_url = "http://" + parse_url
    try:
        parsed = urlparse(parse_url)
        hostname = parsed.netloc
        path = parsed.path

    except ValueError:
        hostname = ""
        path = ""
    url_lower = url.lower()

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_hyphens": url.count("-"),
        "num_digits": sum(
            c.isdigit()
            for c in url
        ),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equal": url.count("="),
        "num_percent": url.count("%"),
        "https": int(
            url_lower.startswith("https")
        ),
        "has_ip": int(
            bool(
                re.search(
                    r"(\d{1,3}\.){3}\d{1,3}",
                    hostname
                )
            )
        ),

        "has_login": int("login" in url_lower),

        "has_verify": int("verify" in url_lower),

        "has_bank": int("bank" in url_lower),

        "has_account": int("account" in url_lower),

        "has_secure": int("secure" in url_lower),

        "has_update": int("update" in url_lower),

        "has_password": int("password" in url_lower)
    }

# CREATE URL FEATURES
url_features = (url_df["text"].apply(extract_url_features))

url_X_features = pd.DataFrame(url_features.tolist())

url_y = url_df["label"]
url_text = url_df["text"]

# 20. SPLIT URL DATA
# Use 200,000 URLs for URL model training
url_sample = url_df.sample(
    n=200000,
    random_state=42,

)
url_X_features = url_sample["text"]
url_y = url_sample["label"]

(url_text_train,url_text_temp,url_y_train,url_y_temp) = train_test_split(
    url_X_features,
    url_y,
    test_size=0.30,
    random_state=42,
    stratify=url_y
)

(url_text_val,url_text_test,url_y_val,url_y_test) = train_test_split(
    url_text_temp,
    url_y_temp,
    test_size=0.50,
    random_state=42,
    stratify=url_y_temp
)








# CHARACTER TF-IDF FOR URL
url_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=20000
)

url_X_train_tfidf = url_vectorizer.fit_transform(url_text_train)

url_X_val_tfidf = url_vectorizer.transform(url_text_val)

url_X_test_tfidf = url_vectorizer.transform(url_text_test)

# URL MODEL
url_model = LogisticRegression(
    max_iter=100,
    solver="saga"
)

url_model.fit(url_X_train_tfidf,url_y_train)

# 23. URL VALIDATION
url_y_val_pred = (url_model.predict(url_X_val_tfidf))

# 24. URL FINAL TEST
url_y_test_pred = (url_model.predict(url_X_test_tfidf))

print("\n========== URL FINAL TEST ==========")

print(
    "Accuracy:",
    accuracy_score(
        url_y_test,
        url_y_test_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        url_y_test,
        url_y_test_pred
    )
)

print("\nConfusion Matrix:")
# print(
#     confusion_matrix(
#         url_y_test,
#         url_y_test_pred
#     )
# )



import joblib
print("SAVING TRAINED MODELS")

joblib.dump(scam_vectorizer,"models/scam_vectorizer.pkl")

joblib.dump(scam_model,"models/scam_model.pkl")

joblib.dump(fake_job_vectorizer,"models/fake_job_vectorizer.pkl")

joblib.dump(fake_job_model,"models/fake_job_model.pkl")

joblib.dump(url_vectorizer,"models/url_vectorizer.pkl")

joblib.dump(url_model,"models/url_model.pkl")

print("All models saved successfully!")

