"""
Resume text preprocessor — cleaning, keyword extraction, section parsing.
"""

import re
import string
from typing import List

# Common tech / domain keywords to surface
TECH_KEYWORDS = [
    "python","java","c++","javascript","typescript","golang","rust","scala","r","matlab",
    "tensorflow","pytorch","keras","sklearn","xgboost","lightgbm","spacy","nltk","huggingface",
    "bert","gpt","transformer","llm","langchain",
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","cassandra","dynamodb",
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","ci/cd",
    "react","vue","angular","nextjs","nodejs","django","flask","fastapi","spring",
    "pandas","numpy","matplotlib","seaborn","plotly","tableau","power bi","excel",
    "git","github","gitlab","linux","bash","agile","scrum","jira","confluence",
    "machine learning","deep learning","nlp","computer vision","mlops","data science",
    "api","rest","graphql","microservices","kafka","rabbitmq","spark","hadoop","airflow",
]

STOP_WORDS = {
    "the","and","or","in","on","at","to","for","of","a","an","is","are","was","were",
    "be","been","have","has","had","do","does","did","will","would","could","should",
    "may","might","i","my","we","our","you","your","he","she","they","their","it","its",
    "with","from","by","as","that","this","these","those","but","if","so","than","then",
    "about","after","before","during","between","through","above","below","up","down",
    "out","off","over","under","again","further","not","no","nor","very","just","also",
}


class ResumePreprocessor:
    def clean(self, text: str) -> str:
        """Remove noise, normalise whitespace, lowercase."""
        text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
        text = re.sub(r"[^\w\s+#./]", " ", text)               # special chars (keep + # . /)
        text = re.sub(r"\s+", " ", text)                        # collapse whitespace
        return text.strip().lower()

    def extract_keywords(self, text: str) -> List[str]:
        """Return tech/domain keywords found in the resume."""
        text_lower = text.lower()
        found = [kw for kw in TECH_KEYWORDS if kw in text_lower]
        # Also surface any CamelCase words not already in list
        camel = re.findall(r"\b[A-Z][a-z]+[A-Z]\w*\b", text)
        return list(dict.fromkeys(found + camel))  # deduplicate, preserve order

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokeniser with stop-word removal."""
        tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    def extract_sections(self, text: str) -> dict:
        """
        Heuristically split resume into sections
        (Skills, Experience, Education, Projects, Summary).
        """
        section_headers = {
            "summary":    r"(summary|objective|profile)",
            "skills":     r"(skills|technical skills|competencies|technologies)",
            "experience": r"(experience|work history|employment)",
            "education":  r"(education|academic|qualification|degree)",
            "projects":   r"(projects|portfolio|work samples)",
            "certifications": r"(certification|certificate|credentials|awards)",
        }
        sections = {}
        lines = text.split("\n")
        current = "misc"
        for line in lines:
            for sec, pattern in section_headers.items():
                if re.search(pattern, line, re.IGNORECASE):
                    current = sec
                    break
            sections.setdefault(current, []).append(line)
        return {k: "\n".join(v).strip() for k, v in sections.items() if v}
