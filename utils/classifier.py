"""
BERT-based Resume Classifier
Simulates transformer-based embeddings for 24 job categories.
Replace _encode() / predict() internals with a real HuggingFace BERT model
when a GPU environment is available.
"""

import numpy as np
from typing import List, Dict

JOB_CATEGORIES = [
    "Data Scientist", "Machine Learning Engineer", "Software Engineer",
    "Frontend Developer", "Backend Developer", "DevOps Engineer",
    "Cloud Architect", "Cybersecurity Analyst", "Data Analyst",
    "Business Analyst", "Product Manager", "Project Manager",
    "UX/UI Designer", "Graphic Designer", "Marketing Analyst",
    "HR Manager", "Financial Analyst", "Sales Manager",
    "Network Engineer", "Database Administrator", "QA Engineer",
    "Embedded Systems Engineer", "Blockchain Developer", "NLP Engineer",
]

ROLE_KEYWORDS: Dict[str, List[str]] = {
    "Data Scientist":            ["python","machine learning","statistics","pandas","numpy","sklearn","jupyter","r","deep learning","model","regression","classification","clustering","scipy"],
    "Machine Learning Engineer": ["pytorch","tensorflow","mlops","kubeflow","mlflow","model deployment","cuda","gpu","transformer","bert","huggingface","neural network","training","inference"],
    "Software Engineer":         ["java","c++","golang","algorithms","data structures","system design","microservices","api","rest","grpc","kafka","redis","oop"],
    "Frontend Developer":        ["react","vue","angular","javascript","typescript","css","html","webpack","tailwind","nextjs","redux","ui","responsive"],
    "Backend Developer":         ["node","django","flask","spring","postgresql","mongodb","api","rest","graphql","docker","kubernetes","sql","orm"],
    "DevOps Engineer":           ["ci/cd","jenkins","docker","kubernetes","terraform","ansible","aws","azure","gcp","pipeline","linux","bash","monitoring","prometheus"],
    "Cloud Architect":           ["aws","azure","gcp","cloud","architecture","vpc","s3","lambda","cloudformation","iam","cost optimization","serverless"],
    "Cybersecurity Analyst":     ["security","penetration testing","siem","firewall","ids","encryption","vulnerability","compliance","soc","threat","forensics","cve"],
    "Data Analyst":              ["sql","tableau","power bi","excel","reporting","dashboard","kpi","data visualization","etl","warehouse","looker","superset"],
    "Business Analyst":          ["requirements","stakeholder","process","bpmn","agile","scrum","gap analysis","use case","user story","jira","confluence"],
    "Product Manager":           ["roadmap","product strategy","okr","user research","a/b testing","go-to-market","sprint","backlog","stakeholder","metrics"],
    "Project Manager":           ["pmp","prince2","gantt","risk management","budget","schedule","agile","scrum","waterfall","deliverable","milestones"],
    "UX/UI Designer":            ["figma","sketch","adobe xd","wireframe","prototype","usability","user research","interaction design","accessibility","heuristic"],
    "Graphic Designer":          ["photoshop","illustrator","indesign","branding","typography","visual identity","print","adobe creative","logo","color theory"],
    "Marketing Analyst":         ["google analytics","seo","sem","digital marketing","campaign","ctr","conversion","social media","email marketing","funnel"],
    "HR Manager":                ["recruitment","onboarding","payroll","performance management","employee relations","hris","talent acquisition","compliance","benefits"],
    "Financial Analyst":         ["financial modeling","excel","valuation","dcf","forecasting","budget","p&l","gaap","bloomberg","cfa","variance analysis"],
    "Sales Manager":             ["crm","salesforce","lead generation","pipeline","quota","b2b","negotiation","revenue","account management","cold calling"],
    "Network Engineer":          ["cisco","ccna","bgp","ospf","routing","switching","vpn","sd-wan","network monitoring","wireshark","firewall","tcp/ip"],
    "Database Administrator":    ["oracle","mysql","postgresql","mongodb","query optimization","backup","replication","dba","rdbms","indexing","stored procedure"],
    "QA Engineer":               ["selenium","cypress","jmeter","test plan","test case","regression","automation","bug","defect","quality","postman","pytest"],
    "Embedded Systems Engineer": ["c","assembly","rtos","microcontroller","arduino","raspberry pi","firmware","hardware","iot","stm32","uart","i2c","spi"],
    "Blockchain Developer":      ["solidity","ethereum","smart contract","web3","defi","nft","truffle","hardhat","consensus","cryptography","metamask","polygon"],
    "NLP Engineer":              ["nlp","spacy","nltk","bert","gpt","text classification","ner","sentiment","transformer","language model","tokenizer","embeddings"],
}


def _keyword_score(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / max(len(keywords), 1)


class BERTResumeClassifier:
    """
    Lightweight keyword-based classifier that mimics a BERT pipeline.

    Production swap-in
    ------------------
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModel.from_pretrained("bert-base-uncased")

    def _encode(self, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:,0,:].squeeze().numpy()  # [CLS] token
    """

    def __init__(self):
        self.categories = JOB_CATEGORIES
        self.model_name = "bert-base-uncased"
        self.embed_dim  = 768

    def _encode(self, text: str) -> np.ndarray:
        """Simulated BERT [CLS] embedding (768-dim)."""
        np.random.seed(abs(hash(text[:50])) % (2**31))
        return np.random.randn(self.embed_dim).astype(np.float32)

    def predict(self, text: str, top_k: int = 3) -> List[Dict]:
        """Return top-k predictions with confidence scores."""
        scores = {}
        for role, kws in ROLE_KEYWORDS.items():
            base  = _keyword_score(text, kws)
            np.random.seed(abs(hash(text[:30] + role)) % (2**31))
            noise = np.random.uniform(0.0, 0.10)
            scores[role] = min(base + noise, 1.0)

        total = sum(scores.values()) or 1.0
        probs = {k: v / total for k, v in scores.items()}
        sorted_roles = sorted(probs, key=probs.get, reverse=True)
        return [{"label": r, "confidence": round(probs[r], 4)} for r in sorted_roles[:top_k]]

    def get_embedding(self, text: str) -> np.ndarray:
        return self._encode(text)
