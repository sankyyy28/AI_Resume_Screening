"""
Sample resumes and demo dataframe for the AI Resume Screening System.
"""

import pandas as pd
import numpy as np
from utils.classifier import JOB_CATEGORIES

# ── Sample resumes (one per category) ────────────────────────────────────────

SAMPLES = {
    "Data Scientist": """
John Doe | Data Scientist
Skills: Python, R, Machine Learning, Deep Learning, TensorFlow, PyTorch, Pandas,
NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter, Statistics, SQL, Spark.
Experience:
- Senior Data Scientist @ TechCorp (2021–Present): Built classification and regression
  models achieving 15% lift in customer retention. Led A/B experiments.
- Data Scientist @ StartupAI (2019–2021): Developed NLP pipeline for sentiment analysis
  using BERT; reduced manual review by 60%.
Education: M.Sc. Data Science, IIT Bombay (2019)
Projects: Fraud Detection System (XGBoost, 98% AUC), Customer Churn Predictor,
Image Classification CNN (ResNet50).
Certifications: AWS Certified ML Specialty, Google Professional Data Engineer.
""",

    "Machine Learning Engineer": """
Jane Smith | Machine Learning Engineer
Skills: PyTorch, TensorFlow, ONNX, CUDA, GPU optimisation, HuggingFace Transformers,
MLflow, Kubeflow, Docker, Kubernetes, Python, C++, BERT, GPT fine-tuning, MLOps.
Experience:
- ML Engineer @ Anthropic (2022–Present): Deployed transformer-based NLP models to
  production serving 10M+ requests/day. Built model-monitoring dashboards.
- ML Engineer @ OpenAI Research (2020–2022): Fine-tuned GPT models; reduced inference
  latency by 40% using ONNX + TensorRT.
Education: B.Tech Computer Science, BITS Pilani (2020)
Projects: Real-time Object Detection (YOLO v8), Resume Screening System (BERT, 92% acc.),
  Text Summarisation API.
""",

    "Frontend Developer": """
Alice Johnson | Frontend Developer
Skills: React, TypeScript, JavaScript, HTML5, CSS3, Tailwind CSS, Next.js, Redux,
Webpack, Vite, Jest, Cypress, Figma, REST APIs, GraphQL.
Experience:
- Senior Frontend Developer @ WebAgency (2021–Present): Architected React component
  library adopted across 5 products; improved Lighthouse score from 62 to 95.
- Frontend Developer @ E-Commerce Startup (2019–2021): Built responsive checkout flow;
  reduced cart abandonment by 22%.
Education: B.Sc. Computer Science, University of Pune (2019)
Projects: Design System (React + Storybook), Real-time Chat UI (WebSockets), PWA Blog.
""",

    "DevOps Engineer": """
Bob Kumar | DevOps Engineer
Skills: Docker, Kubernetes, Terraform, Ansible, Jenkins, GitHub Actions, AWS, Azure,
Prometheus, Grafana, ELK Stack, Linux, Bash, Python, CI/CD, Infrastructure as Code.
Experience:
- DevOps Lead @ CloudFirst (2020–Present): Migrated 40 microservices to Kubernetes;
  achieved 99.99% uptime. Reduced deployment time from 45 min to 8 min.
- DevOps Engineer @ FinTech Co. (2018–2020): Built CI/CD pipelines; automated
  infrastructure provisioning with Terraform across AWS regions.
Education: B.Tech IT, VIT (2018)
Certifications: CKA, AWS DevOps Professional, HashiCorp Terraform Associate.
""",

    "Cybersecurity Analyst": """
Carol Mehta | Cybersecurity Analyst
Skills: Penetration Testing, SIEM (Splunk, QRadar), Vulnerability Assessment, OWASP,
Metasploit, Burp Suite, IDS/IPS, Firewalls, Encryption, Compliance (ISO 27001, SOC 2),
Python, Bash, Threat Intelligence, Digital Forensics.
Experience:
- Security Analyst @ BankCorp (2021–Present): Conducted 30+ pen tests; reduced
  critical vulnerabilities by 75%. Led SOC incident response team.
- Security Engineer @ Gov Agency (2019–2021): Implemented SIEM alerting rules;
  detected APT intrusion within 2 hours.
Education: B.Sc. Information Security, Symbiosis (2019)
Certifications: CEH, OSCP, CompTIA Security+.
""",
}

ALL_SAMPLE_TEXTS = list(SAMPLES.values())


def get_sample_resume(category: str = None) -> str:
    """Return a sample resume. If category is None, return a random one."""
    if category and category in SAMPLES:
        return SAMPLES[category].strip()
    return np.random.choice(ALL_SAMPLE_TEXTS).strip()


def get_demo_dataframe(n: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic demo dataset of n resumes with labels.
    Cycles through available sample texts and augments with noise.
    """
    np.random.seed(42)
    rows = []
    cats = JOB_CATEGORIES[:len(SAMPLES)]  # use categories we have samples for

    for i in range(n):
        cat = cats[i % len(cats)]
        text = SAMPLES.get(cat, ALL_SAMPLE_TEXTS[0])
        # Lightly augment: shuffle sentences
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        np.random.shuffle(sentences)
        rows.append({"resume": ". ".join(sentences[:8]), "label": cat})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
