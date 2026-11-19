"""Who built this app. Shown on request, never pasted into every prompt."""
import re

FOUNDER_NAME = "Sarder Junaid Ahmed"
FOUNDER_MD = """\
**Sarder Junaid Ahmed**: Data Scientist & Machine Learning Engineer.
B.Sc. in Computer Science & Engineering, RUET (Rajshahi University of Engineering & Technology), 2025.

Work on causal machine learning and trustworthy AI: fairness, uncertainty quantification and reproducible research.

**Other projects**
- [StatsPro](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app/): statistical analysis from a CSV to a report
- [ReproHub](https://reproapp-8jb7vbhnqyltxq23bsr8xn.streamlit.app/): checks published statistical claims against raw data

**Links:** [GitHub](https://github.com/Junaid-Ahmed-Rupok) · [LinkedIn](https://www.linkedin.com/in/sarder-junaid-ahmed-059b68240/) · [Portfolio](https://junaid-ahmed-rupok.github.io/__portfolio__Yes/)
"""

_WHO = re.compile(
    r"\b(who|whom)\b[^?.!]{0,40}\b(made|built|created|founded|developed|designed|owns|behind|programmed|wrote)\b"
    r"|\b(your|the)\s+(creator|founder|developer|author|builder|maker)\b"
    r"|\bwho\s+(is|are)\s+(your|the)\s+(creator|founder|developer|author)\b",
    re.I,
)


def is_founder_question(text: str) -> bool:
    """True for questions such as 'who built you?' or 'who is your developer?'."""
    t = text.strip()
    if len(t) > 140:
        return False
    return bool(_WHO.search(t)) and bool(re.search(r"\b(you|your|this (app|bot|chatbot|tool))\b", t, re.I))
