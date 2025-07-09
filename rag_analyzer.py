import json
import json5
import re
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from embedding_db import retrieve_with_context
from embedding_db_cfa import retrieve_cfa_chunks

llm = OllamaLLM(model="mistral")

news_template = """
You are a financial news assistant.
Given the following article, perform three tasks:
1. Provide a detailed summary.
2. Identify the sentiment (Positive/Neutral/Negative).
3. Determine political bias (Left/Center/Right).

Respond ONLY in JSON format like:
{{ "summary": "...", "sentiment": "...", "bias": "..." }}

Article:
{article}
"""
news_prompt = PromptTemplate(input_variables=["article"], template=news_template)
rag_chain = news_prompt | llm

# ✅ More exhaustive prompt for reports
report_template = """
You are a financial analyst with CFA expertise.
Analyze the following 5-year company report. Use CFA standards to explain:

1. Detailed profit trend: any fluctuations, margins, reasons.
2. Revenue trajectory: CAGR, declines, volatility.
3. Debt structure: short-term vs long-term, critical ratios.
4. Provide actionable investment advice (Yes/No), then explain rationale (risk-return, macro factors, industry fit, etc.)

Use the CFA reference strictly to support and validate arguments.

Respond ONLY in JSON like:
{{ "profit": "...", "revenue": "...", "debt": "...", "investment_advice": "..." }}

Company Report:
{company_report}

CFA Reference:
{cfa_knowledge}
"""
report_prompt = PromptTemplate(input_variables=["company_report", "cfa_knowledge"], template=report_template)
report_chain = report_prompt | llm

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    try:
        return json.loads(match.group())
    except:
        return json5.loads(match.group())

def analyze_article_rag(text: str) -> dict:
    try:
        response = rag_chain.invoke({"article": text})
        return extract_json(response)
    except Exception as e:
        print(f"[ERROR] News RAG failed: {e}")
        return {"summary": "", "sentiment": "Unknown", "bias": "Unknown"}

def analyze_financial_report_with_context(report_text: str) -> dict:
    try:
        cfa_chunks = retrieve_cfa_chunks(report_text, top_k=10)
        cfa_context = " ".join(cfa_chunks)

        report_trimmed = report_text[:3000]
        cfa_trimmed = cfa_context[:3000]

        response = report_chain.invoke({
            "company_report": report_trimmed,
            "cfa_knowledge": cfa_trimmed
        })

        return extract_json(response)

    except Exception as e:
        print(f"[ERROR] Report RAG with CFA failed: {e}")
        return {
            "profit": "Unknown",
            "revenue": "Unknown",
            "debt": "Unknown",
            "investment_advice": "Unable to determine"
        }
