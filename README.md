# Analyzing cyber attacks in WSN using ensemble techniques 

## 📌 Project Overview
The **A10 Forensic System** is an advanced security framework designed to detect and mitigate cyber attacks in **Wireless Sensor Networks (WSN)**. By leveraging multiple Machine Learning and Deep Learning algorithms, this system provides real-time traffic analysis, forensic logging, and actionable AI-driven insights to secure network nodes against complex threats like Blackhole, Grayhole, TDMA, and Flooding attacks.

## 🚀 Key Features
*   **Multi-Model Intelligence:** Employing 6 distinct powerful algorithms for robust detection.
*   **Forensic Dashboard:** Real-time visualization of network health, attack distribution, and node status.
*   **Hybrid AI Security Assistant:** Integrated **Google Gemini API** for cloud-based insights with **LLaMA** fallback for local development.
*   **Automated Forensics:** Logs every event with severity levels and target node identification for post-incident analysis.
*   **Deployment Ready:** Fully configured for cloud platforms like Render or Railway.

## 🧠 Model Comparison & Justification
Our system achieves **high accuracy (~100%)** across multiple models. Why do we maintain such a diverse suite of algorithms?
1.  **Comparative Analysis (Benchmarking):** Establish a benchmark for which algorithms are most efficient in resource-constrained WSN environments.
2.  **Defense in Depth:** Tree-based models learn strict rules, while neural networks capture complex non-linear relationships.
3.  **Efficiency vs. Complexity:** Fast models (Decision Trees) for edge nodes vs. robust models (XGBoost/Autoencoders) for cluster heads.

## 🛠️ Technology Stack
*   **Backend:** Python 3.x, Flask
*   **Machine Learning:** Scikit-Learn, XGBoost, Joblib
*   **AI Integration:** Google Gemini API (Cloud), Ollama (Local)
*   **Database:** SQLite
*   **Production Server:** Gunicorn
*   **Frontend:** HTML5, Modern CSS (Glassmorphism), Chart.js

## 📦 Deployment Guide
This system is optimized for deployment on **Render.com**.

### 1. Environment Set-up
Create a `.env` file with the following keys:
```env
MAIL_PASSWORD=your_google_app_password
GEMINI_API_KEY=your_google_gemini_api_key
```

### 2. Cloud Configuration (Render)
*   **Runtime:** Python 3
*   **Build Command:** `pip install -r requirements.txt`
*   **Start Command:** `gunicorn app:app`
*   **Environment Variables:** Add `MAIL_PASSWORD` and `GEMINI_API_KEY` in the dashboard.

## 👥 Developers
*   **Boya Lakshmi Moksha**
*   **Chinnapareddygari Dinesh Reddy**
*   **Penukonda Karthik**
*   **Ramugalla Akash**