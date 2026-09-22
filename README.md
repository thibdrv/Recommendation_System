# 🎬 WildFlix : Movie Recommendation System & Data Analytics

Welcome to the WildFlix repository !

This project delivers an end-to-end movie recommendation engine combined with interactive data filtering and financial business analytics. It addresses classic streaming catalog challenges such as choice overload, quality filtering, and niche genre exposure.

### 🌐 Streamlit Application : <a href="https://wildflix-data-app.streamlit.app/" target="_blank" rel="noopener noreferrer">https://wildflix-data-app.streamlit.app/</a>

Login : demo / demo
______

### 📂 Project Structure :

Full_Film_Recommender.ipynb :

The master Jupyter Notebook containing the full data pipeline, Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, TF-IDF + Cosine Similarity recommendation logic, and Model Evaluation.

The other parts from 01 to 15 are just sections of this file.

### Modular Notebooks :

Individual, split versions of the master notebook covering specific steps (EDA, Preprocessing, Modeling) for improved readability and modular review.

app.py : The Streamlit web application interface for interactive recommendations.

______

## 📚 What This Project Taught Me

This project let me practice and deepen several skills, from data collection all the way to deployment :

### Data & API
- Working with a **real REST API** (TMDB) : handling endpoints, pagination, rate limiting, and structuring the retrieved data.
- **Data cleaning and preparation** (pandas) : handling missing values, deriving new columns (primary genre, IMDb-style weighted rating), filtering content (status, adult content).
- Building a **Power BI-style data model** (fact/dimension tables, aggregates by genre, director, year, country, language).

### Machine Learning / Recommendation
- Implementing several recommendation approaches: **content-based filtering** (TF-IDF + cosine similarity), a **hybrid** approach (content + popularity), and simpler score-based approaches (weighted rating, popularity).
- Gaining a concrete understanding of the trade-offs between these approaches (relevance vs. simplicity, cold start, etc.).
- Working with sparse matrices (`scipy.sparse`) to optimize similarity calculations on a sizeable dataset.

### Application Development
- Building a full interactive interface with **Streamlit** (multi-page navigation, dynamic filters, session state, a basic authentication system).
- Handling **internationalization** (multilingual FR/EN/ES interface).
- Custom visual styling via CSS injected into Streamlit.

### Deployment & Best Practices
- Deploying to **Streamlit Cloud** and solving real-world issues: relative file path handling, a misplaced or incomplete `requirements.txt`, missing dependencies.
- Loading large datasets directly from a **GitHub repo** (raw URLs) as an alternative to local storage.
- Becoming aware of **security considerations**: avoiding exposed API keys or plaintext passwords in source code, and understanding the difference between a demo authentication system and a real production system (password hashing, data persistence).
