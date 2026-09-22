## 🎬 Streamlit Application

This folder contains the user interface of the project, built with **Streamlit**. It provides an interactive web interface for the movie recommendation system.

### Features

- **Personalized recommendations** : enter a movie you like and get similar suggestions, based on several algorithms (TF-IDF + Cosine Similarity, Hybrid, Highest Rated, Most Popular).
- **Advanced filters** : genre, release year, runtime, minimum rating.
- **Catalog** : browse and search through the available movies.
- **Statistics** : interactive visualizations of the catalog data (genre distribution, rating distribution, trends by year).
- **Authentication** : login / sign-up system with favorites management.
- **Multilingual** : interface available in French, English, and Spanish.

### Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Deployment

The app is deployed on **Streamlit Cloud**. The `requirements.txt` file must be placed in the same folder as `app.py` for dependencies to install correctly.

### ⚠️ Important note

The included authentication system is a **demo**, not a production-ready system :
- The credentials (`admin` / `demo`) are provided as examples to test the app.
- Accounts created via "Sign Up" are **not persistent** : they're stored in memory (`session_state`) and reset every time the server restarts.
- Passwords are not hashed this should not be used as-is in a production context with real user data.

### Data structure

The app loads its data directly from CSV files hosted on GitHub (raw URLs), rather than from local files. This includes information on movies, genres, directors, and aggregated statistics, originally generated from the TMDB API.
