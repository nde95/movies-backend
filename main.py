import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
from collections import Counter, defaultdict
from flask_cors import CORS

df = pd.read_csv("movies_dataset.csv")  # import movies from csv dataset

# print(df.head())

df = df[["id", "genres", "title", "vote_average", "keywords", "overview"]]  # drop fields that are unneeded for scoring


# collect and split json from csv file to strings
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return [i['name'] for i in items]
    except:
        return []


df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)


# turn array of values into string of keywords
df['genres'] = df['genres'].apply(lambda x: " ".join([i.replace(" ", "") for i in x]))
df['keywords'] = df['keywords'].apply(lambda x: " ".join([i.replace(" ", "") for i in x]))

# print(df['genres'])


df = df.fillna("")  # fill all empty columns
df['soup'] = df['genres'] + " " + df['keywords'] + " " + df['overview']  # create pool for similar keywords per movie


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])  # turn the keywords into a vector

model = NearestNeighbors(metric='cosine', algorithm='brute')  # nearest neighbor params for searching
model.fit(tfidf_matrix)


# function to return film name
def get_film_index(title):
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError(f"Movie not found: {title}")
    return matches.index[0]


def recommend(movie_titles, n=10):
    indices = [get_film_index(title) for title in movie_titles]

    # Build user vector
    user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))

    # Find nearest neighbors
    distances, neighbors = model.kneighbors(user_vector, n_neighbors=n + len(indices))

    results = []
    scores = []

    for i, idx in enumerate(neighbors[0]):
        if idx not in indices:
            results.append(df.iloc[idx]['title'])
            scores.append(1 - distances[0][i])

    # normalize the scoring to a more user-friendly number instead of a small decimal returned by the cosine
    max_score = max(scores) if scores else 1
    scaled_scores = [round((s / max_score) * 100, 1)  for s in scores]

    return results[:n], scaled_scores[:n]


# FIX THIS
all_genres = []
for row in df['genres']:
    all_genres.extend(row.split())

genre_counts = Counter(all_genres).most_common(10)

TOP_GENRES = {
    "labels": [g[0] for g in genre_counts],
    "values": [g[1] for g in genre_counts]
}

# FIX THIS
genre_ratings = defaultdict(list)

for _, row in df.iterrows():
    genres = row['genres'].split()
    rating = row['vote_average']

    for genre in genres:
        genre_ratings[genre].append(rating)

avg_ratings = {
    genre: sum(ratings)/len(ratings)
    for genre, ratings in genre_ratings.items()
}

sorted_genres = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)[:10]

GENRE_RATINGS = {
    "labels": [g[0] for g in sorted_genres],
    "values": [g[1] for g in sorted_genres]
}

RATINGS = df['vote_average'].tolist() # FIX THIS


app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "movies" not in data:
        return jsonify({"error": "Missing movie list"}), 400

    if len(data["movies"]) > 5:
        return jsonify({"error": "Max 5 movies allowed"}), 400

    try:
        # return recommended movies
        recs, sim_scores = recommend(data["movies"], 10)
        # keep track of user's selected movies indices
        selected_indices = [get_film_index(t) for t in data["movies"]]
        # keep track of recommended movies indices
        rec_indices = [get_film_index(t) for t in recs]

        # return users top genres
        selected_genres = []

        for idx in selected_indices:
            selected_genres.extend(df.iloc[idx]['genres'].split())

        genre_counts = Counter(selected_genres)

        # user-selected keywords
        selected_keywords = []

        for idx in rec_indices:
            selected_keywords.extend(df.iloc[idx]['keywords'].split())

        keyword_counts = Counter(selected_keywords).most_common(10)

        user_keywords = {
            "labels": [k[0] for k in keyword_counts],
            "values": [k[1] for k in keyword_counts]
        }

        # rating comparison for selected and recommended
        selected_ratings = [df.iloc[i]['vote_average'] for i in selected_indices]
        rec_ratings = [df.iloc[i]['vote_average'] for i in rec_indices]

        # ratings distribution
        ratings = df['vote_average'].tolist()

        return jsonify({
            "recommendations": recs,
            "userGenres": {
                "labels": list(genre_counts.keys()),
                "values": list(genre_counts.values())
            },
            "userKeywords": user_keywords,
            "similarityScores": {
                "labels": recs,
                "values": sim_scores
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


