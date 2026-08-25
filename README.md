Machine learning backend API for https://github.com/nde95/movies

# Movie Recommendation API

Backend API for a content-based movie recommendation system built with Python and Flask.

## Overview

This API processes a movie dataset and generates recommendations based on the similarity between movies. Movie genres, keywords, and descriptions are combined and converted into TF-IDF vectors, which are then compared using k-nearest neighbors.

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- k-Nearest Neighbors

## Features

- Movie title lookup
- Content-based movie recommendations
- Movie rating and genre statistics
- TF-IDF feature generation
- k-NN similarity search

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash 
python app.py
```
