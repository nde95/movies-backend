import pandas as pd
import json

# companion script to make a json file with all movie titles for front end

df = pd.read_csv("movies_dataset.csv")


titles = df["title"].dropna().unique().tolist()


titles = sorted(titles)


with open("titles.json", "w", encoding="utf-8") as f:
    json.dump(titles, f, ensure_ascii=False, indent=2)

print("titles.json created")