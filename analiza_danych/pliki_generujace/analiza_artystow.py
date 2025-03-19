import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os

current_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(current_folder)
neighbor_folder = os.path.join(parent_folder, "wykresy")

parent2_folder = os.path.dirname(parent_folder)
data_folder = os.path.join(parent2_folder, "dane")
# Wczytanie pliku JSONL do DataFrame
df = pd.read_json(os.path.join(data_folder, "tracks.jsonl"), lines=True)

# Wyświetlenie pierwszych wierszy
df = df.drop(columns=["id"])
df.rename(columns={"name": "track_name"}, inplace=True)
df["year_of_release"] = df["release_date"].str[:4].astype(int)
moved = ["popularity", "id_artist", "release_date"]
pozostale_kolumny = [col for col in df.columns if col not in moved]
nowa_kolejnosc = moved + pozostale_kolumny
df = df[nowa_kolejnosc]


df_tracks = df.drop(columns=["release_date", "explicit", "key"])

# Wczytanie pliku JSONL do DataFrame
df_artists = pd.read_json(os.path.join(data_folder, "artists.jsonl"), lines=True)

# Wyświetlenie pierwszych wierszy
# df_artists = df_artists.drop(columns=["name"])
df_artists.rename(columns={"id": "id_artist"}, inplace=True)

df = pd.merge(df_tracks, df_artists, on="id_artist", how="right")
df = df.drop(columns=["id_artist"])
print(df.head())

# Lista głównych gatunków
genres_list = [
    "pop",
    "afro",
    "trap",
    "latin",
    "regg",
    "hip hop",
    "rap",
    "rock",
    "k-pop",
    "blues",
    "jazz",
    "r&b",
    "electro",
    "country",
    "alt",
]


df = df[df["popularity"].notna()]
df = df[df["genres"].notna()]
# Tworzenie kolumn binarnych na podstawie obecności podciągów
for genre in genres_list:
    df[genre] = df["genres"].apply(lambda x: 1 if any(genre in g for g in x) else 0)

# Obliczanie średniej popularności dla każdego gatunku
genre_popularity = {
    genre: df.loc[df[genre] == 1, "popularity"].mean() for genre in genres_list
}

# Usuwanie gatunków bez danych
genre_popularity = {k: v for k, v in genre_popularity.items() if not pd.isna(v)}
sorted_genre_popularity = dict(
    sorted(genre_popularity.items(), key=lambda x: x[1], reverse=True)
)

# Tworzenie wykresu z posortowanymi danymi
plt.figure(figsize=(10, 6))
plt.bar(
    sorted_genre_popularity.keys(), sorted_genre_popularity.values(), color="skyblue"
)
plt.xticks(rotation=45, ha="right")
plt.title("Średnia popularność w zależności od gatunku muzyki")
plt.xlabel("Gatunek")
plt.ylabel("Średnia popularność")
plt.tight_layout()
plt.savefig(os.path.join(neighbor_folder, "gatunek_popularność_srednia.png"))
plt.close()


# Po 2015 roku

df = df[df["year_of_release"] > 2020]
num_tracks_above_80 = len(df[df["popularity"] > 80])

# Obliczanie ogólnej liczby utworów
total_tracks = len(df)

# Obliczanie procentu utworów z popularnością większą niż 80
percentage_above_80 = (num_tracks_above_80 / total_tracks) * 100

# Wyświetlenie wyniku
print(
    f"Procent utworów z popularnością większą niż 80 po 2020: {percentage_above_80:.2f}%"
)

num_tracks_above_90 = len(df[df["popularity"] > 90])

# Obliczanie ogólnej liczby utworów
total_tracks = len(df)

# Obliczanie procentu utworów z popularnością większą niż 80
percentage_above_90 = (num_tracks_above_90 / total_tracks) * 100

# Wyświetlenie wyniku
print(
    f"Procent utworów z popularnością większą niż 90 po 2020: {percentage_above_90:.2f}%"
)

num_tracks_above_95 = len(df[df["popularity"] > 95])

# Obliczanie ogólnej liczby utworów
total_tracks = len(df)

# Obliczanie procentu utworów z popularnością większą niż 80
percentage_above_95 = (num_tracks_above_95 / total_tracks) * 100

# Wyświetlenie wyniku
print(
    f"Procent utworów z popularnością większą niż 95 po 2020: {percentage_above_95:.2f}%"
)

df = df[df["popularity"] > 90]


# genres_list = [
#     "pop",
#     "afro",
#     "trap",
#     "latin",
#     "regg",
#     "hip hop",
#     "rap",
#     "rock",
#     "k-pop",
#     "blues",
#     "jazz",
#     "r&b",
#     "electro",
#     "country",
#     "alt",
# ]


# df = df[df["popularity"].notna()]
# df = df[df["genres"].notna()]
# df = df[df["track_name"].notna()]
# # Tworzenie kolumn binarnych na podstawie obecności podciągów
# for genre in genres_list:
#     df[genre] = df["genres"].apply(lambda x: 1 if any(genre in g for g in x) else 0)

# # Obliczanie średniej popularności dla każdego gatunku
# genre_popularity = {
#     genre: df.loc[df[genre] == 1, "popularity"].mean() for genre in genres_list
# }

# # Usuwanie gatunków bez danych
# genre_popularity = {k: v for k, v in genre_popularity.items() if not pd.isna(v)}
# sorted_genre_popularity = dict(
#     sorted(genre_popularity.items(), key=lambda x: x[1], reverse=True)
# )

# # Tworzenie wykresu z posortowanymi danymi
# plt.figure(figsize=(10, 6))
# plt.bar(
#     sorted_genre_popularity.keys(), sorted_genre_popularity.values(), color="skyblue"
# )
# plt.xticks(rotation=45, ha="right")
# plt.title(
#     "Średnia popularność w zależności od gatunku muzyki dla najpopularniejszych utworów po 2015"
# )
# plt.xlabel("Gatunek")
# plt.ylabel("Średnia popularność")
# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "gatunek_popularność_srednia_po2015_90+.png"))
# plt.close()

# df_avg_popularity = df.groupby("name")["popularity"].mean().reset_index()
# df_sorted_avg_popularity = df_avg_popularity.sort_values(
#     by="popularity", ascending=False
# )

# top_20_artists = df_sorted_avg_popularity.head(20)

# # Wykres średniej popularności dla top 20 artystów
# plt.figure(figsize=(16, 8))
# plt.bar(
#     top_20_artists["name"],
#     top_20_artists["popularity"],
#     color="skyblue",
#     edgecolor="black",
# )


# # Dodanie tytułu i etykiet osi
# plt.title("Średnia popularność utworów dla najpopularniejszych artystów po 2015")
# plt.xlabel("Id Artysty")
# plt.ylabel("Średnia Popularność")
# plt.grid(True)
# plt.xticks(rotation=45, ha="right")
# # Wyświetlenie wykresu
# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "średnia_popularność_utworów_2015_90+.png"))
# plt.close()


# df = df[df["year_of_release"] > 2010]

# df = df.sort_values(by="popularity", ascending=False).head(20)
# df["nazwa_utworu_artysta"] = df["track_name"] + " - " + df["name"]


# # Wykres 20 najpopularniejszych utworów
# plt.figure(figsize=(16, 8))
# plt.bar(
#     df["nazwa_utworu_artysta"],
#     df["popularity"],
#     color="skyblue",
#     edgecolor="black",
# )


# # Dodanie tytułu i etykiet osi
# plt.title("20 najpopularniejszych utworów po 2010")
# plt.xlabel("Artysta")
# plt.ylabel("Populrność")
# plt.grid(True)
# plt.xticks(rotation=45, ha="right")
# # Wyświetlenie wykresu
# plt.tight_layout()
# plt.savefig(os.path.join(neighbor_folder, "20_najpopularniejszych_po2010.png"))
# plt.close()
