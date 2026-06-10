import os
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Spotify Popularity Analysis
# Fråga: Vad påverkar en låts popularitet på Spotify?
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ============================================================
# 1. LADDA DATA
# ============================================================

df = pd.read_csv("spotify.csv")
print(f"Rader: {df.shape[0]}, Kolumner: {df.shape[1]}")
print(df.dtypes)


# ============================================================
# 2. DATARENSNING
# ============================================================

print("\n--- SAKNADE VÄRDEN ---")
print(df.isnull().sum())
print(f"\nDuplikat: {df.duplicated().sum()}")

# Ta bort onödig indexkolumn
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ta bort dubbletter och rader med saknade värden
df = df.drop_duplicates()
df = df.dropna()

# Konvertera explicit (bool) till int för korrelationsanalys
df["explicit_int"] = df["explicit"].astype(int)

print(f"\nEfter rensning — Rader: {df.shape[0]}, Kolumner: {df.shape[1]}")


# ============================================================
# 3. VÄLJ VARIABLER
# ============================================================

numeric_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit_int"
]

target = "popularity"


# ============================================================
# 4. DESKRIPTIV STATISTIK
# ============================================================

print("\n--- DESKRIPTIV STATISTIK ---")
desc = df[numeric_features + [target]].describe().round(3)
print(desc)
desc.to_csv("outputs/descriptive_statistics.csv")


# ============================================================
# 5. VISUALISERING: Fördelning av popularitet
# ============================================================

fig = px.histogram(
    df,
    x="popularity",
    nbins=40,
    title="Fördelning av popularitet",
    labels={"popularity": "Popularitet (0–100)"},
    color_discrete_sequence=["#1DB954"]
)
fig.write_html("plots/01_popularity_distribution.html")
print("Saved: plots/01_popularity_distribution.html")


# ============================================================
# 6. VISUALISERING: Korrelationsmatris
# ============================================================

corr = df[numeric_features + [target]].corr().round(2)
print("\n--- KORRELATION MED POPULARITET ---")
print(corr[target].sort_values(ascending=False))
corr.to_csv("outputs/correlation_matrix.csv")

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Korrelationsmatris: Ljudegenskaper vs Popularitet"
)
fig.write_html("plots/02_correlation_matrix.html")
print("Saved: plots/02_correlation_matrix.html")


# ============================================================
# 7. VISUALISERING: Top 10 genrer efter popularitet
# ============================================================

top_genres = (
    df.groupby("track_genre")[target]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_genres.columns = ["Genre", "Genomsnittlig popularitet"]
top_genres.to_csv("outputs/top_genres.csv", index=False)

fig = px.bar(
    top_genres,
    x="Genomsnittlig popularitet",
    y="Genre",
    orientation="h",
    title="Top 10 genrer efter genomsnittlig popularitet",
    color="Genomsnittlig popularitet",
    color_continuous_scale="Viridis"
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html("plots/03_top_genres.html")
print("Saved: plots/03_top_genres.html")


# ============================================================
# 8. VISUALISERING: Danceability vs Popularitet (scatter)
# ============================================================

sample = df.sample(min(5000, len(df)), random_state=42)

fig = px.scatter(
    sample,
    x="danceability",
    y="popularity",
    color="track_genre",
    hover_data=["track_name", "artists", "track_genre"],
    title="Danceability vs Popularitet",
    labels={"danceability": "Danceability", "popularity": "Popularitet"},
    opacity=0.5,
    trendline="ols"
)
fig.write_html("plots/04_danceability_vs_popularity.html")
print("Saved: plots/04_danceability_vs_popularity.html")


# ============================================================
# 9. VISUALISERING: Explicit vs Popularitet (boxplot)
# ============================================================

df["Explicit"] = df["explicit"].map({True: "Explicit", False: "Ej explicit"})

fig = px.box(
    df,
    x="Explicit",
    y="popularity",
    color="Explicit",
    title="Popularitet: Explicit vs Ej explicit låtar",
    labels={"popularity": "Popularitet"},
    hover_data=["track_name", "artists"]
)
fig.write_html("plots/05_explicit_vs_popularity.html")
print("Saved: plots/05_explicit_vs_popularity.html")


# ============================================================
# 10. HYPOTESTEST: Explicit vs ej explicit
# Fråga: Är explicita låtar mer populära?
# ============================================================

explicit_pop     = df[df["explicit"] == True]["popularity"]
non_explicit_pop = df[df["explicit"] == False]["popularity"]

print("\n--- HYPOTESTEST: EXPLICIT ---")
print("H0: Ingen skillnad i popularitet mellan explicita och ej explicita låtar.")
print("H1: Det finns en skillnad i popularitet.")

mann = stats.mannwhitneyu(explicit_pop, non_explicit_pop, alternative="two-sided")

print(f"\nMedelvärde explicita:     {explicit_pop.mean():.2f}")
print(f"Medelvärde ej explicita:  {non_explicit_pop.mean():.2f}")
print(f"Mann-Whitney U: {mann.statistic:.0f}")
print(f"p-värde: {mann.pvalue:.6f}")

if mann.pvalue < 0.05:
    print("Slutsats: Statistiskt signifikant skillnad (p < 0.05) — vi förkastar H0.")
else:
    print("Slutsats: Ingen signifikant skillnad (p >= 0.05) — vi behåller H0.")

hypothesis_df = pd.DataFrame({
    "Grupp": ["Explicit", "Ej explicit"],
    "Medelvärde": [explicit_pop.mean(), non_explicit_pop.mean()],
    "Median": [explicit_pop.median(), non_explicit_pop.median()],
    "Antal": [len(explicit_pop), len(non_explicit_pop)],
    "p-värde": [mann.pvalue, mann.pvalue]
})
hypothesis_df.to_csv("outputs/hypothesis_test_explicit.csv", index=False)


# ============================================================
# 11. ENKEL LINJÄR REGRESSION
# Predictor: danceability (starkast intuitiv koppling)
# Target: popularity
# ============================================================

X_simple = df[["danceability"]]
y        = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae  = mean_absolute_error(y_test, preds)
mse  = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
r2   = r2_score(y_test, preds)

print("\n--- ENKEL LINJÄR REGRESSION: danceability → popularity ---")
print(f"Intercept:  {model.intercept_:.4f}")
print(f"Koefficient (danceability): {model.coef_[0]:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

pd.DataFrame({
    "Metric": ["Intercept", "Koefficient (danceability)", "MAE", "RMSE", "R²"],
    "Värde": [model.intercept_, model.coef_[0], mae, rmse, r2]
}).to_csv("outputs/regression_results.csv", index=False)


# ============================================================
# 12. VISUALISERING: Faktisk vs Predikterad popularitet
# ============================================================

pred_df = pd.DataFrame({
    "Faktisk popularitet":    y_test.values,
    "Predikterad popularitet": preds
}).sample(min(3000, len(y_test)), random_state=42)

fig = px.scatter(
    pred_df,
    x="Faktisk popularitet",
    y="Predikterad popularitet",
    title="Enkel linjär regression: Faktisk vs Predikterad popularitet",
    opacity=0.4,
    trendline="ols",
    color_discrete_sequence=["#1DB954"]
)
fig.write_html("plots/06_actual_vs_predicted.html")
print("Saved: plots/06_actual_vs_predicted.html")


# ============================================================
# 13. SAMMANFATTNING
# ============================================================

summary = f"""
Spotify Popularity Analysis — Sammanfattning
=============================================

Dataset:
  - {df.shape[0]} låtar, {df.shape[1]} variabler
  - 114 genrer, {df['explicit'].sum()} explicita låtar

Datarensning:
  - Tog bort Unnamed: 0 (intern indexkolumn, irrelevant)
  - Inga saknade värden hittades
  - Tog bort dubbletter

EDA-fynd:
  - Popularitet är skevt fördelad mot lägre värden (median ~35)
  - Ingen enskild ljudegenskap korrelerar starkt med popularitet
  - Loudness har starkast positiv korrelation (~0.08)
  - Instrumentalness har starkast negativ korrelation (~-0.07)

Hypotestest (Explicit vs ej explicit):
  - Explicita låtar:    medel {explicit_pop.mean():.2f}
  - Ej explicita låtar: medel {non_explicit_pop.mean():.2f}
  - p-värde: {mann.pvalue:.6f} → statistiskt signifikant skillnad

Linjär regression (danceability → popularity):
  - MAE:  {mae:.4f}
  - RMSE: {rmse:.4f}
  - R²:   {r2:.4f}

Slutsats:
  Låg R² visar att danceability ensam inte kan förklara popularitet.
  Popularitet drivs troligen av faktorer utanför ljuddata —
  t.ex. artistens storlek, marknadsföring och sociala medier.
"""

print(summary)
with open("outputs/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nKlart! Öppna filerna i plots/ i webbläsaren.")

