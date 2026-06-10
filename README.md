Spotify Popularity Analysis
Projektfråga: Vad gör en låt populär på Spotify?

Beskrivning
Detta projekt analyserar ett Spotify-dataset med över 113 000 låtar för att undersöka vilka faktorer som påverkar en låts popularitet. Vi utför datarensning, explorativ dataanalys (EDA), hypotestestning och linjär regression.

Dataset

Källa: Kaggle — Spotify Tracks Dataset
Fil: spotif.csv (alternativt spotify.csv)
Storlek: 114 000 rader × 21 kolumner
Målvariabel: popularity (0–100)

Variabler
VariabelTypBeskrivningpopularityNumeriskPopularitetspoäng (0–100)danceabilityNumeriskHur dansvänlig låten är (0–1)energyNumeriskIntensitet och aktivitet (0–1)loudnessNumeriskGenomsnittlig volym i dBspeechinessNumeriskAndel tal i låten (0–1)acousticnessNumeriskHur akustisk låten är (0–1)instrumentalnessNumeriskAndel instrumentalt innehåll (0–1)livenessNumeriskSannolikhet att låten är liveinspelad (0–1)valenceNumeriskMusikalisk positivitet (0–1)tempoNumeriskBPMduration_msNumeriskLåtlängd i millisekunderexplicitKategoriskOm låten har explicit innehåll (True/False)track_genreKategoriskGenre (114 unika)track_nameTextLåtens namnartistsTextArtist(er)

Installation
Krav

Python 3.10+
pip

Installera beroenden
pip install -r requirements.txt
requirements.txt innehåller:
pandas
plotly
scipy
scikit-learn
statsmodels

Köra projektet

Se till att spotify.csv ligger i samma mapp som main.py
Kör:

(mac)python3 main.py
Output
Scriptet skapar två mappar:
plots/
├── 01_popularity_distribution.html
├── 02_correlation_matrix.html
├── 03_top_genres.html
├── 04_danceability_vs_popularity.html
├── 05_explicit_vs_popularity.html
└── 06_actual_vs_predicted.html

outputs/
├── descriptive_statistics.csv
├── correlation_matrix.csv
├── top_genres.csv
├── hypothesis_test_explicit.csv
├── regression_results.csv
└── summary.txt
Öppna .html-filerna direkt i webbläsaren för interaktiva diagram.

Metodik
1. Datarensning

Tog bort Unnamed: 0 (automatisk indexkolumn utan analytiskt värde)
Kontrollerade saknade värden — inga hittades
Identifierade och tog bort 451 duplicerade rader
Konverterade explicit (bool) till int för korrelationsanalys

2. EDA

Deskriptiv statistik (medelvärde, median, standardavvikelse)
Fördelning av popularitet — högersnedvriden, median ≈ 35
Korrelationsmatris — inga starka samband med popularitet (alla r < 0.1)
Top 10 genrer efter genomsnittlig popularitet

3. Hypotestest
Fråga: Är explicita låtar mer populära än ej explicita?

H₀: Ingen skillnad i popularitet
H₁: Det finns en skillnad
Test: Mann-Whitney U (data är ej normalfördelat — verifierat med Shapiro-Wilk)
Resultat: p < 0.001 → vi förkastar H₀. Explicita låtar är statistiskt signifikant mer populära (medel: 36.52 vs 33.02)

4. Linjär regression

Predictor: danceability
Target: popularity
Modell: popularity = 27.03 + 9.47 × danceability
MAE: 18.93 | RMSE: 22.38 | R²: 0.0014

R² på 0.0014 visar att danceability ensam inte kan förklara popularitet. Slutsatsen är att popularitet på Spotify i stor utsträckning drivs av faktorer utanför ljuddata — artiststorlek, marknadsföring, spellistor och sociala medier.

Resultat i korthet
FyndDetaljPopularitetsfördelningSkevt fördelad, majoritet av låtar har låg popularitetStarkaste korrelationLoudness (r = 0.08) — svag positivSvagaste korrelationInstrumentalness (r = −0.07) — svag negativBästa genrePop-film (medel ≈ 57.8)HypotestestExplicita låtar signifikant mer populära (p < 0.001)Regression R²0.0014 — danceability förklarar < 0.2 % av variansen

Projektstruktur
spotify-popularity-analysis/
├── main.py
├── spotif.csv
├── requirements.txt
├── README.md
├── plots/
│   └── (HTML-diagram)
└── outputs/
    └── (CSV-filer och sammanfattning)
