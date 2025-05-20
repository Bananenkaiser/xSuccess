# xSuccess
Entwicklung des xSuccess-Modells zur Vorhersage der Erfolgswahrscheinlichkeit jeder Aktion im Fußball-Datensatz


## Beschreibung

Dieses Projekt lädt öffentliche StatsBomb-Spieldaten (Saison 2015/2016 der Top 5 Ligen), konvertiert sie ins SPADL-Format, erzeugt daraus leistungsstarke Features und Labels für das xSuccess-Modell, trainiert einen XGBoost-Klassifikator und nutzt `interpret.ml` zur Interpretierbarkeit. Abschließend werden für jeden Spieler Erfolgswahrscheinlichkeiten für jede Aktion berechnet.

## Features & Daten

- **Input-Features für das xSuccess-Modell**  
  - `start_x`, `start_y` — Startposition der Aktion  
  - `end_x`, `end_y` — Endposition der Aktion  
  - `time_seconds` — Spielzeitpunkt in Sekunden  
  - `type_name` — Aktionstyp (Pass, Schuss, Dribbling, …)  
  - `bodypart_name` — Körperteil (Fuß, Kopf, …)  
  - `action_distance` — zurückgelegte Distanz der Aktion  
  - `shot_angle_centered` — zentraler Schusswinkel relativ zur Torlinie  
  - `distance_to_goal` — Entfernung zum Tor  

- **Datenstruktur**  
  - SPADL-Daten (Teams, Spieler, Aktionen) gespeichert in HDF5  
  - Feature- und Label-Matrizen pro Spiel in HDF5  
  - Modell-Vorhersagen und Spieler-Scores in HDF5  

## Modelltraining & Interpretierbarkeit

### 1. Datenaufteilung

- train_test_split von scikit-learn mit train_size=50000, random_state=42 und shuffle=True zur Festlegung eines reproduzierbaren Trainingsdatensatzes.

### 2. Baseline: Dummy-Modell

- DummyClassifier(strategy='most_frequent') als einfaches Referenzmodell.

### 3. XGBoost-Klassifikator

- xgb.XGBClassifier(n_jobs=-1, enable_categorical=True, random_state=42)

- Training via xgb_model.fit(X_train, Y_train)

- Doppeltes Training (mit und ohne zusätzlichem player_id-Feature) zur Untersuchung des Einflusses von Spielerkennungen.

### 4. interpret.ml (Glassbox / EBM)

- ExplainableBoostingClassifier(n_jobs=-1, random_state=42, max_bins=32, interactions=5)

- Training via ebm.fit(X_train, y_train_array)

- Globale Feature-Importances und lokale Erklärungen (ähnlich SHAP) über interpret.show.

### 5. Evaluierung

- Gemeinsame Bewertungsfunktion evaluate_my_model(model, X_test, Y_test) zur Ausgabe folgender Metriken:

  - Brier Score – mittlere quadratische Fehlerabweichung der Wahrscheinlichkeiten.

  - Log Loss – Misst Güte der Wahrscheinlichkeitsprognosen.

  - ROC AUC – Fähigkeit, zwischen Klassen zu unterscheiden.

## Evaluierung der Modelle
**Dummy-Modell**
- strategy='most_frequent'
- Brier score: 0.17194
- log loss score: 6.19735
- ROC AUC: 0.50000


**XGBoost**
- Brier score: 0.06686
- log loss score: 0.21067
- ROC AUC: 0.94445


**interpret.ml (Glassbox)**
- train_size: 100000
- Brier score: 0.07086
- log loss score: 0.22259
- ROC AUC: 0.93857


