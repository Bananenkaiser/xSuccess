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

1. **XGBoost-Modell**  
   - Klassifikator zur Vorhersage von Tor-/Gegentor-Wahrscheinlichkeiten  
   - Hyperparameter-Optimierung via Cross-Validation  

2. **interpret.ml**  
   - Einsatz von `interpret.glass-box` für globale Feature-Importances  
