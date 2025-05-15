from typing import Optional, Tuple
import os
import pandas as pd
import tqdm
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
import socceraction.vaep.features as fs
import socceraction.atomic.vaep.features as atomicfs
import socceraction.vaep.formula as vaepformula
import socceraction.atomic.vaep.formula as atomicvaepformula
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from xgboost import XGBClassifier


def load_features_labels(
  split_games: pd.DataFrame,
  nb_prev_actions: int,
  format: str,
  features_h5: str,
  labels_h5: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
  """
  Load features and labels from stored h5-files.

  ### Parameters
  split_games: pd.DataFrame
    "train_games", "test_games", or "validations_games".
  nb_prev_actions: int
    "nb_prev_actions", number of previous actions included in features for a given action.
  format: str
    "spadl" or "atomic-spadl".
  features_h5: str
    "features_train_h5", "features_test_h5", "features_test_success_h5" or "features_test_fail_h5"; root path, which indicates the h5-file in which the features are saved.
  labels_h5: str
    "labels_train_h5", "labels_test_h5", "labels_test_success_h5" or "labels_test_fail_h5"; root path, which indicates the h5-file in which the labels are saved.

  ### Returns:
  X_split, Y_split: pd.DataFrame, pd.DataFrame
    If successful, else None.
  """
  if format not in {"spadl", "atomic-spadl"}:
    print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
    return None

  if format == "spadl":
    xfns = [
      fs.actiontype,
      fs.actiontype_onehot,
      fs.bodypart,
      fs.bodypart_onehot,
      fs.result,
      fs.result_onehot,
      fs.goalscore,
      fs.startlocation,
      fs.endlocation,
      fs.movement,
      fs.space_delta,
      fs.startpolar,
      fs.endpolar,
      fs.team,
      fs.time,
      fs.time_delta
      ]
    
    Xcols = fs.feature_column_names(xfns, nb_prev_actions)

  elif format == "atomic-spadl":
    xfns = [
      atomicfs.actiontype,
      atomicfs.actiontype_onehot,
      atomicfs.bodypart,
      atomicfs.bodypart_onehot,
      atomicfs.goalscore,
      atomicfs.location,
      atomicfs.polar,
      atomicfs.direction,
      atomicfs.team,
      atomicfs.time,
      atomicfs.time_delta
      ]
    
    Xcols = atomicfs.feature_column_names(xfns, nb_prev_actions)

  # Load features
  X_split = []
  for game_id in tqdm.tqdm(split_games.game_id, desc="Loading features"):
      Xi = pd.read_hdf(features_h5, f"game_{game_id}")
      X_split.append(Xi[Xcols])
  X_split = pd.concat(X_split).reset_index(drop=True)

  # 2. Load labels
  Ycols = ["scores","concedes"]
  Y_split = []
  for game_id in tqdm.tqdm(split_games.game_id, desc="Loading label"):
      Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
      Y_split.append(Yi[Ycols])
  Y_split = pd.concat(Y_split).reset_index(drop=True)

  return X_split, Y_split


def train_model(
  X_train: pd.DataFrame,
  Y_train: pd.DataFrame,
  n_estimators: int,
  max_depth: int,
  n_jobs: int,
  verbosity: int,
  enable_categorical: bool
) -> Tuple[pd.DataFrame, dict]:
  """
  Trains the model based on the specied dataset.

  ### Parameters:
  X_train: pd.DataFrame
    "X_train", which are the features for each game and action of the training data.
  Y_train: pd.DataFrame
    "Y_train", which are the labels for each game and action of the training data.
  n_estimators: int
    The number of trees to fit.
  max_depth: int
     Maximum depth of the tree.
  n_jobs: int
    The number of parallel threads to use.
  verbosity: int
    Verbosity of the output.
  enable_categorical: bool
    Whether to enable categorical feature support.

  ### Returns:
 models: dict
  """
  models = {}

  for col in list(Y_train.columns):
    model = XGBClassifier(
      n_estimators=n_estimators, 
      max_depth=max_depth, 
      n_jobs=-n_jobs, 
      verbosity=verbosity, 
      enable_categorical=enable_categorical
      )
    model.fit(X_train, Y_train[col])
    models[col] = model

  return models


def evaluate_model(
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
        models: dict
):
  """
  Evalute the trained model.

  ### Parameters:
  X_test: pd.DataFrame
    "X_test", which are the features for each game and action of the testing data.
  Y_test: pd.DataFrame
    "Y_test", which are the labels for each game and action of the testing data.
  models: dict
    Models.

  ### Returns:
  Y_hat: pd.DataFrame
  """
  Y_hat = pd.DataFrame()

  for col in Y_test.columns:
    Y_hat[col] = [p[1] for p in models[col].predict_proba(X_test)]
    y = Y_test[col]
    y_hat = Y_hat[col]

    unique_classes = set(y)
    if len(unique_classes) < 2:
      print(f"### Y: {col} ###")
      print(f"  Skipping log_loss & brier_score_loss for {col} (only one class: {unique_classes})")

    else:
      print(f"### Y: {col} ###")
      p = sum(y) / len(y)
      base = [p] * len(y)
      brier = brier_score_loss(y, y_hat)
      print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
      ll = log_loss(y, y_hat)
      print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
      print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

  return Y_hat


def store_predictions(
  test_games: pd.DataFrame,
  Y_hat: pd.DataFrame,
  format: str,
  match_data_h5: str,
  predictions_h5: str
):
  """
  Store predictions of the trained model.

  ### Parameters:
  test_games: pd.DataFrame
    "test_games", which indicates the data used for testing the model.
  Y_hat: pd.DataFrame
    Dataframe containing the predicted probabilities for the positive class for each target column.
  format: str
    "spadl" or "atomic-spadl".
  match_data_h5: str
    "match_data_test_h5", "match_data_test_success_h5", or "match_data_test_fail_h5"; root path, which indicates the h5-file in which the match data in a specified format are saved.
  predictions_h5: str
    "predictions_test_h5", "predictions_test_success_h5" or "predictions_test_fail_h5"; root path, which indicates the h5-file in which the predicitions are to be saved.
  """
  if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None

  A = []

  if format == "spadl":
    # Get rows with game id per action
    for game_id in tqdm.tqdm(test_games.game_id, "Loading game IDs"):
      Ai = pd.read_hdf(match_data_h5, f"actions/game_{game_id}")
      A.append(Ai[["game_id"]])
    A = pd.concat(A)
    A = A.reset_index(drop=True)

    # Concatenate action game id rows with predictions and save per game
    grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
    with pd.HDFStore(predictions_h5) as predictionstore:
      for k, df in tqdm.tqdm(grouped_predictions, desc=f"Storing predictions in {os.path.basename(predictions_h5)}"):
        df = df.reset_index(drop=True)
        predictionstore.put(f"game_{int(k)}", df[Y_hat.columns])

  elif format == "atomic-spadl":
    for game_id in tqdm.tqdm(test_games.game_id, "Loading game IDs"):
      Ai = pd.read_hdf(match_data_h5, f"atomic_actions/game_{game_id}")
      A.append(Ai[["game_id"]])
    A = pd.concat(A)
    A = A.reset_index(drop=True)

    # Concatenate action game id rows with predictions and save per game
    grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
    with pd.HDFStore(predictions_h5) as predictionstore:
      for k, df in tqdm.tqdm(grouped_predictions, desc=f"Storing predictions in {os.path.basename(predictions_h5)}"):
        df = df.reset_index(drop=True)
        predictionstore.put(f"game_{int(k)}", df[Y_hat.columns])

  print(f"Predictions ({format} format) successfully stored.") 


def compute_vaep(
  test_games: pd.DataFrame,
  teams: pd.DataFrame,
  players: pd.DataFrame,
  format: str,
  match_data_h5: str,
  predictions_h5: str
) -> Optional[pd.DataFrame]:
  """
  Compute VAEP values for each game and action.

  ### Parameters:
  test_data: pd.DataFrame
    "test_data", which indicates the data used for testing the model.
  teams: pd.DataFrame
    "teams", which indicates the teams for each game.
  players: pd.DataFrame   
    "players", which indicates the players for each game.
  format: str
    "spadl" or "atomic-spadl".
  match_data_h5: str
    "match_data_test_h5", "match_data_test_success_h5", or "match_data_test_fail_h5"; root path, which indicates the h5-file in which the match data in a specified format are saved.
  predictions_h5: str
    "predictions_test_h5", "predictions_test_success_h5" or "predictions_test_fail_h5"; root path, which indicates the h5-file in which the predictions are saved.

  ### Returns:
  vaep_values: pd.DataFrame
  """
  if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
  
  vaep_values = []

  if format == "spadl":
    for game in tqdm.tqdm(list(test_games.itertuples()), desc="Rating actions"):
      actions = pd.read_hdf(match_data_h5, f"actions/game_{game.game_id}")
      actions = (
        spadl.add_names(actions)
        .merge(players, how="left")
        .merge(teams, how="left")
        .sort_values(["game_id", "period_id", "action_id"])
        .reset_index(drop=True)
        )
      preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
      values = vaepformula.value(actions, preds.scores, preds.concedes)
      vaep_values.append(pd.concat([actions, preds, values], axis=1))
    vaep_values = pd.concat(vaep_values).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
  
  elif format == "atomic-spadl":
    for game in tqdm.tqdm(list(test_games.itertuples()), desc="Rating actions"):
      actions = pd.read_hdf(match_data_h5, f"atomic_actions/game_{game.game_id}")
      actions = (
        atomicspadl.add_names(actions)
        .merge(players, how="left")
        .merge(teams, how="left")
        .sort_values(["game_id", "period_id", "action_id"])
        .reset_index(drop=True)
        )
      preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
      values = atomicvaepformula.value(actions, preds.scores, preds.concedes)
      vaep_values.append(pd.concat([actions, preds, values], axis=1))
    vaep_values = pd.concat(vaep_values).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

  return vaep_values


def store_vaep(
  vaep_values: pd.DataFrame,
  format: str,
  vaep_h5: str
):
  """
  Store computed VAEP values for each game and action in a h5-file.

  ### Parameters:
  vaep_values: pd.DataFrame
    "vaep_values", "vaep_values_success", or "vaep_values_fail"; which contains the vaep values for each game and action.
  format: str
    "spadl" or "atomic-spadl".
  vaep_h5: str
    "vaep_test_h5", "vaep_test_success_h5" or "vaep_test_fail_h5"; root path, which indicates the h5-file in which the vaep values are to be saved.
  """
  if format not in {"spadl", "atomic-spadl"}:
    print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
    return None
  
  games_verbose = tqdm.tqdm(vaep_values.groupby("game_id"), desc=f"Storing VAEP values in {os.path.basename(vaep_h5)}")

  if format == "spadl" or format == "atomic-spadl":
    with pd.HDFStore(vaep_h5) as vaepstore:
      for game_id, game_vaep in games_verbose:
        vaepstore.put(f"game_{game_id}", game_vaep, format='table')

  print(f"VAEP values ({format} format) successfully stored.")


def load_vaep(
  vaep_h5: str
) -> pd.DataFrame:
  """
  Load VAEP values from stored h5-file.

  ### Parameters:
  vaep_h5: str
    "vaep_test_h5", "vaep_test_success_h5" or "vaep_test_fail_h5"; root path, which indicates the h5-file in which the vaep values are to be saved.

  ### Returns:
  vaep_values: pd.DataFrame
  """
  with pd.HDFStore(vaep_h5) as vaepstore:
    vaepstore_keys = tqdm.tqdm(vaepstore.keys(), desc=f"Loading {os.path.basename(vaep_h5)}")

    vaep_values = pd.concat([vaepstore[key] for key in vaepstore_keys], ignore_index=True)

  return vaep_values


def compare_vaep(
  vaep_values: pd.DataFrame,
  vaep_values_success: pd.DataFrame,
  vaep_values_fail: pd.DataFrame,
  comparison: str
) -> pd.DataFrame:
  """
  Compare output of regular and adjusted VAEP values.

  ### Parameters:
  vaep_values: pd.DataFrame
    "vaep_values", which contains the vaep values for each game and action.
  vaep_values_success: pd.DataFrame
    "vaep_values_success", which contains the vaep values for each game and action.
  vaep_values_fail: pd.DataFrame
    "vaep_values_fail", which contains the vaep values for each game and action.
  comparison: str
    "games" or "actions".

  ### Returns:
  vaep_comparison: pd.DataFrame
  """
  if comparison == "games":
    vaep_comparison = None

    for name, df in {"vaep": vaep_values, "vaep_success": vaep_values_success, "vaep_fail": vaep_values_fail}.items():
      df_grouped = df.groupby(['game_id', 'team_name'])['vaep_value'].sum().reset_index()

      df_merged = df_grouped.groupby('game_id').apply(
        lambda x: pd.Series({
          'team_name_1': x.iloc[0]['team_name'],
          'team_name_2': x.iloc[1]['team_name'],
          f'{name}_1': x.iloc[0]['vaep_value'],
          f'{name}_2': x.iloc[1]['vaep_value']
        }),
        include_groups=False
      ).reset_index()

      if vaep_comparison is None:
        vaep_comparison = df_merged
      else:
        vaep_comparison = vaep_comparison.merge(df_merged.drop(columns=["team_name_1", "team_name_2"]),
                                                            on="game_id")

    vaep_comparison.iloc[:, 3:] = vaep_comparison.iloc[:, 3:].round(2)

  elif comparison == "actions":
    vaep_comparison = pd.DataFrame({
      "action": vaep_values["type_name"].head(200),
      "vaep": vaep_values["vaep_value"].head(200),
      "vaep_success": vaep_values_success["vaep_value"].head(200),
      "vaep_fail": vaep_values_fail["vaep_value"].head(200),
    })

  elif comparison == "action_types":
    vaep_comparison = pd.DataFrame({
      "vaep": vaep_values.groupby("type_name").vaep_value.mean(),
      "vaep_success": vaep_values_success.groupby("type_name").vaep_value.mean(),
      "vaep_fail": vaep_values_fail.groupby("type_name").vaep_value.mean()})

  return vaep_comparison


def compare_vaep_on_actions(
    vaep_values: pd.DataFrame,
    vaep_values_success: pd.DataFrame,
    vaep_values_fail: pd.DataFrame
) -> pd.DataFrame:
    """
    Vergleicht die VAEP-Werte für jede einzelne Aktion.
    
    Der zurückgegebene DataFrame enthält:
      - game_id: Identifikation des Spiels
      - player: Name (oder ID) des Spielers, der die Aktion durchgeführt hat
      - action: Der Aktionstyp (z. B. "pass", "shot", etc.)
      - successful: Boolean, ob die Aktion erfolgreich war (basierend auf result_name, z. B. "success")
      - vaep: Der ursprüngliche VAEP-Wert der Aktion
      - vaep_success: Der angepasste (z. B. offensive) VAEP-Wert der Aktion
      - vaep_fail: Der angepasste (z. B. defensive) VAEP-Wert der Aktion

    Es wird vorausgesetzt, dass die drei DataFrames zeilenweise übereinstimmen (d. h. jede Zeile entspricht exakt derselben Aktion).
    """
    # Annahme: In vaep_values gibt es die Spalten "game_id", "player_name", "type_name" und "result_name".
    # Falls du statt "player_name" beispielsweise "player_id" nutzen möchtest, passe dies einfach an.
    compare_df = pd.DataFrame({
        "game_id": vaep_values["game_id"],
        "original_event_id": vaep_values["original_event_id"],
        "action_id": vaep_values["action_id"],
        "player_id": vaep_values["player_id"],
        "action": vaep_values["type_name"],
        "successful": vaep_values["result_name"].apply(lambda x: x == "success") if "result_name" in vaep_values.columns else None,
        "vaep_offensiv": vaep_values["offensive_value"],
        "vaep_defensiv": vaep_values["defensive_value"],
        "vaep": vaep_values["vaep_value"],
        "vaep_success": vaep_values_success["vaep_value"],
        "vaep_fail": vaep_values_fail["vaep_value"]
    })
    
    return compare_df