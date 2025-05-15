from typing import Optional
import os
import pandas as pd
import tqdm
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
import socceraction.vaep.features as fs
import socceraction.atomic.vaep.features as atomicfs


def select_features(
    format: str
) -> Optional[list[object]]:
    """
    Selects pre-defined features.

    ### Parameters:
    format: str
        "spadl" or "atomic-spadl".

    ### Returns:
    xfns: list[object]
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

    return xfns


def generate_features(
    games: pd.DataFrame,
    xfns: list[object],
    nb_prev_actions: int,
    format: str,
    match_data_h5: str 
) -> Optional[dict]:
    """
    Generates given features for each game and action.

    ### Parameters:
    games: pd.DataFrame
        "train_games", "test_games", or "validation_games"; which indicates the games stored in h5-files.
    xfns: list[object]
        List of features.
    nb_prev_actions: int
        "nb_prev_actions", number of previous actions included in features for a given action.
    format: str
        "spadl" or "atomic-spadl".
    match_data_h5: str
        "match_data_train_h5", "match_data_test_h5", "match_data_test_success_h5", or "match_data_test_fail_h5"; root path, which indicates the h5-file in which the match data in a specified format are saved.
    
    ### Returns:
    X_dict: dict
        If successful, else None.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Generating features")
    X_dict = {}

    if format == "spadl":
        with pd.HDFStore(match_data_h5) as matchdatastore:
            for game in games_verbose:
                actions = matchdatastore[f"actions/game_{game.game_id}"]
                gamestates = fs.gamestates(spadl.add_names(actions), nb_prev_actions)
                gamestates = fs.play_left_to_right(gamestates, game.home_team_id)

                X_dict[game.game_id] = pd.concat([fn(gamestates) for fn in xfns], axis=1)

    elif format == "atomic-spadl":
        with pd.HDFStore(match_data_h5) as matchdatastore:
            for game in games_verbose:
                actions = matchdatastore[f"atomic_actions/game_{game.game_id}"]
                gamestates = atomicfs.gamestates(atomicspadl.add_names(actions), nb_prev_actions)
                gamestates = atomicfs.play_left_to_right(gamestates, game.home_team_id)

                X_dict[game.game_id] = pd.concat([fn(gamestates) for fn in xfns], axis=1)

    return X_dict


def store_features(
    X_dict: dict,
    format: str,
    features_h5: str,
):
    """
    Stores generated features for each game and action in a h5-file.

    ### Parameters:
    X_dict: dict
        "X_dict", which is a dict that contains the features for each game and action.
    format: str
        "spadl" or "atomic-spadl".
    features_h5: str
        "features_train_h5", "features_test_h5", "features_test_success_h5" or "features_test_fail_h5"; root path, which indicates the h5-file in which the features are to be saved.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    games_verbose = tqdm.tqdm(list(X_dict.keys()), desc=f"Storing features in {os.path.basename(features_h5)}")

    if format == "spadl" or format == "atomic-spadl":
        with pd.HDFStore(features_h5) as featurestore:
            for game_id in games_verbose:
                featurestore.put(f"game_{game_id}", X_dict[game_id], format='table')

    print(f"Features ({format} format) successfully stored.")
    