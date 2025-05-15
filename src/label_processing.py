from typing import Optional
import os
import pandas as pd
import tqdm
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
import socceraction.vaep.labels as lab
import socceraction.atomic.vaep.labels as atomiclab


def select_labels(
    format: str
) -> Optional[list[object]]:
    """
    Select pre-defined labels.

    ### Parameters:
    format: str
        "spadl" or "atomic-spadl".

    ### Returns:
    yfns: list[object]
        If successful, else None.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    if format == "spadl":
        yfns = [
            lab.scores, 
            lab.concedes, 
            lab.goal_from_shot
        ]
    elif format == "atomic-spadl":
        yfns = [
            atomiclab.scores, 
            atomiclab.concedes, 
            atomiclab.goal_from_shot
        ]

    return yfns


def generate_labels(
    games: pd.DataFrame,
    yfns: list[object],
    format: str,
    match_data_h5: str
) -> Optional[pd.DataFrame]:
    """
    Generates given labels for each game and action.

    ### Parameters:
    games: pd.DataFrame
        "train_games", "test_games", or "validation_games"; which indicates the games stored in h5-files.
    yfns: list[object]
        List of labels.
    format: str
        "spadl" or "atomic-spadl".
    match_data_h5: str
        "match_data_train_h5", "match_data_test_h5", "match_data_test_success_h5", or "match_data_test_fail_h5": Root path, which indicates the h5-file in which the match data in a specified format are saved.
    
    ### Returns:
    Y_dict: dict
        If successful, else None.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Generating labels")
    Y_dict = {}

    if format == "spadl":
        with pd.HDFStore(match_data_h5) as matchdatastore:
            for game in games_verbose:
                actions = matchdatastore[f"actions/game_{game.game_id}"]
                gamestates = spadl.add_names(actions)

                Y_dict[game.game_id] = pd.concat([fn(gamestates) for fn in yfns], axis=1)

    elif format == "atomic-spadl":
        with pd.HDFStore(match_data_h5) as matchdatastore:
            for game in games_verbose:
                actions = matchdatastore[f"atomic_actions/game_{game.game_id}"]
                gamestates = atomicspadl.add_names(actions)

                Y_dict[game.game_id] = pd.concat([fn(gamestates) for fn in yfns], axis=1)

    return Y_dict


def store_labels(
    Y_dict: dict,
    format: str,
    labels_h5: str
):
    """
    Stores generated labels for each game and action in a h5-file.

    ### Parameters:
    Y_dict: dict
        "Y_dict", which is a dict that contains the labels for each game and action.
    format: str
        "spadl" or "atomic-spadl".
    labels_h5: str
        "labels_train_h5", "labels_test_h5", "labels_test_success_h5" or "labels_test_fail_h5"; root path, which indicates the h5-file in which the labels are to be saved.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    games_verbose = tqdm.tqdm(list(Y_dict.keys()), desc=f"Storing labels in {os.path.basename(labels_h5)}")

    if format == "spadl" or format == "atomic-spadl":
        with pd.HDFStore(labels_h5) as labelstore:
            for game_id in games_verbose:
                labelstore.put(f"game_{game_id}", Y_dict[game_id], format='table')

    print(f"Labels ({format} format) successfully stored.")
