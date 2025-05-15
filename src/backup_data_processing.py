from typing import Optional, Tuple
import os
import pandas as pd
import tqdm
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.data.wyscout import WyscoutLoader, PublicWyscoutLoader
from socceraction.data.opta import OptaLoader
from sklearn.model_selection import train_test_split


def config_h5_file_paths(
    format: str,
    datafolder: str
) -> Optional[Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str]]:
    """
    Configurates the root path for each h5-file.

    ### Parameters:
    format: str
        "spadl" or "atomic-spadl".
    datafolder: str
        Root path, which indicates the folder in which the h5-files are to be saved. 

    ### Returns:
    match_data_h5, match_data_train_h5, match_data_test_h5, match_data_test_success_h5, match_data_test_fail_h5, features_train_h5, features_test_h5, features_test_success_h5, features_test_fail_h5, labels_train_h5, labels_test_h5, labels_test_success_h5, labels_test_fail_h5, predictions_test_h5, predictions_test_success_h5, predictions_test_fail_h5, vaep_h5_test, vaep_test_success_h5, vaep_test_fail_h5: str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str
        If successful, else None.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None
    
    # Create data folder if it doesn't exist
    if os.path.exists(datafolder):
        print(f"The folder '{datafolder}' already exists.")
    else:
        os.mkdir(datafolder)
        print(f"Directory {datafolder} created.")
    
    # Configure file and folder names
    if format == "spadl":
        match_data_h5 = os.path.join(datafolder, "match_data.h5")
        match_data_train_h5 = os.path.join(datafolder, "match_data_train.h5")
        match_data_test_h5 = os.path.join(datafolder, "match_data_test.h5")
        match_data_test_success_h5 = os.path.join(datafolder, "match_data_test_success.h5")
        match_data_test_fail_h5 = os.path.join(datafolder, "match_data_test_fail.h5")

        features_train_h5 = os.path.join(datafolder, "features_train.h5")
        features_test_h5 = os.path.join(datafolder, "features_test.h5")
        features_test_success_h5 = os.path.join(datafolder, "features_test_success.h5")
        features_test_fail_h5 = os.path.join(datafolder, "features_test_fail.h5")

        labels_train_h5 = os.path.join(datafolder, "labels_train.h5")
        labels_test_h5 = os.path.join(datafolder, "labels_test.h5")
        labels_test_success_h5 = os.path.join(datafolder, "labels_test_success.h5")
        labels_test_fail_h5 = os.path.join(datafolder, "labels_test_fail.h5")

        predictions_test_h5 = os.path.join(datafolder, "predictions_test.h5")
        predictions_test_success_h5 = os.path.join(datafolder, "predictions_test_success.h5")
        predictions_test_fail_h5 = os.path.join(datafolder, "predictions_test_fail.h5")

        vaep_test_h5 = os.path.join(datafolder, "vaep_test.h5")
        vaep_test_success_h5 = os.path.join(datafolder, "vaep_test_success.h5")
        vaep_test_fail_h5 = os.path.join(datafolder, "vaep_test_fail.h5")
        
    elif format == "atomic-spadl":
        match_data_h5 = os.path.join(datafolder, "atomic_match_data.h5")
        match_data_train_h5 = os.path.join(datafolder, "atomic_match_data_train.h5")
        match_data_test_h5 = os.path.join(datafolder, "atomic_match_data_test.h5")
        match_data_test_success_h5 = os.path.join(datafolder, "atomic_match_data_test_success.h5")
        match_data_test_fail_h5 = os.path.join(datafolder, "atomic_match_data_test_fail.h5")

        features_train_h5 = os.path.join(datafolder, "atomic_features_train.h5")
        features_test_h5 = os.path.join(datafolder, "atomic_features_test.h5")
        features_test_success_h5 = os.path.join(datafolder, "atomic_features_test_success.h5")
        features_test_fail_h5 = os.path.join(datafolder, "atomic_features_test_fail.h5")

        labels_train_h5 = os.path.join(datafolder, "atomic_labels_train.h5")
        labels_test_h5 = os.path.join(datafolder, "atomic_labels_test.h5")
        labels_test_success_h5 = os.path.join(datafolder, "atomic_labels_test_success.h5")
        labels_test_fail_h5 = os.path.join(datafolder, "atomic_labels_test_fail.h5")

        predictions_test_h5 = os.path.join(datafolder, "atomic_predictions_test.h5")
        predictions_test_success_h5 = os.path.join(datafolder, "atomic_predictions_test_success.h5")
        predictions_test_fail_h5 = os.path.join(datafolder, "atomic_predictions_test_fail.h5")

        vaep_test_h5 = os.path.join(datafolder, "atomic_vaep_test.h5")
        vaep_test_success_h5 = os.path.join(datafolder, "atomic_vaep_test_success.h5")
        vaep_test_fail_h5 = os.path.join(datafolder, "atomic_vaep_test_fail.h5")

    return match_data_h5, match_data_train_h5, match_data_test_h5, match_data_test_success_h5, match_data_test_fail_h5, features_train_h5, features_test_h5, features_test_success_h5, features_test_fail_h5, labels_train_h5, labels_test_h5, labels_test_success_h5, labels_test_fail_h5, predictions_test_h5, predictions_test_success_h5, predictions_test_fail_h5, vaep_test_h5, vaep_test_success_h5, vaep_test_fail_h5


def fetch_match_data(
    data_provider: str,
    source: str,
    seasons: list[str],
    competitions: list[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    filepath: Optional[str] = None
) -> Optional[Tuple[object, pd.DataFrame, pd.DataFrame]]:
    """
    Loads data based on the specified data provider and source.

    ### Parameters:
    data_provider: str
        Name of the data provider ("statsbomb", "wyscout", "opta").
    source: str
        "free", "subscription", or "local".
    seasons: list
        List of season names to load.
    competitions: list
        List of competition names to load.
    username: str, optional
        Username for subscription-based access.
    password: str, optional
        Password for subscription-based access.
    filepath: str, optional
        Root path for local stored data.

    ### Returns:
    loader, selected_competitions, games: object, pd.DataFrame, pd.DataFrame 
        If successful, else None.
    """
    if source not in {"free", "subscription", "local"}:
        print("Error: Invalid source. Please choose 'free', 'subscription', or 'local'.")
        return None

    # Configurate StatsBomb Loader
    if data_provider.lower() == "statsbomb":
        if source == "free":
            loader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})
        elif source == "subscription":
            if not username or not password:
                print("Error: Username and password are required for subscription-based access to StatsBomb.")
                return None
            loader = StatsBombLoader(getter="remote", creds={"user": username, "passwd": password})
        elif source == "local":
            if not filepath:
                print(f"Error: Filepath is required for local access to StatsBomb data. Please ensure the directory exists: {filepath}.")
                return None
            loader = StatsBombLoader(getter="local", root=filepath)

    # Configurate Wyscout Loader
    elif data_provider.lower() == "wyscout":
        if source == "subscription":
            if not username or not password:
                print("Error: Username and password are required for subscription-based access to Wyscout.")
                return None
            loader = WyscoutLoader(getter="remote", creds={"user": username, "passwd": password})
        elif source == "local":
            if not filepath:
                print(f"Error: Filepath is required for local access to Wyscout data. Please ensure the directory exists: {filepath}.")
                return None
            loader = WyscoutLoader(getter="local", root=filepath, feeds={
                "competitions": "competitions.json",
                "seasons": "seasons_{competition_id}.json",
                "games": "matches_{season_id}.json",
                "events": "matches/events_{game_id}.json",
            })
        elif source == "free":
            if not filepath:
                print(f"Error: Filepath is required for accessing public Wyscout data. Please ensure the directory exists: {filepath}.")
                return None
            loader = PublicWyscoutLoader(root=filepath)

    # Configurate Opta Loader 
    elif data_provider.lower() == "opta":
        if source != "local":
            print("Error: Opta data is only available from local sources. Please set 'source' to 'local'.")
            return None
        if not filepath:
            print(f"Error: Filepath is required for local access to Opta data. Please ensure the directory exists: {filepath}.")
            return None
        # Detect parser type based on data structure
        if "xml" in filepath.lower():
            loader = OptaLoader(root=filepath, parser="xml")
        elif "json" in filepath.lower():
            loader = OptaLoader(root=filepath, parser="json")
        elif "statsperform" in filepath.lower():
            loader = OptaLoader(root=filepath, parser="statsperform")
        elif "whoscored" in filepath.lower():
            loader = OptaLoader(root=filepath, parser="whoscored")
        else:
            print("Error: Please specify a valid Opta parser type: 'xml', 'json', 'statsperform', or 'whoscored'.")
            return None

    else:
        print(f"Error: The data provider '{data_provider}' is not supported. Please choose 'statsbomb', 'wyscout', or 'opta'.")
        return None

    # Load and validate data
    try:
        available_competitions = loader.competitions()

        if seasons is None or competitions is None:
            print("Error: Both 'seasons' and 'competitions' must be specified.")
            return None

        invalid_seasons = set(seasons) - set(available_competitions['season_name'])
        if invalid_seasons:
            print(
                f"Error: The following seasons are not available: {invalid_seasons}. "
                f"Available seasons: {sorted(set(available_competitions['season_name']))}"
            )
            return None

        invalid_competitions = set(competitions) - set(available_competitions['competition_name'])
        if invalid_competitions:
            print(
                f"Error: The following competitions are not available: {invalid_competitions}. "
                f"Available competitions: {sorted(set(available_competitions['competition_name']))}"
            )
            return None

        selected_competitions = available_competitions[
            (available_competitions['season_name'].isin(seasons)) &
            (available_competitions['competition_name'].isin(competitions))
        ]

        games = pd.concat([
            loader.games(row.competition_id, row.season_id)
            for row in selected_competitions.itertuples()
        ])

        return loader, selected_competitions, games

    except Exception as e:
        # If no data could be loaded, login data or filepath may be incorrect
        print(f"Error: Unable to load data. Please check your username, password, or filepath. Details: {e}")
        return None


def generate_match_data(
    games: pd.DataFrame,
    loader: object,
    format: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]]:
    """
    Generates data to a specified format.

    ### Parameters:
    games: pd.DataFrame
        "games", which indicates the games included in the selected competitions.
    loader: object
        "loader", which indicates the API client to fetch the data.
    format: str
        "spadl" or "atomic-spadl".
    
    ### Returns:
    If format == "spadl":
        teams, players, actions: pd.DataFrame, pd.DataFrame, dict
            If successful, else None.
    Elif format == "atomic-spadl":
        teams, players, atomic_actions: pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]
            If successful, else None.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None

    # 0) competitions generieren und auf die ausgewählten games filtern
    competitions = loader.competitions()
    # optional: nur die Wettbewerbe, die in `games` vorkommen
    comps = competitions[
        competitions["competition_id"].isin(games["competition_id"].unique()) &
        competitions["season_id"].     isin(games["season_id"].     unique())
    ].reset_index(drop=True)

    games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Converting match data")
    teams, players = [], []
    actions = {}
    atomic_actions = {}

    for game in games_verbose:
        teams.append(loader.teams(game.game_id))
        players.append(loader.players(game.game_id))
        events = loader.events(game.game_id)
        if format == "spadl":
            actions[game.game_id] = spadl.statsbomb.convert_to_actions(
                events, 
                home_team_id=game.home_team_id,
                xy_fidelity_version=1,
                shot_fidelity_version=1
                )
        elif format == "atomic-spadl":
            actions = spadl.statsbomb.convert_to_actions(
                events, 
                home_team_id=game.home_team_id,
                xy_fidelity_version=1,
                shot_fidelity_version=1
                )
            atomic_actions[game.game_id] = atomicspadl.convert_to_atomic(actions)

    teams = pd.concat(teams).drop_duplicates(subset="team_id")
    players = pd.concat(players)

    if format == "spadl":
        return comps, teams, players, actions
    
    elif format == "atomic-spadl":
        return comps,teams, players, atomic_actions


def split_games(
        games: pd.DataFrame,
        shared_games: Optional[str] = None,
        train_percentage: Optional[int] = None,
        validation_percentage: Optional[int] = None,
        random_state: Optional[int] = None,
        shuffle: Optional[bool] = None,
        stratify: Optional[str] = None # Das wurde hinzugefügt
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Splits games into training, testing, and optionally validation sets based on the given percentage.

    ### Parameters:
    games: pd.DataFrame
      "games", which indicates the games stored in match_data_h5.
    shared_games: str, optional
      "train_test" or "train_test_validation"; if specified, use all data for both training, testing, and optionally validation; all other parameters are ignored.
    train_percentage: int, optional
      The percentage of the data to use for training (0-100).
    validation_percentage: int, optional
      The percentage of the data to use for validation (0-100). Validation is only performed if specified.
    random_state: int, optional
      If specified, a consistent seed for reproducibility.
    shuffle: bool, optional
      Whether to shuffle the data before splitting.

    ### Returns:
    train_games, test_games, validation_games: pd.DataFrame, pd.DataFrame, pd.DataFrame if specified else None
    """
    if shared_games == "train_test":
        train_games = games
        test_games = games
        validation_games = None

    elif shared_games == "train_test_validation":
        train_games = games
        test_games = games
        validation_games = games

    else:
        if stratify:
            stratify_vals = games[stratify]
        else:
            stratify_vals = None

        train_games, test_games = train_test_split(
            games,
            train_size=train_percentage / 100,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_vals # Das wurde hinzugefügt
        )

        if validation_percentage:
            validation_size = validation_percentage / 100
            test_games, validation_games = train_test_split(
                test_games,
                test_size=validation_size / (1 - train_percentage / 100),
                random_state=random_state,
                shuffle=shuffle
            )
        else:
            validation_games = None

    return train_games, test_games, validation_games

def split_match_data(
    split_games: pd.DataFrame,
    player_games: pd.DataFrame,
    actions: dict[int, pd.DataFrame]
) -> Tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Splits match data based on games split (train_games, test_games, and validation_games).

    ### Parameters:
    split_games: pd.DataFrame
        "train_games", "test_games", or "validations_games".
    player_games: pd.DataFrame
        "players_games", which includes filtered data from "players".
    actions: dict[int, pd.DataFrame]
        "actions", which are the actions for each game in a specified format.

    ### Returns:
    split_players_games, split_actions: pd.DataFrame, dict[int, pd.DataFrame]
    """
    split_game_ids = set(split_games["game_id"])

    split_player_games = player_games[player_games["game_id"].isin(split_game_ids)]
    split_actions = {game_id: df for game_id, df in actions.items() if game_id in split_game_ids}

    return split_player_games, split_actions


def adjust_results(
    test_games: pd.DataFrame,
    actions: dict[int, pd.DataFrame]
) -> Tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """
    Adjust results of all actions to either successful (= 1) or fail (= 0).

    ### Parameters:
    test_games: pd.DataFrame
      "test_games", which indicates the data used for testing the model.
    actions: dict[int, pd.DataFrame]
      "actions", which are the actions for each game in a specified format.

    # Returns:
    actions_success, actions_fail: dict[int, pd.DataFrame], dict[int, pd.DataFrame]
    """
    actions_success = {
        key: df.assign(result_id=1) if key in set(test_games["game_id"]) else df
        for key, df in actions.items()
    }

    actions_fail = {
        key: df.assign(result_id=0) if key in set(test_games["game_id"]) else df
        for key, df in actions.items()
    }

    return actions_success, actions_fail


def store_match_data(
        competitions: pd.DataFrame,
        games: pd.DataFrame,
        teams: pd.DataFrame,
        players: pd.DataFrame,
        actions: dict[int, pd.DataFrame],
        format: str,
        match_data_h5: str,
        player_games: Optional[pd.DataFrame] = None
):
    """
    Stores data locally.

    ### Parameters:
    games: pd.DataFrame
        "games", "train_games", "test_games", or "validations_games"; which indicates the games included in the selected competitions.
    teams: pd.DataFrame
        "teams", which indicates the teams for each game.
    players: pd.DataFrame
        "players", which indicates the players for each game.
    actions: dict[int, pd.DataFrame]
        "actions", "actions_success" or "actions_fail"; which are the actions for each game in a specified format.
    format: str
        "spadl" or "atomic-spadl".
    match_data_h5: str
        "match_data_train_h5", "match_data_test_h5", "match_data_test_success_h5" or "match_data_test_fail_h5"; root path, which indicates the h5-file in which the match data in a specified format are to be saved.
    player_games: pd.DataFrame, optional
        If provided, it contains the already filtered `players` DataFrame.
    """
    if format not in {"spadl", "atomic-spadl"}:
        print("Error: Invalid format. Please choose 'spadl' or 'atomic-spadl'.")
        return None

    with pd.HDFStore(match_data_h5) as matchdatastore:
        matchdatastore["competitions"] = competitions        
        matchdatastore["games"] = games
        matchdatastore["teams"] = teams
        matchdatastore["players"] = players[['player_id', 'player_name', 'nickname']].drop_duplicates(
            subset='player_id')
        if player_games is not None:
            matchdatastore["player_games"] = player_games
        else:
            matchdatastore["player_games"] = players[
                ['player_id', 'game_id', 'team_id', 'is_starter', 'starting_position_id', 'starting_position_name',
                 'minutes_played']]

        for game_id in actions.keys():
            if format == "spadl":
                matchdatastore[f"actions/game_{game_id}"] = actions[game_id]

            elif format == "atomic-spadl":
                matchdatastore[f"atomic_actions/game_{game_id}"] = actions[game_id]

    print(f"Match data ({format} format) successfully stored.")


def load_match_data(
        format: str,
        match_data_h5: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Load match data from stored h5-file.

    ### Parameters:
    format: str
        "spadl" or "atomic-spadl".
    match_data_h5: str
        "match_data_train_h5", "match_data_test_h5", "match_data_test_success_h5", or "match_data_test_fail_h5"; root path, which indicates the h5-file in which the match data in a specified format are saved.

    ### Returns:
    games, teams, players, player_games, actions: pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]
    """
    actions = {}

    with pd.HDFStore(match_data_h5) as matchdatastore:
        competitions = matchdatastore["competitions"]
        games = matchdatastore["games"]
        teams = matchdatastore["teams"]
        players = matchdatastore["players"]
        player_games = matchdatastore["player_games"]

        games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Loading match data")
        for game in games_verbose:
            if format == "spadl":
                actions[game.game_id] = matchdatastore[f"actions/game_{game.game_id}"]

            elif format == "atomic-spadl":
                actions[game.game_id] = matchdatastore[f"atomic_actions/game_{game.game_id}"]

    return competitions, games, teams, players, player_games, actions
