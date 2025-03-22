"""
This file contains the main data structures, graph algorithms, tree algorithms,
and the core logic functions.
"""

import os
import ast
import math
import pandas as pd
import networkx as nx
from anytree import Node
import random


def load_steam_games(file_path: str) -> pd.DataFrame:
    """
    Always load all lines from steam_games.json (no user prompt).
    Show a loading progress anyway for better user feedback.
    Also fix potential float app_name by converting it to string afterwards.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    # Count total lines
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as fcount:
        for _ in fcount:
            total_lines += 1

    print(f"Loading steam_games.json with NO limit, total lines = {total_lines} ...")

    data = []
    line_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            # Show loading progress (every ~5% if possible)
            if total_lines > 0:
                progress_ratio = line_count / total_lines
                if line_count % max(1, math.floor(total_lines / 20)) == 0:
                    percentage = int(progress_ratio * 100)
                    print(f"Loading steam_games: {percentage}% ({line_count}/{total_lines} lines)")

            obj = ast.literal_eval(line)
            data.append(obj)

    df = pd.DataFrame(data)

    # Convert app_name to string to avoid 'float' object no attribute 'lower'
    if 'app_name' in df.columns:
        df['app_name'] = df['app_name'].astype(str).fillna("")

    return df


def load_user_items(user_items_file: str) -> pd.DataFrame:
    """
    Read user-game records with a user-chosen line limit, after warning that
    more lines => more accuracy, slower building. Show loading progress as well.
    """
    if not os.path.exists(user_items_file):
        print(f"File not found: {user_items_file}")
        return pd.DataFrame()

    print("\n[Warning] The more lines you load, the more accurate, but the slower the building of the graph/tree.")
    user_input = input("Enter the maximum number of lines to load for 'user_items' (0 or blank means all): ").strip()
    if not user_input:
        user_input = "0"
    try:
        max_lines = int(user_input)
    except ValueError:
        max_lines = 0

    # Count total lines
    total_lines = 0
    with open(user_items_file, 'r', encoding='utf-8') as fcount:
        for _ in fcount:
            total_lines += 1

    if max_lines == 0 or max_lines > total_lines:
        max_lines = total_lines

    print(f"Loading {user_items_file} with limit = {max_lines} lines (out of {total_lines} total)...")

    rows = []
    line_count = 0

    with open(user_items_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line_count >= max_lines:
                break
            line_count += 1

            line = line.strip()
            if not line:
                continue

            # Show progress in increments (about every 5%)
            if max_lines > 0:
                progress_ratio = line_count / max_lines
                if line_count % max(1, math.floor(max_lines / 20)) == 0:
                    percentage = int(progress_ratio * 100)
                    print(f"Loading user_items: {percentage}% ({line_count}/{max_lines} lines)")

            obj = ast.literal_eval(line)
            user_id = obj['user_id']
            steam_id = obj['steam_id']
            items_list = obj.get('items', [])
            for item in items_list:
                row = {
                    'user_id': user_id,
                    'steam_id': steam_id,
                    'item_id': item.get('item_id'),
                    'item_name': item.get('item_name'),
                    'playtime_forever': item.get('playtime_forever'),
                    'playtime_2weeks': item.get('playtime_2weeks')
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_user_reviews(user_reviews_file: str) -> pd.DataFrame:
    """
    Read user review data with a user-chosen line limit, after warning that
    more lines => more accuracy, slower building. Show loading progress.
    """
    if not os.path.exists(user_reviews_file):
        print(f"File not found: {user_reviews_file}")
        return pd.DataFrame()

    print("\n[Warning] The more lines you load, the more accurate, but the slower the building of the graph/tree.")
    user_input = input("Enter the maximum number of lines to load for 'user_reviews' (0 or blank means all): ").strip()
    if not user_input:
        user_input = "0"
    try:
        max_lines = int(user_input)
    except ValueError:
        max_lines = 0

    # Count total lines
    total_lines = 0
    with open(user_reviews_file, 'r', encoding='utf-8') as fcount:
        for _ in fcount:
            total_lines += 1

    if max_lines == 0 or max_lines > total_lines:
        max_lines = total_lines

    print(f"Loading {user_reviews_file} with limit = {max_lines} lines (out of {total_lines} total)...")

    rows = []
    line_count = 0

    with open(user_reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line_count >= max_lines:
                break
            line_count += 1
            line = line.strip()
            if not line:
                continue

            # Show progress in increments (about every 5%)
            if max_lines > 0:
                progress_ratio = line_count / max_lines
                if line_count % max(1, math.floor(max_lines / 20)) == 0:
                    percentage = int(progress_ratio * 100)
                    print(f"Loading user_reviews: {percentage}% ({line_count}/{max_lines} lines)")

            obj = ast.literal_eval(line)
            user_id = obj.get('user_id')
            reviews_list = obj.get('reviews', [])
            for review in reviews_list:
                row = {
                    'user_id': user_id,
                    'item_id': review.get('item_id'),
                    'recommend': review.get('recommend'),
                    'review_text': review.get('review'),
                    'posted': review.get('posted'),
                    'helpful': review.get('helpful'),
                    'funny': review.get('funny')
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_game_graph(df_user_items: pd.DataFrame) -> nx.Graph:
    """
    Build a graph based on user-game relationships:
    - Nodes: games (item_id)
    - Edges: if two games were played by the same user, add an edge and increment weight

    We also display a progress bar while building.
    """
    G = nx.Graph()

    # Group user->list_of_games
    grouped = df_user_items.groupby('user_id')['item_id'].apply(list)
    users = grouped.index.tolist()
    total_users = len(users)
    user_count = 0

    print("Building the game graph...")

    for user_id, item_list in grouped.items():
        user_count += 1
        # Show progress for building the graph every ~5%
        if total_users > 0 and user_count % max(1, math.floor(total_users / 20)) == 0:
            percentage = int((user_count / total_users) * 100)
            print(f"Building graph: {percentage}% ({user_count}/{total_users} users processed)")

        unique_items = set(item_list)
        unique_items = list(unique_items)

        if len(unique_items) < 2:
            continue

        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                gameA = unique_items[i]
                gameB = unique_items[j]

                # If node doesn't exist, add it
                if not G.has_node(gameA):
                    G.add_node(gameA)
                if not G.has_node(gameB):
                    G.add_node(gameB)

                if not G.has_edge(gameA, gameB):
                    G.add_edge(gameA, gameB, weight=1)
                else:
                    G[gameA][gameB]['weight'] += 1

    print("Game graph construction complete.")
    return G


def build_genre_tree(df_games: pd.DataFrame) -> Node:
    """
    Build a tree based on each game's 'genres' field:
    - Root ("All Games")
    - Each unique genre is a child of Root
    - Games are children of their respective genre node

    We'll also display a progress bar based on total rows in df_games.
    """
    root = Node("All Games")
    genre_dict = {}

    total_games = len(df_games)
    processed_games = 0

    print("Building the genre tree...")

    for idx, row in df_games.iterrows():
        processed_games += 1
        if total_games > 0 and processed_games % max(1, math.floor(total_games / 20)) == 0:
            percentage = int((processed_games / total_games) * 100)
            print(f"Building tree: {percentage}% ({processed_games}/{total_games} games processed)")

        game_id = row['id']
        game_name = row.get('app_name', f"Game_{game_id}")

        # Convert to string to avoid float-lower errors
        if not isinstance(game_name, str):
            game_name = str(game_name)

        genres = row.get('genres', [])
        if not isinstance(genres, list):
            genres = []

        for g in genres:
            if g not in genre_dict:
                genre_dict[g] = Node(g, parent=root)
            Node(game_name, parent=genre_dict[g])

    print("Genre tree construction complete.")
    return root


def recommend_by_graph(
    game_graph: nx.Graph,
    target_game_id: str,
    top_n: int = 5,
    id_to_name: dict = None,
    id_to_genres: dict = None
) -> list:
    """
    Using graph-based search or neighbor analysis, find games most similar to the target (sorted by edge weight).
    If id_to_name is provided, convert item IDs to names and skip unknown items.
    """
    if target_game_id not in game_graph:
        return []

    neighbors = game_graph[target_game_id]
    # Sort neighbors by descending weight
    sorted_neighbors = sorted(
        neighbors.items(),
        key=lambda x: x[1]["weight"],
        reverse=True
    )
    recommended_ids = [nb[0] for nb in sorted_neighbors[:top_n]]

    # If we have a dictionary for ID->name, filter out unknowns
    # so that we only keep IDs that do map to a known name.
    if id_to_name is not None:
        final_names = []
        for gid in recommended_ids:
            # We skip IDs that have no name in the dict
            if gid in id_to_name:
                final_names.append(id_to_name[gid])
        return final_names
    else:
        return recommended_ids


def hybrid_recommendation(
    game_graph: nx.Graph,
    genre_tree_root: Node,
    target_game_id: str,
    target_genre: str,
    id_to_name: dict = None,
    id_to_genres: dict = None,
    name_to_id: dict = None,
    id_to_rating: dict = None,
    num_random_picks: int = 5,
    top_cutoff: int = 20
) -> list:
    """
    1) Find the top N graph-based games similar to target_game_id
    2) Randomly pick from the top-rated games in target_genre
    3) Intersect the two sets (both are game names)
    """

    # Step A: graph-based top 20 IDs -> convert to names (and skip unknown)
    rec_graph_names = recommend_by_graph(
        game_graph, target_game_id, top_n=20,
        id_to_name=id_to_name,
        id_to_genres=id_to_genres
    )
    rec_graph_set = set(rec_graph_names)

    # Step B: random high-rating from the tree
    # We ignore the old "recommend_by_tree"
    rec_tree_random = recommend_by_tree_random_high_rating(
        genre_tree_root,
        target_genre,
        name_to_id=name_to_id,
        id_to_rating=id_to_rating,
        num_random_picks=num_random_picks,
        top_cutoff=top_cutoff
    )
    rec_tree_set = set(rec_tree_random)

    # Intersection in terms of game names
    final_rec = list(rec_graph_set & rec_tree_set)

    # Optional: if empty, show some message or do something else
    if not final_rec:
        print(f"\n[Hybrid] The intersection of graph-similar vs. random top-rated in '{target_genre}' is empty!")
        return []

    return final_rec


def recommend_by_tree_random_high_rating(
        genre_tree_root: Node,
        target_genre: str,
        name_to_id: dict,
        id_to_rating: dict,
        num_random_picks: int = 5,
        top_cutoff: int = 20
) -> list:
    """
    Gathers all games under the target genre node, looks up their rating,
    sorts them by rating descending, then randomly picks `num_random_picks`
    from the top `top_cutoff` results (or fewer if not enough games).

    :param genre_tree_root: root of the genre tree
    :param target_genre: the genre of interest (must match node.name in the tree)
    :param name_to_id: dict mapping game_name -> item_id
    :param id_to_rating: dict mapping item_id -> numeric rating
    :param num_random_picks: how many random picks to return
    :param top_cutoff: how many top results to consider for random picking
    :return: list of game names (randomly chosen from the top rated set)
    """

    # 1. Gather all game names in that genre
    all_games_in_genre = []

    def dfs(node: Node):
        if node.name == target_genre:
            for child in node.children:
                all_games_in_genre.append(child.name)
        for c in node.children:
            dfs(c)

    dfs(genre_tree_root)

    # 2. Convert those names to IDs, then look up ratings
    rated_games = []
    for gname in all_games_in_genre:
        if gname.lower() in name_to_id:
            gid = name_to_id[gname.lower()]
            rating = id_to_rating.get(gid, 0.0)  # default rating 0 if not found
            rated_games.append((gname, rating))

    # 3. Sort by rating descending
    rated_games.sort(key=lambda x: x[1], reverse=True)

    # 4. Slice the top_cutoff, then pick randomly
    top_candidates = rated_games[:top_cutoff]
    if not top_candidates:
        return []

    # If there are fewer than num_random_picks, just return them all
    if len(top_candidates) <= num_random_picks:
        return [g[0] for g in top_candidates]

    # Otherwise, pick random subset
    chosen = random.sample(top_candidates, num_random_picks)
    # chosen is list of (gname, rating) tuples
    return [item[0] for item in chosen]


def build_popularity_stats(df_games: pd.DataFrame, df_user_items: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame containing popularity statistics for each game:
    - item_id
    - number of distinct users (popularity)
    - app_name
    - genres

    We'll then use this table to extract top 10 overall and top 10 in a category.
    """
    # Count how many distinct users own each item_id
    pop_series = df_user_items.groupby('item_id')['user_id'].nunique()
    pop_df = pop_series.reset_index()  # columns will be ['item_id', 0]
    pop_df.columns = ['item_id', 'popularity']  # rename the second column

    # Merge with df_games on item_id = df_games['id']
    # We want [id, app_name, genres, popularity]
    merged = pd.merge(pop_df, df_games, left_on='item_id', right_on='id', how='inner')

    # Convert app_name to string just in case
    merged['app_name'] = merged['app_name'].astype(str).fillna("")
    # Ensure genres is a list, if possible
    merged['genres'] = merged['genres'].apply(lambda g: g if isinstance(g, list) else [])
    return merged[['item_id', 'app_name', 'genres', 'popularity']]


def get_top10_overall_games(pop_stats: pd.DataFrame) -> list:
    """
    Return the names of the top 10 most popular games overall (by popularity descending).
    """
    # Sort by popularity desc, pick top 10
    sorted_df = pop_stats.sort_values(by='popularity', ascending=False).head(10)
    return sorted_df['app_name'].tolist()


def get_top10_in_category_games(pop_stats: pd.DataFrame, category: str) -> list:
    """
    Filter pop_stats to only games whose 'genres' contain the given category,
    then return the top 10 by popularity (descending).
    """
    # Filter rows that have category in genres
    filtered = pop_stats[pop_stats['genres'].apply(lambda g: category in g)]
    sorted_df = filtered.sort_values(by='popularity', ascending=False).head(10)
    return sorted_df['app_name'].tolist()
