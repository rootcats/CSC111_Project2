"""
main.py

Entry point file:
1. Load all data (games, user_items, user_reviews)
2. Build the graph and tree
3. Show popularity rankings and let the user interactively input a game for recommendations.
"""

from visualizations import visualize_genre_distribution_from_bundle

from function import (
    load_steam_games,
    load_user_items,
    load_user_reviews,
    build_game_graph,
    build_genre_tree,
    recommend_by_graph,
    recommend_by_tree_random_high_rating,
    hybrid_recommendation,
    build_popularity_stats,
    get_top10_overall_games,
    get_top10_in_category_games
)


def main():
    """
    This main function:
    1. Loads data (games, user_items, user_reviews)
    2. Builds the game graph and genre tree
    3. Shows a "top 10 games overall" ranking
    4. Lets the user input a game name and genre to get recommendations
       and also see the top 10 popular games in that genre.
    """

    # 1) Load Data
    df_games = load_steam_games("steam_games.json")
    df_user_items = load_user_items("australian_users_items.json")
    df_user_reviews = load_user_reviews("australian_user_reviews.json")
    id_to_name = df_games.set_index('id')['app_name'].to_dict()
    # For genres:
    temp = df_games.set_index('id')['genres'].apply(lambda x: x if isinstance(x, list) else [])
    id_to_genres = temp.to_dict()
    rating_series = df_user_reviews.groupby('item_id')['recommend'].mean()
    id_to_rating = rating_series.to_dict()
    print("Data loading completed. Now building the graph and tree...")

    import json
    with open("bundle_data.json", "r", encoding="utf-8") as f:
        bundle_data = json.load(f)
    print(f"Loaded {len(bundle_data)} bundle entries.")
    visualize_genre_distribution_from_bundle("bundle_data.json")

    # 2) Build Graph & Tree
    game_graph = build_game_graph(df_user_items)
    genre_tree_root = build_genre_tree(df_games)
    print("Graph & tree building complete!")

    # 3) Build popularity stats and show top 10 overall
    pop_stats = build_popularity_stats(df_games, df_user_items)
    top10_all = get_top10_overall_games(pop_stats)
    print("\n===== TOP 10 MOST POPULAR GAMES (OVERALL) =====")
    for i, game_name in enumerate(top10_all, 1):
        print(f"{i}. {game_name}")

    print("\n===== Build Completed: Graph & Tree & Popularity Stats =====")

    # Create a dictionary: app_name -> id (for user input lookup)
    name_to_id = {}
    for idx, row in df_games.iterrows():
        game_id = row['id']
        app_name = str(row.get('app_name', f"Game_{game_id}"))
        name_to_id[app_name.lower()] = game_id

    # Interactive loop
    while True:
        user_input = input("\nPlease enter your favorite game name (type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        # Try to find the game ID from the dictionary
        lower_name = user_input.lower()
        if lower_name not in name_to_id:
            print("Game not found. Please try again or type 'exit' to quit.")
            continue

        # Optionally ask for a genre
        genre_input = input("Enter a genre you're interested in (e.g. 'Action'), or press Enter to skip: ").strip()
        if not genre_input:
            genre_input = "Action"  # default fallback

        # 4) Show top 10 for that genre
        top10_cat = get_top10_in_category_games(pop_stats, genre_input)
        print(f"\n===== TOP 10 MOST POPULAR GAMES IN GENRE '{genre_input}' =====")
        if len(top10_cat) == 0:
            print("No games found in that genre.")
        else:
            for i, gname in enumerate(top10_cat, 1):
                print(f"{i}. {gname}")

        # We have game_id from the dictionary
        # ...
        # After the user picks 'genre_input' and we print top10_cat:

        # We have a random approach to recommend some top-rated games from the chosen genre
        rec_tree_random = recommend_by_tree_random_high_rating(
            genre_tree_root,
            genre_input,
            name_to_id=name_to_id,
            id_to_rating=id_to_rating,
            num_random_picks=5,
            top_cutoff=20
        )
        print(f"\n[Tree-based Recommendation: Random Highly Rated] {rec_tree_random}")

        # We have game_id from the dictionary
        game_id = name_to_id[lower_name]

        rec_graph = recommend_by_graph(game_graph, game_id, top_n=5, id_to_name=id_to_name, id_to_genres=id_to_genres)
        rec_hybrid = hybrid_recommendation(
            game_graph,
            genre_tree_root,
            game_id,  # the ID
            genre_input,  # the genre
            id_to_name=id_to_name,
            id_to_genres=id_to_genres,
            name_to_id=name_to_id,
            id_to_rating=id_to_rating,
            num_random_picks=5,
            top_cutoff=20
        )
        rec_tree = recommend_by_tree_random_high_rating(
            genre_tree_root,
            genre_input,
            name_to_id=name_to_id,
            id_to_rating=id_to_rating,
            num_random_picks=5,
            top_cutoff=20
        )

        print(f"\n[Graph Similarity Recommendation] Games most similar to '{user_input}' (ID: {game_id}): {rec_graph}")
        print(f"[Tree-based Recommendation] Games in the '{genre_input}' category (partial): {rec_tree[:10]}...")
        print(
            f"[Hybrid Recommendation] Intersection of games similar to '{user_input}' and in '{genre_input}' category: \
{rec_hybrid}")

    print("\nExiting program. Goodbye!")


if __name__ == "__main__":
    main()
