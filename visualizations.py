import ast
from collections import defaultdict
import plotly.graph_objects as go

def visualize_genre_distribution_from_bundle(bundle_json_path: str):
    """
    Reads bundle_data.json, counts genre frequency across all bundles,
    and generates an interactive bar chart using Plotly.
    """

    with open(bundle_json_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        bundle_data = [ast.literal_eval(line) for line in raw_lines if line.strip().startswith("{")]

    bundle_items = []
    for bundle in bundle_data:
        for item in bundle.get("items", []):
            genres = item.get("genre", "").split(", ")
            bundle_items.append({
                "Game": item.get("item_name", "Unknown Game"),
                "Genre": genres
            })

    genre_count = defaultdict(int)
    for item in bundle_items:
        for genre in item["Genre"]:
            if genre:
                genre_count[genre.strip()] += 1

    genres = list(genre_count.keys())
    counts = list(genre_count.values())

    fig = go.Figure(data=[go.Bar(
        x=genres,
        y=counts,
        text=counts,
        textposition='auto',
        hovertext=genres,
    )])

    fig.update_layout(
        title="\U0001F3AE Genre Frequency in Steam Bundles",
        xaxis_title="Genre",
        yaxis_title="Number of Games",
    )

    fig.show()
