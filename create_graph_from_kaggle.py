import os
import json
import torch
import pandas as pd
import kagglehub
import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# 1. Handle for the CSV with 1M tracks and their features
FEATURES_DATASET = "amitanshjoshi/spotify-1million-tracks"

# 2. Handle for the 1M playlist dataset (JSONs)
PLAYLIST_DATASET = "himanshuwagh/spotify-million"

# 3. Path to save the final graph file
OUTPUT_FILE = "spotify_combined_graph.pt"
# -------------------------------------------------------------------

def load_track_feature_store(dataset_handle):
    """
    Downloads the 1M track CSV from KaggleHub and loads it as a
    feature lookup table.
    """
    print(f"Downloading feature dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    
    csv_path = os.path.join(dataset_path, "spotify_data.csv")
    print(f"Loading feature store from {csv_path}...")
    
    feature_cols = [
        'popularity', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'time_signature', 'duration_ms'
    ]
    key_cols = ['artist_name', 'track_name', 'track_id']
    
    try:
        df = pd.read_csv(csv_path, usecols=key_cols + feature_cols)
    except ValueError as e:
        print(f"Error: CSV file is missing required columns.")
        print(f"Missing: {e}")
        return None, None
    
    print("Feature store loaded. Creating lookup key...")

    # Drop any tracks that have no ID
    df = df.dropna(subset=['track_id'])

    # Format the ID to match the JSONs
    print("Standardizing track URIs for matching...")
    df['track_uri'] = 'spotify:track:' + df['track_id'].astype(str)

    # Drop duplicates before setting the index
    print(f"Dropping duplicate track URIs. Original size: {len(df)}")
    df = df.drop_duplicates(subset=['track_uri'])
    print(f"New size after de-duping: {len(df)}")
    
    # --- Feature Preprocessing ---
    print("Cleaning and normalizing features...")
    
    # CRITICAL FIX: Ensure all feature columns are numeric
    for col in feature_cols:
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where ANY feature is NaN (after conversion)
    print(f"Rows before removing NaN features: {len(df)}")
    df = df.dropna(subset=feature_cols)
    print(f"Rows after removing NaN features: {len(df)}")
    
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Now, set the index (which is now guaranteed to be unique)
    df = df.set_index('track_uri')
    
    print("Feature store ready.")
    return df, feature_cols

def parse_playlists_and_build_graph(dataset_handle, feature_store, feature_cols):
    """
    Downloads the 1M playlist dataset, parses all 1,000 JSON slices, 
    matches tracks to the feature store, and builds the HeteroData object.
    """
    print(f"Downloading playlist dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    
    # The JSON files are in a subdirectory named 'data'
    playlist_dir = os.path.join(dataset_path, 'data')
    
    json_files = [f for f in os.listdir(playlist_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON slices. Starting graph creation...")

    # Mappings from URI/PID to a unique integer index
    playlist_map = {}  # pid -> int_id
    track_map = {}     # track_uri -> int_id
    artist_map = {}    # artist_uri -> int_id
    
    # Edge lists
    pt_edges_src = []
    pt_edges_dst = []
    ta_edges_src = []
    ta_edges_dst = []
    
    # This will store our track features, in the correct order
    track_features_list = []

    # --- Pass 1: Parse JSONs, build mappings, and create edge lists ---
    for file_name in tqdm(json_files, desc="Parsing JSON slices"):
        file_path = os.path.join(playlist_dir, file_name)
        with open(file_path, 'r') as f:
            slice_data = json.load(f)
            
            for playlist in slice_data['playlists']:
                # Get or create playlist node ID
                pid = playlist['pid']
                if pid not in playlist_map:
                    playlist_map[pid] = len(playlist_map)
                playlist_idx = playlist_map[pid]
                
                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    artist_uri = track['artist_uri']
                    
                    # CRITICAL: We only add the track to our graph
                    # IF it exists in our feature store.
                    if track_uri in feature_store.index:
                        
                        # --- Track Node ---
                        if track_uri not in track_map:
                            track_map[track_uri] = len(track_map)
                            # Add its features to our list, in order
                            features = feature_store.loc[track_uri][feature_cols].values
                            
                            # CRITICAL FIX: Convert to float64 and ensure it's a proper numpy array
                            features = features.astype(np.float64)
                            track_features_list.append(features)
                        track_idx = track_map[track_uri]
                        
                        # --- Artist Node ---
                        if artist_uri not in artist_map:
                            artist_map[artist_uri] = len(artist_map)
                        artist_idx = artist_map[artist_uri]

                        # --- Add Edges ---
                        pt_edges_src.append(playlist_idx)
                        pt_edges_dst.append(track_idx)
                        
                        ta_edges_src.append(track_idx)
                        ta_edges_dst.append(artist_idx)

    print("Parsing complete. Assembling tensors...")

    # --- Pass 2: Assemble the HeteroData object ---
    data = HeteroData()

    num_playlists = len(playlist_map)
    num_tracks = len(track_map)
    num_artists = len(artist_map)
    
    # --- Add Node Features ---
    # For playlists and artists, we don't have features from the CSV.
    # We just tell PyG how many nodes there are.
    # Your model can (and should) create learnable embeddings for them.
    data['playlist'].num_nodes = num_playlists
    data['artist'].num_nodes = num_artists

    # For tracks, we have our rich numerical features!
    # CRITICAL FIX: Explicitly stack as float64 numpy array first
    track_features_array = np.vstack(track_features_list).astype(np.float32)
    data['track'].x = torch.from_numpy(track_features_array)

    # --- Add Edge Indices ---
    data['playlist', 'contains', 'track'].edge_index = torch.tensor([pt_edges_src, pt_edges_dst], dtype=torch.long)
    data['track', 'by', 'artist'].edge_index = torch.tensor([ta_edges_src, ta_edges_dst], dtype=torch.long)
    
    # De-duplicate edges (e.g., same track-artist pair)
    print("Coalescing graph (de-duplicating edges)...")
    data = data.coalesce()

    print("Graph assembly complete.")
    return data


def main():
    # Step 1: Download & load the CSV as a feature lookup table
    feature_store, feature_cols = load_track_feature_store(FEATURES_DATASET)
    if feature_store is None:
        return

    # Step 2: Download & parse playlists, matching them to build the graph
    graph_data = parse_playlists_and_build_graph(PLAYLIST_DATASET, feature_store, feature_cols)

    print("\n--- Final Graph Structure ---")
    print(graph_data)
    print(f"Track features shape: {graph_data['track'].x.shape}")
    
    # Step 3: Save the final "big graph" to disk
    print(f"\nSaving graph to {OUTPUT_FILE}...")
    torch.save(graph_data, OUTPUT_FILE)
    print("Done! You can now load this file for training.")


if __name__ == "__main__":
    main()

