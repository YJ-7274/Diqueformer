import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
import numpy as np
import os

GRAPH_FILE = "spotify_combined_graph_local.pt"
CHECKPOINT_FILE = "checkpoints/checkpoint_epoch_10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 64
TOP_K = 10

class ExphormerModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_playlists: int,
        num_artists: int,
        track_feat_dim: int,
        metadata,
        num_layers: int = 2,
    ):
        super().__init__()

        self.playlist_emb = nn.Embedding(num_playlists, hidden_channels)
        self.artist_emb = nn.Embedding(num_artists, hidden_channels)
        self.track_lin = nn.Linear(track_feat_dim, hidden_channels)

        self.type_emb = nn.Embedding(3, hidden_channels)

        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def encode_full_graph(self, data: HeteroData):
        device = next(self.parameters()).device

        num_pl = data['playlist'].num_nodes
        num_tr = data['track'].num_nodes
        num_ar = data['artist'].num_nodes

        track_x = data['track'].x.to(device)

        x_pl = self.playlist_emb(torch.arange(num_pl, device=device))
        x_tr = self.track_lin(track_x)
        x_ar = self.artist_emb(torch.arange(num_ar, device=device))

        x = torch.cat([x_pl, x_tr, x_ar], dim=0)

        type_ids = torch.cat([
            torch.zeros(num_pl, dtype=torch.long, device=device),
            torch.ones(num_tr, dtype=torch.long, device=device),
            torch.full((num_ar,), 2, dtype=torch.long, device=device)
        ])
        x = x + self.type_emb(type_ids)

        edge_indices = []
        off_pl = 0
        off_tr = off_pl + num_pl
        off_ar = off_tr + num_tr

        e_pl_tr = data[('playlist', 'contains', 'track')].edge_index.to(device)
        e_pl_tr = torch.stack([
            e_pl_tr[0] + off_pl,
            e_pl_tr[1] + off_tr
        ])
        edge_indices.append(e_pl_tr)
        edge_indices.append(e_pl_tr.flip(0))

        e_tr_ar = data[('track', 'by', 'artist')].edge_index.to(device)
        e_tr_ar = torch.stack([
            e_tr_ar[0] + off_tr,
            e_tr_ar[1] + off_ar
        ])
        edge_indices.append(e_tr_ar)
        edge_indices.append(e_tr_ar.flip(0))

        edge_index = torch.cat(edge_indices, dim=1)

        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))

        return (
            x[off_pl : off_tr],
            x[off_tr : off_ar],
            x[off_ar : off_ar + num_ar],
        )

    def score(self, pl_emb, tr_emb):
        return self.predictor(torch.cat([pl_emb, tr_emb], dim=-1)).view(-1)


def recommend_for_playlist(model, data, playlist_id, top_k=10):

    model.eval()
    with torch.no_grad():
        print("Encoding full graph embeddings...")
        pl_x, tr_x, _ = model.encode_full_graph(data)

        pl_vec = pl_x[playlist_id].unsqueeze(0)

        num_tracks = tr_x.size(0)
        tr_vecs = tr_x

        pl_expanded = pl_vec.repeat(num_tracks, 1)

        logits = model.score(pl_expanded, tr_vecs)
        scores = torch.sigmoid(logits)

        existing = data[('playlist','contains','track')].edge_index
        existing_tracks = existing[1][existing[0] == playlist_id].tolist()

        scores[existing_tracks] = -1e9

        top_scores, top_indices = torch.topk(scores, top_k)

        return top_indices.cpu().tolist(), top_scores.cpu().tolist()

def main():
    print(f"Using device: {DEVICE}")

    data = torch.load(GRAPH_FILE, weights_only=False)
    print("Graph loaded.")

    model = ExphormerModel(
        hidden_channels=HIDDEN_CHANNELS,
        num_playlists=data['playlist'].num_nodes,
        num_artists=data['artist'].num_nodes,
        track_feat_dim=data['track'].x.shape[1],
        metadata=data.metadata(),
    ).to(DEVICE)

    print(f"Loading checkpoint: {CHECKPOINT_FILE}")
    ckpt = torch.load("checkpoints/checkpoint_epoch_10.pt", map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded.")

    playlist_id = 42
    print(f"\nGenerating top-{TOP_K} recommendations for playlist {playlist_id}...")
    track_ids, scores = recommend_for_playlist(model, data, playlist_id, TOP_K)

    print("\nRecommended track IDs:\n", track_ids)
    print("Scores:\n", scores)

    playlist_id = 42

    recall, hidden_tracks, predictions = recall_at_k_for_playlist(
        model, data, playlist_id, visible_k=5, top_k=1000
    )

    print("Hidden tracks (ground truth):", hidden_tracks)
    print("Predicted top tracks:", predictions)
    print(f"Recall@10 = {recall:.4f}")

def recall_at_k_for_playlist(model, data, playlist_id, visible_k=5, top_k=10):
    edge_index = data[('playlist', 'contains', 'track')].edge_index
    all_tracks = edge_index[1][edge_index[0] == playlist_id].cpu().tolist()

    if len(all_tracks) <= visible_k:
        print("Playlist too small — skipping.")
        return None

    visible = set(all_tracks[:visible_k])
    hidden = set(all_tracks[visible_k:])

    model.eval()
    with torch.no_grad():
        pl_x, tr_x, _ = model.encode_full_graph(data.to(DEVICE))

    pl_emb = pl_x[playlist_id].unsqueeze(0)

    num_tracks = tr_x.size(0)
    pl_exp = pl_emb.repeat(num_tracks, 1)
    logits = model.score(pl_exp, tr_x)
    scores = torch.sigmoid(logits)

    for t in visible:
        scores[t] = -1e9

    top_vals, top_idx = torch.topk(scores, top_k)
    top_idx = top_idx.cpu().tolist()

    hits = len(hidden.intersection(top_idx))
    recall = hits / len(hidden)

    return recall, hidden, top_idx

if __name__ == "__main__":
    main()
