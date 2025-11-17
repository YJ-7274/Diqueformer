import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import torch.serialization
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

GRAPH_FILE = "spotify_combined_graph_local.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_CHANNELS = 64
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.001


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

        self.hidden_channels = hidden_channels
        self.num_playlists = num_playlists
        self.num_artists = num_artists
        self.track_feat_dim = track_feat_dim
        self.metadata = metadata

        self.playlist_emb = nn.Embedding(num_playlists, hidden_channels)
        self.artist_emb = nn.Embedding(num_artists, hidden_channels)
        self.track_lin = nn.Linear(track_feat_dim, hidden_channels)
        self.type_emb = nn.Embedding(3, hidden_channels)

        convs = []
        for _ in range(num_layers):
            convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs = nn.ModuleList(convs)

        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def _encode_and_flatten(self, batch: HeteroData):

        device = batch['track'].x.device

        pl_ids = batch['playlist'].n_id
        ar_ids = batch['artist'].n_id

        x_pl = self.playlist_emb(pl_ids)
        x_tr = self.track_lin(batch['track'].x)
        x_ar = self.artist_emb(ar_ids)

        num_pl = x_pl.size(0)
        num_tr = x_tr.size(0)
        num_ar = x_ar.size(0)

        x = torch.cat([x_pl, x_tr, x_ar], dim=0)  # [N_total, H]
        num_total = num_pl + num_tr + num_ar

        # ---- C. Add node-type embeddings ----
        type_id = torch.cat(
            [
                torch.full((num_pl,), 0, dtype=torch.long, device=device),  # playlist
                torch.full((num_tr,), 1, dtype=torch.long, device=device),  # track
                torch.full((num_ar,), 2, dtype=torch.long, device=device),  # artist
            ],
            dim=0,
        )
        x = x + self.type_emb(type_id)  # [N_total, H]

        off_pl = 0
        off_tr = off_pl + num_pl
        off_ar = off_tr + num_tr

        edge_indices = []

        # (playlist, contains, track)
        if ('playlist', 'contains', 'track') in batch.edge_types:
            e_pl_tr = batch[('playlist', 'contains', 'track')].edge_index
            # local indices: playlists in [0, num_pl), tracks in [0, num_tr)
            e_pl_tr = e_pl_tr.clone()
            e_pl_tr[0] += off_pl
            e_pl_tr[1] += off_tr
            edge_indices.append(e_pl_tr)
            # reverse: track -> playlist
            edge_indices.append(e_pl_tr.flip(0))

        # (track, by, artist)
        if ('track', 'by', 'artist') in batch.edge_types:
            e_tr_ar = batch[('track', 'by', 'artist')].edge_index
            e_tr_ar = e_tr_ar.clone()
            e_tr_ar[0] += off_tr
            e_tr_ar[1] += off_ar
            edge_indices.append(e_tr_ar)

            # reverse: artist -> track
            edge_indices.append(e_tr_ar.flip(0))

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        return x, edge_index, (num_pl, num_tr, num_ar)

    def forward(self, batch: HeteroData):
        """
        batch: HeteroData mini-batch from LinkNeighborLoader, containing:
          - playlist, track, artist nodes
          - edges for both relations
          - edge_label_index and edge_label for ('playlist', 'contains', 'track')

        Returns:
          logits: [num_edge_labels] (before sigmoid)
        """
        device = next(self.parameters()).device
        batch = batch.to(device)

        x, edge_index, (num_pl, num_tr, num_ar) = self._encode_and_flatten(batch)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        off_pl = 0
        off_tr = off_pl + num_pl

        x_pl_out = x[off_pl : off_pl + num_pl]
        x_tr_out = x[off_tr : off_tr + num_tr]

        target_edge = ('playlist', 'contains', 'track')
        edge_label_index = batch[target_edge].edge_label_index
        row, col = edge_label_index

        pl_emb = x_pl_out[row]
        tr_emb = x_tr_out[col]

        logits = self.predictor(torch.cat([pl_emb, tr_emb], dim=-1)).view(-1)
        return logits

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for step, batch in tqdm(enumerate(loader), desc="Training", total=len(loader)):
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)

        label = batch[('playlist', 'contains', 'track')].edge_label
        label = label.to(pred.dtype)

        loss = F.binary_cross_entropy_with_logits(pred, label)

        loss.backward()
        optimizer.step()

        num_edges = label.numel()
        total_loss += loss.item() * num_edges
        total_examples += num_edges

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def test(model, loader, device):
    """
    Evaluate model on a validation/test loader using ROC-AUC.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
        batch = batch.to(device)

        pred = model(batch)

        label = batch[('playlist', 'contains', 'track')].edge_label
        label = label.float()

        all_preds.append(torch.sigmoid(pred).cpu())
        all_labels.append(label.cpu())

    if not all_labels:
        return float('nan')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return float('nan')

    auc = roc_auc_score(all_labels, all_preds)
    return float(auc)

def main():
    print(f"Using device: {DEVICE}")

    if not os.path.exists(GRAPH_FILE):
        print(f"Error: Graph file not found at {GRAPH_FILE}")
        print("Please run the graph creation script first.")
        return
        
    print(f"Loading graph from {GRAPH_FILE}...")
    torch.serialization.add_safe_globals([HeteroData])
    data = torch.load(GRAPH_FILE, weights_only=False)
    print("Graph loaded successfully:")
    print(data)
    
    print("\nSplitting graph links for training, validation, and testing")
    target_edge = ('playlist', 'contains', 'track')
    
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[target_edge],
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
        is_undirected=False
    )

    train_data, val_data, test_data = transform(data)
    
    print("\nData split complete:")
    print(f"  Training data:   {train_data}")
    print(f"  Validation data: {val_data}")
    print(f"  Test data:       {test_data}")

    print("\nCreating 'chunk' loaders...")
    
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[15, 10],  # 2 hop neighborhood sampling
        batch_size=BATCH_SIZE,
        edge_label_index=(target_edge, train_data[target_edge].edge_label_index),
        neg_sampling_ratio=1.0,  # 1 negative sample per 1 positive
        shuffle=True,
        num_workers=0,
    )
    
    try:
        print("Loader length:", len(train_loader))
    except:
        print("Loader length unavailable (normal for LinkNeighborLoader)")

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[15, 10],
        batch_size=BATCH_SIZE,
        edge_label_index=(target_edge, val_data[target_edge].edge_label_index),
        neg_sampling_ratio=1.0, # Evaluate on equal positive/negative
        shuffle=False,
        num_workers=0,
    )
    
    print("Loaders created.")
    
    print("\nInitializing model")
    model = ExphormerModel(
        hidden_channels=HIDDEN_CHANNELS,
        num_playlists=data['playlist'].num_nodes,
        num_artists=data['artist'].num_nodes,
        track_feat_dim=data['track'].x.shape[1],
        metadata=data.metadata(),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model ready for training.")
    print(model)

    print("\nStarting Training Loop")
    
    best_val_auc = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        loss = train(model, train_loader, optimizer, DEVICE)
        print(f"Epoch {epoch}: Training Loss = {loss:.4f}")

        val_auc = test(model, val_loader, DEVICE)
        print(f"Epoch {epoch}: Validation AUC = {val_auc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }, f"checkpoints/checkpoint_epoch_{epoch}.pt")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/exphormer_best_model.pt")
            print("Saved new best model")
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()