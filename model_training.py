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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# -------------------------------------------------------------------
#  CONFIGURATION
# -------------------------------------------------------------------

GRAPH_FILE = "spotify_combined_graph_local.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
HIDDEN_CHANNELS = 64
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# -------------------------------------------------------------------
#  1. MODEL SKELETON
# -------------------------------------------------------------------


class ExphormerModel(nn.Module):
    """
    Heterogeneous playlist–track–artist link prediction model with:

    - Node encoders (ID embeddings + track feature projection)
    - Node-type embeddings (playlist / track / artist)
    - Flattening to a homogeneous graph
    - Simple GraphSAGE backbone (placeholder for Exphormer/SpExphormer)
    - MLP link predictor on (playlist, track) pairs
    """

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
        self.metadata = metadata  # not used directly, but available if needed

        # ------------------------------------------------------------
        # 1. Node encoders
        # ------------------------------------------------------------
        # Feature-less nodes: playlists and artists.
        self.playlist_emb = nn.Embedding(num_playlists, hidden_channels)
        self.artist_emb = nn.Embedding(num_artists, hidden_channels)

        # Track nodes: project 15-dim features to hidden_channels.
        self.track_lin = nn.Linear(track_feat_dim, hidden_channels)

        # Node-type embeddings: 0 = playlist, 1 = track, 2 = artist.
        self.type_emb = nn.Embedding(3, hidden_channels)

        # ------------------------------------------------------------
        # 2. Backbone (placeholder): GraphSAGE stack
        #    Later you can replace this with Exphormer / SpExphormer.
        # ------------------------------------------------------------
        convs = []
        for _ in range(num_layers):
            convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs = nn.ModuleList(convs)

        # ------------------------------------------------------------
        # 3. Link predictor for (playlist, contains, track)
        # ------------------------------------------------------------
        # Simple MLP on concatenated (playlist_emb, track_emb).
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    # ----------------------------------------------------------------
    # Helper: encode nodes and build homogeneous x, edge_index
    # ----------------------------------------------------------------
    def _encode_and_flatten(self, batch: HeteroData):
        """
        Given a HeteroData mini-batch from LinkNeighborLoader, produce:
          - x: [N_total, H] flattened node features
          - edge_index: [2, E_total] homogeneous edges (with reverse edges)
          - (num_pl, num_tr, num_ar): counts for slicing back
        """

        device = batch['track'].x.device  # everything should be on same device

        # ---- A. Encode per-type features ----
        # LinkNeighborLoader provides global node IDs in .n_id.
        pl_ids = batch['playlist'].n_id        # [num_pl_batch]
        ar_ids = batch['artist'].n_id          # [num_ar_batch]

        x_pl = self.playlist_emb(pl_ids)       # [num_pl, H]
        x_tr = self.track_lin(batch['track'].x)  # [num_tr, H]
        x_ar = self.artist_emb(ar_ids)         # [num_ar, H]

        num_pl = x_pl.size(0)
        num_tr = x_tr.size(0)
        num_ar = x_ar.size(0)

        # ---- B. Concatenate to a single node feature matrix ----
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

        # ---- D. Build homogeneous edge_index with offsets ----
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
            edge_index = torch.cat(edge_indices, dim=1)  # [2, E_total]
        else:
            # Edge-less batch (unlikely but safe-guard)
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        return x, edge_index, (num_pl, num_tr, num_ar)

    # ----------------------------------------------------------------
    # Forward: link prediction on (playlist, contains, track)
    # ----------------------------------------------------------------
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

        # 1) Encode nodes and build homogeneous graph
        x, edge_index, (num_pl, num_tr, num_ar) = self._encode_and_flatten(batch)

        # 2) Run backbone (GraphSAGE stack here; swap with Exphormer later)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # 3) Unflatten: slice back to per-type embeddings
        off_pl = 0
        off_tr = off_pl + num_pl
        # off_ar = off_tr + num_tr  # if needed

        x_pl_out = x[off_pl : off_pl + num_pl]  # [num_pl, H]
        x_tr_out = x[off_tr : off_tr + num_tr]  # [num_tr, H]

        # 4) Link prediction for (playlist, contains, track)
        target_edge = ('playlist', 'contains', 'track')
        edge_label_index = batch[target_edge].edge_label_index  # [2, num_edges]
        row, col = edge_label_index  # row: playlist local idx, col: track local idx

        pl_emb = x_pl_out[row]  # [num_edges, H]
        tr_emb = x_tr_out[col]  # [num_edges, H]

        logits = self.predictor(torch.cat([pl_emb, tr_emb], dim=-1)).view(-1)
        return logits


# -------------------------------------------------------------------
#  2. TRAINING & VALIDATION FUNCTIONS (Skeleton)
# -------------------------------------------------------------------

def train(model, loader, optimizer, device):
    """
    One epoch of training.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0

    for step, batch in tqdm(enumerate(loader), desc="Training"):
        # 1. Move batch to device
        batch = batch.to(device)

        # DEBUG: inspect the first batch
        if step < 5:
            edge_index = batch.get_edge_index()
            x = batch.x
            print("[DEBUG] batch ", step)
            print("x.shape:", x.shape)
            print("edge_index dtype:", edge_index.dtype)
            print("edge_index min:", edge_index.min().item(),
                  "max:", edge_index.max().item())
            # hard check
            assert edge_index.min().item() >= 0, "negative node index"
            assert edge_index.max().item() < x.size(0), (
                f"edge_index has node >= x.size(0): "
                f"max={edge_index.max().item()} vs x.size(0)={x.size(0)}"
            )

        # 2. Clear gradients
        optimizer.zero_grad()

        # 3. Forward pass: logits for all edge labels in this batch
        pred = model(batch)  # shape [num_edges_in_batch]

        # 4. Get labels for (playlist, contains, track)
        label = batch[('playlist', 'contains', 'track')].edge_label
        label = label.to(pred.dtype)  # float32 for BCE loss

        # 5. Compute loss
        loss = F.binary_cross_entropy_with_logits(pred, label)

        # 6. Backprop + update
        loss.backward()
        optimizer.step()

        # 7. Accumulate (weighted by number of edges)
        num_edges = label.numel()
        total_loss += loss.item() * num_edges
        total_examples += num_edges

    # Average loss per edge over the epoch
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def test(model, loader, device):
    """
    Evaluate model on a validation/test loader using ROC-AUC.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        # 1. Move batch to device
        batch = batch.to(device)

        # 2. Forward pass: logits
        pred = model(batch)  # [num_edges]

        # 3. Get labels
        label = batch[('playlist', 'contains', 'track')].edge_label
        label = label.float()

        # 4. Store predictions (as probabilities) and labels on CPU
        all_preds.append(torch.sigmoid(pred).cpu())
        all_labels.append(label.cpu())

    if not all_labels:
        return float('nan')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Edge case: if all labels are the same, ROC-AUC is undefined
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return float('nan')

    auc = roc_auc_score(all_labels, all_preds)
    return float(auc)


# -------------------------------------------------------------------
#  3. MAIN EXECUTION (Loading, Splitting, Training)
# -------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")

    # --- 1. Load the "Big Graph" ---
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: Graph file not found at {GRAPH_FILE}")
        print("Please run the graph creation script first.")
        return
        
    print(f"Loading graph from {GRAPH_FILE}...")
    data = torch.load(GRAPH_FILE)
    print("Graph loaded successfully:")
    print(data)
    
    # --- 2. Split the Graph Links ---
    print("\nSplitting graph links for training, validation, and testing...")
    target_edge = ('playlist', 'contains', 'track')
    
    # This transform splits the *edges* for link prediction.
    # It "hides" 10% for validation and 10% for testing.
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

    # --- 3. Create "Chunk" Loaders (LinkNeighborLoader) ---
    print("\nCreating 'chunk' loaders...")
    
    # Training loader: samples positive and negative links
    # FIX: Access edge_label_index from the specific edge type
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[15, 10],  # 2-hop neighborhood sampling
        batch_size=BATCH_SIZE,
        edge_label_index=(target_edge, train_data[target_edge].edge_label_index),
        neg_sampling_ratio=1.0,  # 1 negative sample per 1 positive
        shuffle=True,
        num_workers=0 
    )

    # Validation loader: samples positive and negative links
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[15, 10],
        batch_size=BATCH_SIZE,
        edge_label_index=(target_edge, val_data[target_edge].edge_label_index),
        neg_sampling_ratio=1.0, # Evaluate on equal positive/negative
        shuffle=False,
        num_workers=0
    )
    
    print("Loaders created.")
    
    # You can inspect the first "chunk" from the loader:
    # first_chunk = next(iter(train_loader))
    # print("\nFirst training chunk (subgraph):")
    # print(first_chunk)
    
    # --- 4. Initialize Model and Optimizer ---
    print("\nInitializing model...")
    model = ExphormerModel(
        hidden_channels=HIDDEN_CHANNELS,
        num_playlists=data['playlist'].num_nodes,
        num_artists=data['artist'].num_nodes,
        track_feat_dim=data['track'].x.shape[1],
        metadata=data.metadata(),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Optimizer would be defined here
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model ready for training.")
    print(model)

    # --- 5. Training Loop Skeleton ---
    print("\n--- Starting Training Loop ---")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # --- TRAINING ---
        loss = train(model, train_loader, optimizer, DEVICE) # Pass None for optimizer
        print(f"Epoch {epoch}: Training Loss = {loss:.4f}")
        
        # --- VALIDATION ---
        val_auc = test(model, val_loader, DEVICE)
        print(f"Epoch {epoch}: Validation AUC = {val_auc:.4f}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()

