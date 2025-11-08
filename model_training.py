import torch
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
import os

# -------------------------------------------------------------------
#  CONFIGURATION
# -------------------------------------------------------------------

GRAPH_FILE = "spotify_combined_graph.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_CHANNELS = 64
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# -------------------------------------------------------------------
#  1. MODEL SKELETON
# -------------------------------------------------------------------

class ExphormerModel(Module):
    """
    This is where you would define your Exphormer model.
    It should be compatible with PyTorch Geometric's HeteroData.
    """
    def __init__(self, hidden_channels, num_playlists, num_artists, track_feat_dim, metadata):
        super().__init__()
        # --- Skeleton ---
        # 1. Define embeddings for feature-less nodes:
        # self.playlist_emb = Embedding(num_playlists, hidden_channels)
        # self.artist_emb = Embedding(num_artists, hidden_channels)
        
        # 2. Define a linear projection for track features:
        # self.track_lin = Linear(track_feat_dim, hidden_channels)
        
        # 3. Define your heterogeneous Exphormer layers:
        # self.exphormer_conv1 = ...
        # self.exphormer_conv2 = ...
        
        # 4. Define the final link predictor:
        # self.predictor = ...
        pass

    def forward(self, batch):
        """
        Forward pass takes a 'batch' (a mini subgraph chunk).
        """
        # --- Skeleton ---
        # 1. Get initial node representations from the batch
        #    - Use batch['...'].n_id for embeddings
        #    - Use batch['...'].x for features
        
        # 2. Pass them through your Exphormer GNN layers
        #    - x_dict = self.gnn(x_dict, batch.edge_index_dict)
        
        # 3. Get the final embeddings for the target links
        #    - src_nodes = x_dict['playlist'][batch[...].edge_label_index[0]]
        #    - dst_nodes = x_dict['track'][batch[...].edge_label_index[1]]
        
        # 4. Return the predicted score
        #    - return self.predictor(src_nodes, dst_nodes)
        
        return None

# -------------------------------------------------------------------
#  2. TRAINING & VALIDATION FUNCTIONS (Skeleton)
# -------------------------------------------------------------------

def train(model, loader, optimizer, device):
    """
    One epoch of training.
    """
    model.train()
    total_loss = 0
    
    # --- Skeleton ---
    # for batch in tqdm(loader, desc="Training"):
        # 1. Move batch to device
        # 2. Clear gradients: optimizer.zero_grad()
        # 3. Get predictions: pred = model(batch)
        # 4. Get labels: label = batch[...].edge_label.float()
        # 5. Calculate loss: loss = F.binary_cross_entropy_with_logits(pred, label)
        # 6. Backprop: loss.backward()
        # 7. Update weights: optimizer.step()
        # 8. Add to total loss
        
    print("  (Skeleton: Training loop would run here)")
    return 0.0  # Return dummy loss

@torch.no_grad()
def test(model, loader, device):
    """
    Tests the model on a validation/test loader.
    """
    model.eval()
    
    # --- Skeleton ---
    # all_preds = []
    # all_labels = []
    # for batch in tqdm(loader, desc="Validating"):
        # 1. Move batch to device
        # 2. Get predictions: pred = model(batch)
        # 3. Get labels: label = batch[...].edge_label
        # 4. Store preds and labels
    
    # 5. Concatenate all preds and labels
    # 6. Calculate metric: return roc_auc_score(all_labels, all_preds)

    print("  (Skeleton: Validation loop would run here)")
    return 0.0  # Return dummy AUC score

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
    data = torch.load(GRAPH_FILE, weights_only=False)
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
        metadata=data.metadata()
    ).to(DEVICE)
    
    # Optimizer would be defined here
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model ready for training.")
    print(model)

    # --- 5. Training Loop Skeleton ---
    print("\n--- Starting Training Loop ---")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # --- TRAINING ---
        loss = train(model, train_loader, None, DEVICE) # Pass None for optimizer
        print(f"Epoch {epoch}: Training Loss = {loss:.4f}")
        
        # --- VALIDATION ---
        val_auc = test(model, val_loader, DEVICE)
        print(f"Epoch {epoch}: Validation AUC = {val_auc:.4f}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()

