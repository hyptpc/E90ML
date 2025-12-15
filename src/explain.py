import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from common import (
    E90GraphDataset,
    create_gnn_model_from_params,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PTH_DIR,
    DEFAULT_TUNE_DIR,
    DEFAULT_LABEL_MAPPING,
    get_config_value,
    _resolve_seed,
    split_track_features,
    load_best_params,
    resolve_model_output_path,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    load_data,
    apply_plot_style,
)

def explain_model(config, base_dir):
    apply_plot_style()
    plt.switch_backend('Agg')

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("tuning", {})
    explain_cfg = config.get("explain", {})
    
    # Output location
    save_dir_raw = explain_cfg.get("save_dir", "explanations")
    output_dir = resolve_dir(save_dir_raw, DEFAULT_OUTPUT_DIR, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Explanation results will be saved to: {output_dir}")

    # Filenames
    feat_prefix = explain_cfg.get("feature_importance_prefix", "feat_imp")
    edge_prefix = explain_cfg.get("edge_importance_prefix", "edges")
    
    # Target selection
    target_labels = explain_cfg.get("target_labels", [0, 1])
    samples_per_label = int(explain_cfg.get("samples_per_label", 2))

    # --- 1. Load Data (Evaluation mode) ---
    files = resolve_data_files(data_cfg, base_dir)
    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping", DEFAULT_LABEL_MAPPING)
    if not files or not tree_name or not features or not label_column:
        raise ValueError("Config must define data.files, data.tree_name, data.feature_columns, and data.label_column.")
    feature_cols_dict = split_track_features(features)
    
    fraction = float(explain_cfg.get("fraction", 0.01))
    seed = _resolve_seed(explain_cfg.get("seed"), config.get("seed"))
    df, num_classes = load_data(
        files=files, tree_name=tree_name, features=features,
        label_column=label_column, label_mapping=label_mapping,
        fraction=fraction, random_state=seed,
    )
    dataset = E90GraphDataset(df, feature_cols_dict, label_column)
    loader = PyGDataLoader(dataset, batch_size=1, shuffle=True)

    # --- 2. Load Model ---
    params, best_params_path = load_best_params(training_cfg, tuning_cfg, base_dir, DEFAULT_TUNE_DIR)

    model_output_path = resolve_model_output_path(training_cfg, base_dir, DEFAULT_PTH_DIR, must_exist=True)
    
    device = resolve_device(
        explain_cfg.get("device")
        or training_cfg.get("device")
        or config.get("device", "cpu")
    )
    model = create_gnn_model_from_params(params, input_dim=5, num_classes=num_classes).to(device)
    
    checkpoint = torch.load(model_output_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    # --- 3. Setup Explainer ---
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification' if num_classes > 2 else 'binary_classification',
            task_level='graph',
            return_type='probs',
        ),
    )

    # --- 4. Run Explanation ---
    feature_names = ["ux", "uy", "uz", "dedx", "is_scat"]
    counts = {l: 0 for l in target_labels}

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        label = int(batch.y.item())
        
        if label in target_labels and counts[label] < samples_per_label:
            print(f"Explaining event {i} (Label: {label})...")
            
            explanation = explainer(batch.x, batch.edge_index, batch=batch.batch)
            
            # Save Feature Importance Plot
            node_mask = explanation.node_mask.cpu().detach().numpy()
            avg_importance = np.mean(node_mask, axis=0)
            
            plt.figure(figsize=(8, 5))
            plt.bar(feature_names, avg_importance, color='skyblue')
            plt.title(f"Feature Importance (Event {i}, Label {label})")
            plt.ylabel("Importance Score")
            plt.tight_layout()
            save_path = output_dir / f"{feat_prefix}_evt{i}_label{label}.png"
            plt.savefig(save_path)
            plt.close()

            # Save Edge Importance List
            edge_mask = explanation.edge_mask.cpu().detach().numpy()
            edge_index = batch.edge_index.cpu().numpy()
            top_indices = np.argsort(edge_mask)[::-1][:5]
            
            txt_path = output_dir / f"{edge_prefix}_evt{i}_label{label}.txt"
            with open(txt_path, "w") as f:
                f.write(f"Top important edges for Event {i} (Label {label}):\n")
                for idx in top_indices:
                    src, dst = edge_index[:, idx]
                    score = edge_mask[idx]
                    f.write(f"Node {src} -> Node {dst} : Score {score:.4f}\n")
            
            counts[label] += 1
            
        if all(c >= samples_per_label for c in counts.values()):
            break

    print("Explanation finished.")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Explain GNN model.")
    args.add_argument("-c", "--config", required=True)
    args = args.parse_args()
    config, base_dir = load_config(args.config)
    explain_model(config, base_dir)
