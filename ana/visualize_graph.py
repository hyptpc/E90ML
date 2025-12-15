import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from common import load_config, load_data, E90GraphDataset, resolve_data_files

def visualize_event(config, base_dir, event_idx=0):
    # 1. データロード (共通関数を使用)
    data_cfg = config.get("data")
    files = resolve_data_files(data_cfg, base_dir)
    feature_cols = data_cfg.get("feature_columns")
    
    # 少量だけロード
    df, _ = load_data(files, data_cfg["tree_name"], feature_cols, 
                      data_cfg["label_column"], None, fraction=0.01, random_state=42)
    
    # Dataset作成
    feature_cols_dict = {
        't0': feature_cols[0:4], 't1': feature_cols[4:8], 't2': feature_cols[8:12]
    }
    dataset = E90GraphDataset(df, feature_cols_dict, data_cfg["label_column"])
    data = dataset[event_idx]  # 指定したイベントを取得

    # 2. PyG -> NetworkX 変換
    # to_undirected=True にするとエッジの矢印を消せます（今回は双方向結合なので見やすくするため推奨）
    G = to_networkx(data, to_undirected=True)

    # 3. 描画設定
    plt.figure(figsize=(8, 6))
    
    # ノードの色分け (is_scat フラグを見る)
    # xの最後の要素が is_scat (1=ScatPi, 0=Track)
    node_colors = []
    labels = {}
    for i in range(data.num_nodes):
        is_scat = data.x[i, -1].item()
        if is_scat == 1.0:
            node_colors.append("red")
            labels[i] = r"$\pi_{scat}$"
        else:
            node_colors.append("skyblue")
            labels[i] = f"Track {i}"

    # レイアウト決定 (spring_layout 等が一般的)
    pos = nx.spring_layout(G, seed=42) 

    # 描画
    nx.draw(G, pos, with_labels=True, labels=labels, 
            node_color=node_colors, node_size=2000, 
            font_size=12, font_weight="bold", edge_color="gray")
    
    plt.title(f"Event Graph Representation (Label: {data.y.item()})")
    plt.savefig("event_graph_sample.png", dpi=300)
    print("Saved event_graph_sample.png")

if __name__ == "__main__":
    config, base_dir = load_config("../param/usr/exampleGNN.yaml")
    visualize_event(config, base_dir)