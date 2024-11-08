# src/data_processing.py
import json
import pandas as pd
import torch
from torch_geometric.data import Data

def load_data(node_path, edge_path):
    # JSON 파일에서 데이터 로드
    with open(node_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(edge_path, "r", encoding="utf-8") as f:
        edges = json.load(f)
    
    # 노드 데이터프레임 생성
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    return nodes_df, edges_df

def create_graph(nodes_df, edges_df):
    # 노드 ID와 특성을 텐서로 변환
    x = torch.tensor(range(len(nodes_df)), dtype=torch.long)  # 노드 ID를 텐서로 변환 (임베딩 사용 시 대체 가능)
    
    # 엣지 연결 정보 생성
    edge_index = torch.tensor(
        [edges_df["source_node_id"].tolist(), edges_df["target_node_id"].tolist()],
        dtype=torch.long
    )

    # PyTorch Geometric의 그래프 데이터 객체 생성
    data = Data(x=x, edge_index=edge_index)
    
    return data

# 파일 경로
node_path = "data/processed/nodes_korean_final.json"
edge_path = "data/processed/edges_korean_final.json"

# 데이터 로드 및 그래프 생성
nodes_df, edges_df = load_data(node_path, edge_path)
graph_data = create_graph(nodes_df, edges_df)

# 결과 확인
print(graph_data)
