import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib import font_manager
import os

# 절대 경로로 폰트 파일 설정
font_path = "font\malgun.ttf"  # 폰트 절대 경로를 사용합니다.
font_prop = font_manager.FontProperties(fname=font_path)

def visualize_graph(data, nodes_df, num_nodes=100):
    # PyTorch Geometric의 Data 객체를 NetworkX 그래프로 변환
    G = to_networkx(data)
    
    # 노드 수가 많을 경우 일부만 표시
    if num_nodes < len(G):
        G = G.subgraph(list(G.nodes)[:num_nodes])

    # 노드 ID를 한글 레이블로 매핑
    labels = {i: nodes_df.loc[i, "label"] for i in G.nodes}

    # 그래프 시각화
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # 노드 위치 고정
    nx.draw(G, pos, with_labels=False, node_size=500, node_color="lightblue", edge_color="gray", font_size=10)
    
    # 각 노드에 레이블 표시 (폰트 적용)
    for node, (x, y) in pos.items():
        plt.text(x, y, labels[node], fontproperties=font_prop, fontsize=8, ha='center', color="black")
    
    plt.show()

# 사용 예시
if __name__ == "__main__":
    from data_processing import load_data, create_graph
    
    # 데이터 로드 및 그래프 생성
    node_path = "data/processed/nodes_korean_final.json"
    edge_path = "data/processed/edges_korean_final.json"
    nodes_df, edges_df = load_data(node_path, edge_path)
    graph_data = create_graph(nodes_df, edges_df)
    
    # 그래프 시각화
    visualize_graph(graph_data, nodes_df)
