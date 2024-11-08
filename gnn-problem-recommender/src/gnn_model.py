import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
from data_processing import load_data, create_graph
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 학습 함수
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 평가 함수
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
        val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average="weighted")
    return train_acc, val_acc, test_acc, f1

# 시각화 및 그래프 저장 함수
def visualize_and_save_metrics(epochs, losses, train_accs, val_accs, test_accs, f1_scores, save_dir="data/processed/gnn_score"):
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(epochs)

    # 손실 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_loss.png")
    plt.close()

    # 정확도 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy')
    plt.plot(epochs_range, test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy_over_epochs.png")
    plt.close()

    # F1 점수 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, f1_scores, label='F1 Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/f1_score_over_epochs.png")
    plt.close()
    print(f"Graphs saved to {save_dir}")

# 주요 학습 과정
def main():
    # 데이터 로드 및 전처리
    node_path = "data/processed/nodes_korean_final.json"
    edge_path = "data/processed/edges_korean_final.json"
    nodes_df, edges_df = load_data(node_path, edge_path)
    data = create_graph(nodes_df, edges_df)

    # 주제 키워드와 라벨 매핑 생성
    topics = ["Finance", "Loan", "Cryptocurrency", "Portfolio", "Economic Growth", 
              "Income Tax", "Insurance", "Stock Market", "International Trade", "Pension"]
    topic_to_label = {topic: idx for idx, topic in enumerate(topics)}
    label_to_topic = {idx: topic for topic, idx in topic_to_label.items()}

    # 각 노드에 주제 키워드 라벨링
    num_nodes = data.num_nodes
    data.x = torch.eye(num_nodes, dtype=torch.float32)  # 노드 특성
    data.y = torch.tensor([topic_to_label[np.random.choice(topics)] for _ in range(num_nodes)], dtype=torch.long)

    # 라벨-주제 매핑 저장
    with open("data/processed/topic_mapping.pkl", "wb") as f:
        pickle.dump(label_to_topic, f)
    print("Label-to-topic mapping saved to data/processed/topic_mapping.pkl")

    # 학습/검증/테스트 마스크 설정
    indices = np.random.permutation(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size+val_size]] = True
    data.test_mask[indices[train_size+val_size:]] = True

    # 모델 초기화 및 학습
    input_dim = data.x.shape[1]
    hidden_dim = 64
    # 주요 학습 과정에서 output_dim을 64로 설정
    output_dim = 64
    model = GCN(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 학습 및 평가 지표 저장을 위한 리스트 초기화
    losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    f1_scores = []

    epochs = 100
    for epoch in range(epochs):
        loss = train(model, data, optimizer, criterion)
        losses.append(loss)

        train_acc, val_acc, test_acc, f1 = evaluate(model, data)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        f1_scores.append(f1)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}, F1 Score: {f1:.4f}")

    # 학습 종료 후 시각화 및 그래프 저장 함수 호출
    visualize_and_save_metrics(epochs, losses, train_accs, val_accs, test_accs, f1_scores)

    # 모델 및 데이터 저장
    model_save_path = "models/gnn_model.pth"
    data_save_path = "data/processed/gnn_data.pkl"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    with open(data_save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved to {model_save_path}")
    print(f"Data saved to {data_save_path}")

if __name__ == "__main__":
    main()
