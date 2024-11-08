import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import argparse  # 명령줄 인자 처리를 위한 라이브러리
from gnn_model import GCN

# 데이터 및 모델 로드
data_path = "data/processed/gnn_data.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)
print("Data loaded with x shape:", data.x.shape)

# 라벨-주제 매핑 로드
with open("data/processed/topic_mapping.pkl", "rb") as f:
    label_to_topic = pickle.load(f)
    topic_to_label = {v: k for k, v in label_to_topic.items()}
print("Topic mapping loaded.")

# 학습된 모델 불러오기 (output_dim=10 설정)
input_dim = data.x.shape[1]
hidden_dim = 64
output_dim = 64
model = GCN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("models/gnn_model.pth"))
model.eval()
print("Model loaded for recommendation system.")

# 추천 시스템 클래스 정의
class RecommenderSystem:
    def __init__(self, model, data, topic_to_label, label_to_topic, db_path="data/recommendation_problems.db"):
        self.model = model
        self.data = data
        self.topic_to_label = topic_to_label
        self.label_to_topic = label_to_topic
        self.db_path = db_path
        self.node_embeddings = None

    def generate_node_embeddings(self):
        """ 학습된 모델을 사용하여 각 노드의 임베딩을 생성 """
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model(self.data.x, self.data.edge_index).cpu().numpy()

    def get_keyword_embedding(self, keyword):
        """키워드에 해당하는 노드의 GNN 임베딩을 가져오는 함수"""
        label = self.topic_to_label.get(keyword)
        if label is None:
            print(f"'{keyword}'에 해당하는 라벨을 찾을 수 없습니다.")
            return None
        labels = self.data.y.cpu().numpy()
        node_ids = np.where(labels == label)[0]
        if len(node_ids) == 0:
            print(f"'{keyword}'에 해당하는 노드를 찾을 수 없습니다.")
            return None
        # 첫 번째 노드의 임베딩을 사용
        return self.node_embeddings[node_ids[0]]

    def recommend_problems_by_keyword(self, keyword, top_k=5):
        """ 입력 키워드의 임베딩을 기반으로 가장 관련성 높은 문제를 추천 """
        # 키워드의 GNN 임베딩 생성
        keyword_embedding = self.get_keyword_embedding(keyword)
        if keyword_embedding is None:
            return []

        # 데이터베이스에서 문제 임베딩 가져오기
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT problem_id, title, description, embedding FROM problems")
        problems = cursor.fetchall()
        conn.close()

        # 코사인 유사도를 계산하여 가장 유사한 문제 추천
        recommendations = []
        for problem_id, title, description, embedding_str in problems:
            problem_embedding = np.fromstring(embedding_str, sep=',')
            similarity = cosine_similarity([keyword_embedding], [problem_embedding])[0][0]
            recommendations.append((problem_id, title, description, similarity))

        # 유사도가 높은 순으로 정렬 후 상위 추천
        recommendations = sorted(recommendations, key=lambda x: x[3], reverse=True)[:top_k]
        return [(r[0], r[1], r[2]) for r in recommendations]

if __name__ == "__main__":
    # 명령줄 인자로 키워드를 입력받기 위한 설정
    parser = argparse.ArgumentParser(description="Recommend problems based on a keyword.")
    parser.add_argument("keyword", type=str, help="Keyword for problem recommendation")
    args = parser.parse_args()

    # RecommenderSystem 인스턴스 생성 및 추천 실행
    recommender = RecommenderSystem(model, data, topic_to_label, label_to_topic)
    recommender.generate_node_embeddings()

    # 사용자 입력 키워드 기반 추천
    user_keyword = args.keyword
    recommendations = recommender.recommend_problems_by_keyword(user_keyword, top_k=5)
    
    print("Recommended problems based on the keyword:", user_keyword)
    if recommendations:
        for problem in recommendations:
            print(f"Problem ID: {problem[0]}, Title: {problem[1]}, Description: {problem[2]}")
    else:
        print("No recommendations found for the entered keyword.")
