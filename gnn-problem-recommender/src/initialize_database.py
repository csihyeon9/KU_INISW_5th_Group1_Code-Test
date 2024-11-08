import sqlite3
import numpy as np

# 데이터베이스 연결 및 테이블 생성
db_path = "data/recommendation_problems.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 기존 테이블 삭제 및 새 테이블 생성
cursor.execute("DROP TABLE IF EXISTS problems")
cursor.execute("""
    CREATE TABLE problems (
        problem_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        topic TEXT NOT NULL,
        embedding TEXT  -- GNN 임베딩 값을 문자열로 저장
    )
""")

# 임의의 문제 데이터와 임베딩 추가
problems = [
    (1, "Understanding Finance", "Basics of finance concepts", "Finance"),
    (2, "Loan Types and Options", "Different types of loans available", "Loan"),
    (3, "Cryptocurrency Basics", "Introduction to cryptocurrency", "Cryptocurrency"),
    (4, "Portfolio Diversification", "Importance of diversification in a portfolio", "Portfolio"),
    (5, "Economic Growth Factors", "Understanding what drives economic growth", "Economic Growth"),
]

# 각 문제에 대한 임베딩 값 생성 및 데이터베이스에 추가
for problem_id, title, description, topic in problems:
    embedding_vector = np.random.rand(64)  # 예제: 64차원 랜덤 임베딩 벡터
    embedding_str = ','.join(map(str, embedding_vector))
    cursor.execute("INSERT INTO problems (problem_id, title, description, topic, embedding) VALUES (?, ?, ?, ?, ?)",
                   (problem_id, title, description, topic, embedding_str))

conn.commit()
conn.close()
print("Database initialized with embeddings.")
