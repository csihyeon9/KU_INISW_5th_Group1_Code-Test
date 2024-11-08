import sqlite3

def display_db_contents(db_path="data/recommendation_problems.db"):
    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 테이블 내용 조회 및 출력
    cursor.execute("SELECT * FROM problems")
    rows = cursor.fetchall()
    
    # 출력
    print("ID\tTitle\t\t\tDescription\t\t\tTopic")
    print("="*80)
    for row in rows:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")
    
    # 연결 닫기
    conn.close()

# 호출하여 데이터베이스 내용 출력
display_db_contents()
