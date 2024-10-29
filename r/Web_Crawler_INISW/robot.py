import requests

def fetch_robots_txt(url):
    # robots.txt의 URL을 구성
    if not url.endswith('/'):
        url += '/'
    robots_url = url + "robots.txt"
    
    try:
        response = requests.get(robots_url)
        
        # 응답이 성공적일 경우 내용 반환
        if response.status_code == 200:
            print("robots.txt 파일을 성공적으로 가져왔습니다.")
            return response.text
        elif response.status_code == 404:
            print("robots.txt 파일을 찾을 수 없습니다.")
        else:
            print(f"요청 실패: 상태 코드 {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"오류 발생: {e}")
    return None

# 예시
url = "https://www.example.com"
robots_txt_content = fetch_robots_txt(url)

if robots_txt_content:
    print("\nrobots.txt 내용:")
    print(robots_txt_content)
