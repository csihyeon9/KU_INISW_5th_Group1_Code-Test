import requests

def is_valid_robots_txt(content):
    # robots.txt의 주요 키워드가 포함되어 있는지 확인
    required_keywords = ["User-agent", "Disallow"]
    return any(keyword in content for keyword in required_keywords)

def get_robots_txt_if_valid(url):
    # URL에 "/"가 없으면 추가
    if not url.endswith('/'):
        url += '/'
    robots_url = url + "robots.txt"
    
    try:
        response = requests.get(robots_url)
        
        # 파일이 있는 경우 (상태 코드 200)
        if response.status_code == 200:
            # 파일이 robots.txt 형식에 맞는지 확인
            if is_valid_robots_txt(response.text):
                print("robots.txt 파일이 맞습니다. 내용을 가져옵니다.")
                return response.text
            else:
                print("파일은 존재하지만 robots.txt 형식이 아닙니다.")
                return None
        elif response.status_code == 404:
            print("robots.txt 파일이 없습니다.")
            return None
        else:
            print(f"요청 실패: 상태 코드 {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"오류 발생: {e}")
        return None

# 예시 URL
url = "https://www.example.com"
robots_txt_content = get_robots_txt_if_valid(url)

if robots_txt_content:
    print("\nrobots.txt 내용:")
    print(robots_txt_content)
else:
    print("유효한 robots.txt 파일이 없거나 가져오는 데 실패했습니다.")
