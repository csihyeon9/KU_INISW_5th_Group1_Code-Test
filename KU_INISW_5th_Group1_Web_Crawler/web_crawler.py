from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from xml.etree.ElementTree import Element, SubElement, ElementTree
import time

# WebDriver 설정
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# 목표 웹사이트로 이동
url = "https://csihyeon.tistory.com/"  # 스크래핑할 웹사이트 URL
driver.get(url)
time.sleep(3)  # 페이지 로딩 대기 시간 (필요에 따라 조정)

# <section class="h-entry"> 태그를 스크래핑
entries = driver.find_elements(By.CSS_SELECTOR, 'section.h-entry')

# XML 파일 구조 생성
root = Element('entries')  # XML 루트 노드 생성

for idx, entry in enumerate(entries, start=1):
    # 각 게시물을 XML에 추가
    entry_element = SubElement(root, 'entry', id=str(idx))
    content = entry.text.strip()
    
    content_element = SubElement(entry_element, 'content')
    content_element.text = content

# XML 파일로 저장
tree = ElementTree(root)
xml_path = 'scraped_entries.xml'  # 저장할 XML 파일 이름
tree.write(xml_path, encoding='utf-8', xml_declaration=True)

print(f"Scraped data has been saved to {xml_path}")

# 드라이버 종료
driver.quit()
