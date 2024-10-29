# main_scraper.py

from xml.etree.ElementTree import Element, SubElement, ElementTree
from tag_detec import get_entries_from_url
from robots_fetcher import get_allow_directives
import requests
from bs4 import BeautifulSoup

# 웹사이트 리스트를 파일에서 읽기
with open("weblist.txt", "r") as file:
    urls = [line.strip() for line in file if line.strip()]  # 빈 줄 제외

# XML 파일 구조 생성
root = Element('scraped_data')

for url in urls:
    try:
        # Allow 지시문 가져오기
        allow_directives = get_allow_directives(url)

        # <section class="h-entry"> 태그 내용 가져오기
        entries_text = get_entries_from_url(url)

        # URL에 GET 요청하여 제목 추출
        response = requests.get(url)
        response.raise_for_status()  # 상태 코드가 정상인지 확인
        
        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 제목 추출
        title = soup.title.string if soup.title else "No title found"
        
        # URL 정보를 XML에 추가
        url_element = SubElement(root, 'url', href=url)
        title_element = SubElement(url_element, 'title')
        title_element.text = title

        # h-entry 태그를 XML 구조에 추가
        entries_element = SubElement(url_element, 'entries')
        for idx, content in enumerate(entries_text, start=1):
            entry_element = SubElement(entries_element, 'entry', id=str(idx))
            content_element = SubElement(entry_element, 'content')
            content_element.text = content

        # Allow 지시문을 XML에 추가
        allow_element = SubElement(url_element, 'allow_directives')
        if allow_directives:
            for directive in allow_directives:
                directive_element = SubElement(allow_element, 'directive')
                directive_element.text = directive
        else:
            allow_element.text = "No Allow directives found."
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")

# XML 파일로 저장
xml_path = 'scraped_entries_and_allow_directives.xml'
tree = ElementTree(root)
tree.write(xml_path, encoding='utf-8', xml_declaration=True)

print(f"Scraped data has been saved to {xml_path}")
