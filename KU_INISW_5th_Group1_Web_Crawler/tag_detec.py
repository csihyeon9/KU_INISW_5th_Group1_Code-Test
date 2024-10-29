# tag_detec.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_entries_from_url(url):
    # WebDriver 설정
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # 웹 페이지 열기
    driver.get(url)
    time.sleep(3)  # 페이지 로딩 대기 (필요에 따라 조정)

    # <section class="h-entry"> 태그를 찾고 텍스트 추출
    entries = driver.find_elements(By.CSS_SELECTOR, 'section.h-entry')
    entries_text = [entry.text.strip() for entry in entries]

    # 드라이버 종료
    driver.quit()

    return entries_text
