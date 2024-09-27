from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def chatgpt_crawl(question):
    # Chrome 웹드라이버 설정
    driver = webdriver.Chrome()
    
    try:
        # OpenAI ChatGPT 페이지로 이동
        driver.get("https://chat.openai.com/")
        
        # 로그인 (이 부분은 수동으로 해야 할 수 있습니다)
        # 여기에 로그인 로직을 추가하세요
        
        # 채팅 입력 필드 찾기
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[data-id='root']"))
        )
        
        # 질문 입력 및 전송
        input_box.send_keys(question)
        input_box.send_keys(Keys.RETURN)
        
        # 응답 대기
        time.sleep(10)  # 응답 시간에 따라 조정 필요
        
        # 응답 추출
        response = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.markdown"))
        )
        
        return response.text
    
    finally:
        driver.quit()

# 사용 예시
question = "인공지능이란 무엇인가요?"
answer = chatgpt_crawl(question)
print(f"질문: {question}")
print(f"답변: {answer}")
