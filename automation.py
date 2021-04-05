from selenium import webdriver

browser = webdriver.Firefox(executable_path='./geckodriver.exe')

browser.maximize_window()

browser.get('https://www.seleniumeasy.com/test/basic-first-form-demo.html')

assert "show message" in browser.page_source