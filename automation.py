from selenium import webdriver

browser = webdriver.Firefox(executable_path='./geckodriver.exe')

browser.maximize_window()

browser.get('https://www.seleniumeasy.com/test/basic-first-form-demo.html')
show_msg_button = browser.find_element_by_class_name('btn-default')

assert "Show Message" in browser.page_source

user_message = browser.find_element_by_id('user-message')
user_message.clear()
user_message.send_keys('COOOOOOOOOOOL')

browser.close()