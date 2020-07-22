from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time

url = "https://www.sephora.com/profile/MyAccount/Orders"
def getHtml(url, loadmore = False, waittime = 5):
    browser = webdriver.Chrome('chromedriver')
    browser.get(url)
    time.sleep(waittime)
    browser.find_element_by_id('signin_username').send_keys('yangdajiemaimaimai@163.com')
    time.sleep(waittime)
    browser.find_element_by_id('signin_password').send_keys('19880814lyy')
    time.sleep(waittime)
    browser.find_element_by_xpath('//*[@id="modalDialog"]/div[1]/div/form/button').click()
    time.sleep(waittime)
    view_details_list = browser.find_elements_by_class_name("css-14vlm64 ")
    html = []
    for i in range(len(view_details_list)):
        browser.find_element_by_id('view Details')[i].click()
        html.append(browser.page_source)
        print(i)
    browser.quit()
    return html


page = getHtml(url, False, 1)
print(page)