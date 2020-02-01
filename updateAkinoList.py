import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from shutil import copyfile,move
import shutil

"WARNING, BEFORE YOU RUN THE FILE PLEASE MAKE A COPY OF THE ORIGINAL FILE TO THE UTILITIES"

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
browser = webdriver.Chrome(options=options)
url2 = "https://www.youtube.com/user/akino33/videos"
browser.get(url2)
time.sleep(2)

try:
    master_youtube_table = pd.read_csv("master_youtube_akino33.csv")
    MASTER_EXIST = True
    print("Existing length :",len(master_youtube_table))
    numbering = len(os.listdir('UTILITIES'))
    shutil.move('master_youtube_akino33.csv','UTILITIES/master_youtube_akino33_prev{}.csv'.format(numbering+1))
except:
    MASTER_EXIST = False

def scroll_down(master):
    body = browser.find_element_by_tag_name('body')
    if master:
        scrolls = 30
    else:
        scrolls = 250
    for _ in range(scrolls):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)

scroll_down(MASTER_EXIST)

html_string = browser.page_source
soup = BeautifulSoup(html_string, 'html.parser')
browser.quit()

data_rows_yt  = []
links = soup.find_all('a', attrs={'class': 'yt-simple-endpoint style-scope ytd-grid-video-renderer'})
for link in links:
    title = link.text
    splits = title.split(' ')
    if 'Takeoff' in splits:
        ops_type = 'Takeoff'
    else:
        ops_type = 'Landing'
    if 'Airbus' in splits:
        aircraft_type = splits[splits.index('Airbus')+1]
        reg_no = splits[splits.index('Airbus')+2]
    elif 'Boeing' in splits:
        aircraft_type = splits[splits.index('Boeing')+1]
        reg_no = splits[splits.index('Boeing')+2]
    else:
        aircraft_type = ''
        reg_no = ''
    urllink = 'https://www.youtube.com'+link['href']
    data_rows_yt.append({'title':title,'urllink':urllink,'ops_type':ops_type,'aircraft_type':aircraft_type,'reg_no':reg_no})
    
youtube_data = pd.DataFrame(data_rows_yt)

try: 
    master_youtube_table = pd.concat([youtube_data,master_youtube_table],axis=0,join='outer')
    master_youtube_table = master_youtube_table.drop_duplicates()
    master_youtube_table['code'] = [x.split("=")[1] for x in master_youtube_table['urllink'].tolist()]
    master_youtube_table = master_youtube_table.drop_duplicates('code')
    master_youtube_table = master_youtube_table.drop(columns=['code'])
    master_youtube_table.to_csv('master_youtube_akino33.csv',index=False)
except:
    youtube_data.to_csv('master_youtube_akino33.csv',index=False)
    master_youtube_table = youtube_data

print("Length of Master Youtube Table is {}".format(len(master_youtube_table)))

