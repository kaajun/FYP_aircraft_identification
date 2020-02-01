from os import walk
import os
import sys
import pandas as pd
from tqdm import tqdm

# try:
#         link = sys.argv[1]
# except IndexError:
#         scriptName = sys.argv[0]
#         print ("Usage: python " + scriptName + " linkOfVideo")
#         exit()
#Change this path with yours.
#Also make sure that youtube-dl and ffmpeg installed.
#Previous versions of youtube-dl can be slow for downloading audio. Make sure you have downloaded the latest version from webpage.
#https://github.com/rg3/youtube-dl

# existing_files = os.listdir("./youtube-dl/Output")
# existing_files = [x[:-4] for x in existing_files]
# master_youtube_akino = pd.read_csv('./master_youtube_akino33.csv')
# master_youtube_akino = master_youtube_akino[~(master_youtube_akino['title'].isin(existing_files))]
# master_youtube_akino = master_youtube_akino.dropna()

cur = pd.read_csv("master_youtube_akino33.csv")
prev = pd.read_csv("UTILITIES\master_youtube_akino33_prev.csv")
li_cur = cur['urllink'].tolist()
li_cur = [x.split("=")[1] for x in li_cur]
li_prev = prev['urllink'].tolist()
li_prev = [x.split("=")[1] for x in li_prev]
cur['code'] = li_cur
dff = list(set(li_cur) - set(li_prev))
master_youtube_akino = cur[cur['code'].isin(dff)]

print("Total to be download audio file",len(master_youtube_akino))

mypath = "D:\FYP\youtube-dl"
os.chdir(mypath)

links = master_youtube_akino['urllink'].tolist()
titles = master_youtube_akino['title'].tolist()
ii = 0
for link in links:
    try:
        os.system('youtube-dl -o "./Result/%(title)s.%(ext)s" --extract-audio ' + link)
    except:
        break

f = []
for (dirpath, dirnames, filenames) in walk(os.path.join(mypath,'Result')):
    f.extend(filenames)
    break

j=0
for i in tqdm(range(0, len(f))):
        if ".m4a" in f[i] or '.opus' in f[i]:
                vidName = f[i]
                print(vidName)
                try:
                    cmdstr = "ffmpeg -i \"" "./Result/"+ vidName + "\" -f wav -flags bitexact \"" + vidName[:-5] + ".wav"  + "\""
                    #print(cmdstr)
                    os.system(cmdstr)
                    j+=1
                    os.remove("./Result/"+vidName) #Will remove original opus file. Comment it if you want to keep that file.
                    #os.replace(os.path.join(mypath,vidName[:-5]+".wav"),mypath+"\Ouput\{}.wav".format(vidName[:-5]))
                except:
                    break

print("Number elapse ",j)