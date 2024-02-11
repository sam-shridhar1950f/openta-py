from pytube import YouTube
skeleton = "http://www.youtube.com/watch?v="
id = "6OxyB9yuSzo"
yt = YouTube(skeleton + id)
title = yt.title

words = title.split(" ")

print(words)

lecture = ""
for i in range(len(words)):
    if 'lecture' in words[i].lower():
        lecture = words[i] + words[i + 1]
lecture = lecture.replace(':','')

print(lecture)

import os

# folder path
dir_path = r'C:\Users\samsh\Videos\ta.ai\src\courses\Stanford\CS105'
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)