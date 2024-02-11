from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist, YouTube
from deepmultilingualpunctuation import PunctuationModel
import os
from os.path import exists

def scrape_yt_list(playlist, course_name, school_name: str):
    
# https://www.youtube.com/playlist?list=PLoROMvodv4rPzLcXBhbCFt8ahPrQGFSmN (CS 105)
def scrape_yt_playlist(playlist: str, course_name : str, school_name: str):

    directory = school_name
    parent_dir = r"courses/"
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path) # creates dir for course
    except FileExistsError:
        pass
        
    directory = course_name
    parent_dir = r"courses/" + school_name
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path) # creates dir for course
    except FileExistsError:
        pass


    model = PunctuationModel()
    playlist = Playlist(playlist) # CS105 Lectures
    #num_vids = len(playlist.video_urls) # Number of videos in the playlist
    video_ids = playlist.trimmed(69) # video ids
    count_track = 0
    for id in video_ids:
        count_track += 1
        short_id = id[-11:]
        name = "myfile"+str(short_id)+".txt"
        file_exists = exists(path+"/"+name)
        if file_exists:
            continue
        #print(id)
        
        try:
            srt = YouTubeTranscriptApi.get_transcript(short_id, languages=['en', 'en-US']) # gets transcript of vid in playlist
        except:
            srt = YouTubeTranscriptApi.get_transcript(short_id, languages=['en-US']) # gets transcript of vid in playlist
        

        #print(srt)
        temp_text = ""
        for info in srt:
           # l[0:2] = [' '.join(l[0:2])]
           
            temp_text += "{}\n".format(info["text"])
            #print(temp_text)
        # create folder
        
        #f = open(path+name, "x")

        skeleton = "http://www.youtube.com/watch?v="
        yt = YouTube(skeleton + id)
        title = yt.title
        words = title.split(" ")
        print(words)
        lecture = ""
        print(yt.title)
        lec_number = False
        for i in range(len(words)):
            if 'lecture' in words[i].lower() or 'lec' in words[i].lower():
                lecture = 'Lecture' + words[i + 1]
                lec_number = True
        
        if lec_number == False:
            lecture = 'Lecture' + str(count_track)
                

           

        print(lecture)
        lecture = lecture.replace(':','')
        print(lecture)

        
        with open(r'C:/Users/samsh/Videos/ta.ai/src/courses'+"/"+school_name+"/"+course_name+"/"+lecture, "w") as text_file:
            text_file.write(model.restore_punctuation(temp_text)) # adds punctuation to text
            text_file.close()


        #print(model.restore_punctuation(temp_text))
        

#scrape_yt("https://www.youtube.com/playlist?list=PLoROMvodv4rPzLcXBhbCFt8ahPrQGFSmN", "CS105", "Stanford") # CS105 Stanford Playlist
# scrape_yt("https://www.youtube.com/playlist?list=PLUl4u3cNGP619EG1wp0kT-7rDE_Az5TNd", "MIT 6.0002", "MIT")
# scrape_yt("https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4", "CS224N") # CS224 Stanford Playlist
#scrape_yt("https://www.youtube.com/playlist?list=PLOspHqNVtKADfxkuDuHduUkDExBpEt3DF", "AI-Essentials", "IBM")
# add change:
# 1. create folder for <university_name>, list respective course under it
# 2. add keyword parameter to function for course content, quizzes, or tests. 
        # Under each course, data is added in different bins
        # if client wants a sample question, TA will pull from bin (using context embedding) and reword question into newer question using GPT
        # GPT waits for answer, if answer is incorrect --> store into memory as weak point
# 3. (IMPORTANT) Re-do this script to make the filenames the # of the lecture.
            # That way, the user can query "Lecture: 1: XXX"
            # in essence, it allows the user to specify exactly which lecture they need help with and reduces search time via cosine