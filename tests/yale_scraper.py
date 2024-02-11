from urllib.request import urlopen
from bs4 import BeautifulSoup
import itertools
import os
import shutil
directory = r'C:\Users\samsh\Videos\ta.ai\src\courses\Yale\GG140'
texts = []
for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if len(texts) == len(os.listdir(directory)):
            break
        print(path)
        with open(path,errors="ignore") as f:
            soup = BeautifulSoup(f, features="html.parser")
            print("bab")
        # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

        # get text
            text = soup.get_text()

        # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
            #print(next(itertools.islice(chunks, 5, None)))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            texts.append(text)
            #f.close()


for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

print("hello")
print(len(texts))
for i in range(len(texts)):
    print("hello")
    name_builder = "Lecture" + str((i + 1))
    path = directory + "/" + name_builder
    print(path)
    with open(path, "w") as f:
        f.write(texts[i])    
# c = 0
# d = -1
# for filename in os.listdir(directory):
#     c += 1
#     d += 1
#     fn = os.path.join(directory, filename)
#     with open(fn) as f:
#     # checking if it is a file
       
#         new_name = "Lecture" + str(c)
#         new_file_name = fn.replace(filename, new_name)
#         os.rename(fn, new_file_name)

#         f.write(texts[d])
#         f.close()

        
       
        
        
       

    #print(text)