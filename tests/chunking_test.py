with open(r"C:\Users\samsh\Videos\ta.ai\src\courses\Stanford\CS105\Lecture1.1") as f:
    content = f.read()
    num_words = len(content.split())
    print(num_words)
    span = 300
    words = content.split()
   
    l = [" ".join(words[i:i+span]) for i in range(0, num_words, span)]
    print(len(l[0].split()))
    #print(l)