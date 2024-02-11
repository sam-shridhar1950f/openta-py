import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP
import pinecone
import re
from transformers import pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from .summary import summarize
import time
# 33290b8c-00ff-432a-9042-ba9bb087eebb
# Note To Self: 8000 tokens is about 6500 words, so use as modulo for calculating chunks
# openai.api_key = "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" # "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" # openai api key
# pinecone.init(api_key="190dabf8-0c0b-4690-8733-0be7fcecdb34") # init pinecone db

class Course:
    
    
    def __init__(self, course_title, school, api_key, path="", pinecone_index="", messages=[]):
        self.course_title = course_title # Title of course
        self.school = school # Institution that offers the course
        self.path = "courses/"+school+"/"+course_title
        self.api_key = api_key
        self.pinecone_index = school.lower() + "-" + course_title.lower()
        self.messages = [{
            "role": "system",
            "content": f"If someone asks who you are, respond that you are a teaching assistant for the course {course_title} at {school}"
            }]
        if self.pinecone_index in pinecone.list_indexes():
            pass
        else:
            pinecone.create_index(self.pinecone_index, dimension=1536, metric="cosine") # name of index needs to be lowercase
    def __str__(self):
        return f"{self.course_title}{self.school})" 

    def embed_data(self):
        for filename in os.listdir(self.path): # iterating over files
            fp = self.path + "/" + filename
            stripped_fp = self.path + "/"
            print(filename)
            shortened_responses = []
            with open(fp, encoding="utf8", errors="ignore") as f: # open(stripped_fp+"myfilexs_jmQt5D0s.txt") as f2:
                content = f.read()
                #content =  content.decode('utf-8','ignore').encode("utf-8")
                # unrelated_content = f2.read()
                # query = "what is virtual memory?"
                # sample_content = "in what context should I use serif fonts in my webpages" # "there that seems pretty seraphi, and if you look at the news stories here, you can see that they have serifs as well, and there is some sans serif. over on the right side, that whole sidebar is largely sans serif, and up at the top, right underneath the name, the new york times. if you look at the world, us politics, new york business- that's all sans serif as well, and so we scroll a little bit further down the website. you'll notice that this webpage is largely serifs, and the news articles themselves are serifs, and so that kind of makes sense if you think about, like, what's new york times probably trying to portray here. they're trying to portray that they are formal and authoritative. what about the washington post? so the washington post also largely uh has a lot of serifs here."
                halved = False
                try:
                    response = openai.Embedding.create( # related text (large)
                    model= "text-embedding-ada-002",
                    input=[content])
                except:
                    print("error")
                    halved = True
                    num_words = len(content.split())
                    # divisor = (num_words // 6000) + 1 # do this logic tomorrow
                    span = 3000
                    words = content.split()
                    l = [" ".join(words[i:i+span]) for i in range(0, num_words, span)]

                    # shortened_content_half1 = ' '.join(content.split()[:(num_words//2)])
                    # shortened_content_half2 = ' '.join(content.split()[(num_words//2):])
                   # print(len(shortened_content.split()))
                    print("before")
                    
                    for text in l:
                        response_section = openai.Embedding.create( # shortened
                        model= "text-embedding-ada-002",
                        input=[text])
                        shortened_responses.append(response_section)
                        print("after")

                        

                    # response_half1 = openai.Embedding.create( # shortened
                    # model= "text-embedding-ada-002",
                    # input=[shortened_content_half1])
                    # print("after")
                    # response_half2 = openai.Embedding.create( # shortened
                    # model= "text-embedding-ada-002",
                    # input=[shortened_content_half2])



                # response2 = openai.Embedding.create( # query
    	        # model= "text-embedding-ada-002",
    	        # input=[query])

                # response3 = openai.Embedding.create( # related text (chunk)
    	        # model= "text-embedding-ada-002",
    	        # input=[sample_content])

                # response4 = openai.Embedding.create( # related text (chunk)
    	        # model= "text-embedding-ada-002",
    	        # input=[unrelated_content])
                index = pinecone.Index(self.pinecone_index)
                # print(filename[7:])
                c = 0
                if halved:
                    for resp in shortened_responses:
                        c += 1

                        resp_data = resp["data"][0]["embedding"]
                    
                        # response_data = response_data_half1 + response_data_half2
                        index.upsert([(filename + "-" + str(c), resp_data, {"lecture": filename[7:]})])
                        #index.upsert([(filename + "-" + "2", response_data_half2, {"lecture": filename[7:]})])
                    
                else:
                    response_data = response["data"][0]["embedding"]
                    index = pinecone.Index(self.pinecone_index)
                    index.upsert([(filename, response_data, {"lecture": filename[7:]})])
       


                #response_data = response["data"][0]["embedding"]
                # response2_data = response2["data"][0]["embedding"]
                # response3_data = response3["data"][0]["embedding"]
                # response4_data = response4["data"][0]["embedding"]
                # index = pinecone.Index(self.pinecone_index)
                # index.upsert([(filename, response_data)])
                # result = 1 - distance.cosine(response2_data, response4_data) # high num means more similar
            # checking if it is a file
                
            # Embed a line of text
                
	    
	# Extract the AI output embedding as a list of floats
                
                #print(response)
                #return result


    # Requires prompt asking for a sample quiz question
    # Returns string representing a sample question based on context embedding gpt
    def check_record_stats(self):
        index = pinecone.Index(self.pinecone_index)
        return index.describe_index_stats()
    
    def query(self, prompt, model="gpt-4"): # try gpt-4-32k if this runs out of tokens
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a brief response for the query. If the context seems irrelevant to the query, ignore the context proceed to respond normally."
        # counter = -1
        # meta_num = 0
        # lec_specific = False
        # split_prompt = prompt.split()
        # if "lecture" or "lec" in prompt.split():

        #     lec_specific = True
        #     for p in split_prompt:
        #         counter += 1
        #         print(counter)
        #         print(prompt.split())
        #         if p == "lecture" or p == "lec":
        #             meta_num = str(split_prompt[counter + 1].strip())
        #             print(meta_num)
        #             meta_num = re.sub("[^0-9]", "", meta_num)
        #             print("hai")
        #             print(meta_num)
        #             break

        

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        # if lec_specific:
        #     print("yuh")
        #     possible_contexts = index.query(
        #     vector=query_response_data,
        #     top_k=3,
        #     filter={
        #     "lecture": {"$eq": meta_num},
        #     },
        #     include_values=False
        #     )
        



        possible_contexts = index.query(
            vector=query_response_data,
            top_k=3,
            include_values=False
            )

        context = possible_contexts["matches"][0]["id"]
        context_copy = possible_contexts["matches"][0]["id"]
        #print(possible_contexts)
        if "-" in context:
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 
            
     

        # print(possible_contexts["matches"])
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            if "-" in context_copy:
                span = 3000
                content = f.read()
                num_words = len(content.split())
                content = content.split()

                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                content = l[ind - 1]
            else:

                content = f.read()
            build_context = header + "\n" + content + "\n" + prompt
            self.messages.append(dict(role = "user", content = build_context))
            try:
                #print(ind)
               # print(len(self.messages[1]['content'].split()))
                response = openai.ChatCompletion.create( 
                    model=model,
                    messages = self.messages,
                    max_tokens=1000,
                    temperature=0.5,
                    ) 
            except:
                #print("checking")

                print(self.messages)
                #print(len(self.messages[1]['content'].split()))


                # time.sleep() does not work. Instead, we have three options:
                    # chunk data into smaller pieces (~3000-4000 or less)
                    # if except is triggered, clear everything before current build_context and continue
                    # store message history before current build context elsewhere temporarily, run response, and add message history back when build_context is removed
                        # This is the best idea. Will work for follow up too bc follow up doesn't require large build contexts

                    # ^ This did not work. Next solution:
                    # 1. Chunk data into smaller pieces (~2000) <--- Seems like better solution until I can request a rate limit increase
                    # 2. Summarize build contexts with classifier
                #print(len(self.messages[1]['content'].split()))
                time.sleep(6)
                temp_msgs = [self.messages[0], self.messages[-1]]
                response = openai.ChatCompletion.create( 
                    model=model,
                    messages = temp_msgs,
                    max_tokens=1000,
                    temperature=0.5,
                    ) 

            self.messages[-1] = (dict(role = "user", content = prompt))
            message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
            return message
    


        message = response.choices[0].text.strip()
        return message
    
    def query_quiz(self, prompt, model="gpt-4-turbo-preview"):
        # test case: Give me a five question quiz about lecture 12.
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        #print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a quiz according to the user's specifications (if there are no specifications, use general rules). Use normal text, avoid markdown and ### signs:"

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        
        possible_contexts = index.query(
        vector=query_response_data,
        filter={
        "lecture": {"$eq": lec_num},
        },
        top_k=1,
        include_values=False
        )

        #print(possible_contexts)
        multiple_matches = False
        context = possible_contexts["matches"][0]["id"]
        if "-" in context:
            multiple_matches = True
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 

        #print(possible_contexts["matches"])
        #print(context)
        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            summaries = []
            #classifier = pipeline("summarization")
            if multiple_matches:
                span = 3000
                content = f.read()
                print(content)
                num_words = len(content.split())
                content = content.split()
                
                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                for section in l:
                    tokens = 3000 // len(l)
                    """
                    # TRY USING HUGGINFACE SUMMARIZER TO SUMMARIZE TEXT TO AVOID RATE LIMIT ERRORS
                    # MAKE TEST FIRST TO EVAL HOW LONG SUMMARIES ARE (SHOULD BE SHORT)
                    """
                    #print(len(section.split()))
                    response_summary = summarize(section, 0.03)
                    #print(response_summary)
                    # summary_prompt = section + "\n" + "Summarize this section of text in detail." 
                    # response_summary = openai.ChatCompletion.create( 
                    # model=model,
                    # messages=[{"role": "user","content":section}],
                    # max_tokens=tokens,# try raising this number
                    # temperature=0.5,
                    # ) 
                    summaries.append(response_summary) # ['choices'][0]['message']['content'])
                final_summary = "\n".join(summaries)
                header = f"Using this summary below as context, provide a quiz according to the user's specifications (if there are no specifications, use general rules):"
                build_context = header + "\n" + final_summary + "\n" + prompt
                #print(final_summary)
                
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(0.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content']
                return message

                

            else:
                context = f.read()
                build_context = header + "\n" + context + "\n" + prompt
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(0.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        )

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
                return message


    def summarize_query(self, prompt, model="gpt-4"):
        # test case: Give me a five question quiz about lecture 12.
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = "Summarize the lecture transcript below in 4-5 sentences."

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        
        possible_contexts = index.query(
        vector=query_response_data,
        filter={
        "lecture": {"$eq": lec_num},
        },
        top_k=1,
        include_values=False
        )

        # print(possible_contexts)
        multiple_matches = False
        context = possible_contexts["matches"][0]["id"]
        if "-" in context:
            multiple_matches = True
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 

       # print(possible_contexts["matches"])
       # print(context)
        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            summaries = []
            #classifier = pipeline("summarization")
            if multiple_matches:
                span = 3000
                content = f.read()
                num_words = len(content.split())
                content = content.split()
                
                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                for section in l:
                    tokens = 3000 // len(l)
                    """
                    # TRY USING HUGGINFACE SUMMARIZER TO SUMMARIZE TEXT TO AVOID RATE LIMIT ERRORS
                    # MAKE TEST FIRST TO EVAL HOW LONG SUMMARIES ARE (SHOULD BE SHORT)
                    """
                    #print(len(section.split()))
                    print("cool")
                    response_summary = summarize(section, 0.03)
                    #print(response_summary)
                    # summary_prompt = section + "\n" + "Summarize this section of text in detail." 
                    # response_summary = openai.ChatCompletion.create( 
                    # model=model,
                    # messages=[{"role": "user","content":section}],
                    # max_tokens=tokens,# try raising this number
                    # temperature=0.5,
                    # ) 
                    summaries.append(response_summary) # ['choices'][0]['message']['content'])
                final_summary = "\n".join(summaries)
                header = f"Make this summary of {lec_num} below more substantial, readable, formal, and academic in nature. Make sure to introduce your response as a summary of lecture {lec_num}."
                build_context = header + "\n" + final_summary # + "\n"  + prompt
                # print(final_summary)
                
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(1.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content']
                return message

                

            else:
                context = f.read()
                build_context = header + "\n" + context # + "\n" + prompt
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(0.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        )

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
                return message




    def lecture_notes_query(self, prompt, model="gpt-4-turbo-preview"):
        # test case: Give me a five question quiz about lecture 12.
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = f"Using this transcript of lecture {lec_num} as context, write a summarizing lecture outline according to the user's specifications. Make sure to introduce your response as an outline for lecture {lec_num}."

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        
        possible_contexts = index.query(
        vector=query_response_data,
        filter={
        "lecture": {"$eq": lec_num},
        },
        top_k=1,
        include_values=False
        )

        # print(possible_contexts)
        multiple_matches = False
        context = possible_contexts["matches"][0]["id"]
        if "-" in context:
            multiple_matches = True
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 

       # print(possible_contexts["matches"])
       # print(context)
        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            summaries = []
            #classifier = pipeline("summarization")
            if multiple_matches:
                span = 3000
                content = f.read()
                num_words = len(content.split())
                content = content.split()
                
                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                for section in l:
                    tokens = 3000 // len(l)
                    """
                    # TRY USING HUGGINFACE SUMMARIZER TO SUMMARIZE TEXT TO AVOID RATE LIMIT ERRORS
                    # MAKE TEST FIRST TO EVAL HOW LONG SUMMARIES ARE (SHOULD BE SHORT)
                    """
                    #print(len(section.split()))
                   # print("cool")
                    response_summary = summarize(section, 0.03)
                    #print(response_summary)
                    # summary_prompt = section + "\n" + "Summarize this section of text in detail." 
                    # response_summary = openai.ChatCompletion.create( 
                    # model=model,
                    # messages=[{"role": "user","content":section}],
                    # max_tokens=tokens,# try raising this number
                    # temperature=0.5,
                    # ) 
                    summaries.append(response_summary) # ['choices'][0]['message']['content'])
                final_summary = "\n".join(summaries)
                header = f"Make this text about {lec_num} below more substantial, readable, formal, and academic in nature, and then convert it into a summarizing lecture outline according to the user's specifications. Make sure to introduce your response as an outline for lecture {lec_num}."
                build_context = header + "\n" + final_summary  + "\n"  + prompt
                # print(final_summary)
                
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(1.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content']
                return message

                

            else:
                context = f.read()
                build_context = header + "\n" + context + "\n" + prompt
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(0.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        )

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
                return message

    def notes_query(self, prompt, model="gpt-4-turbo-preview"):
        # test case: Give me a five question quiz about lecture 12.
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = f"Using this text as context, write specific, detailed notes according to the user's specifications. Make sure to introduce your response as notes for lecture {lec_num}."

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        
        possible_contexts = index.query(
        vector=query_response_data,
        filter={
        "lecture": {"$eq": lec_num},
        },
        top_k=1,
        include_values=False
        )

        # print(possible_contexts)
        multiple_matches = False
        context = possible_contexts["matches"][0]["id"]
        if "-" in context:
            multiple_matches = True
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 

       # print(possible_contexts["matches"])
       # print(context)
        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            summaries = []
            #classifier = pipeline("summarization")
            if multiple_matches:
                span = 3000
                content = f.read()
                num_words = len(content.split())
                content = content.split()
                
                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                for section in l:
                    tokens = 3000 // len(l)
                    """
                    # TRY USING HUGGINFACE SUMMARIZER TO SUMMARIZE TEXT TO AVOID RATE LIMIT ERRORS
                    # MAKE TEST FIRST TO EVAL HOW LONG SUMMARIES ARE (SHOULD BE SHORT)
                    """
                    #print(len(section.split()))
                    #print("cool")
                    response_summary = summarize(section, 0.03)
                    #print(response_summary)
                    # summary_prompt = section + "\n" + "Summarize this section of text in detail." 
                    # response_summary = openai.ChatCompletion.create( 
                    # model=model,
                    # messages=[{"role": "user","content":section}],
                    # max_tokens=tokens,# try raising this number
                    # temperature=0.5,
                    # ) 
                    summaries.append(response_summary) # ['choices'][0]['message']['content'])
                final_summary = "\n".join(summaries)
                header = f"Make this text below more substantial, readable, formal, and academic in nature, and then convert it into specific, detailed notes about the content according to the user's specifications. Make sure to introduce your response as notes for lecture {lec_num}."
                build_context = header + "\n" + final_summary  + "\n"  + prompt
                # print(final_summary)
                
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(1.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content']
                return message

                

            else:
                context = f.read()
                build_context = header + "\n" + context + "\n" + prompt
                self.messages.append(dict(role = "user", content = build_context))
                try:
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 
                except:
                    time.sleep(0.5) # this could minimum be 0.006
                    response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        )

                self.messages[-1] = (dict(role = "user", content = prompt))
                message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
                return message

    def prompt_selector(self):
        pass

    def follow_up_query(self, prompt, model="gpt-4"):
        header = "Allow the user to follow up on the previous query and response: "
        f"Respond to the following follow up question. If it references specific content of {self.school} {self.course_title}, still answer the question directly and to the best of your knowledge."
        build_context = header + "\n" + prompt
        self.messages.append(dict(role = "user", content = build_context))
        response = openai.ChatCompletion.create( 
                model=model,
                messages = self.messages,
                max_tokens=1000,
                temperature=0.5,
                ) 
        self.messages[-1] = (dict(role = "user", content = prompt))
        # replace self.messages[-1] with prompt
        # self.messages[-1] = prompt
        # do this for every query type
        message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
        return message

    # user requests sample exam for lectures [x,y], method returns exam with answers
    def test_query(self, prompt, model="gpt-4-turbo-preview"):
         # test case: Give me a five question quiz about lecture 12.
         # Give me a test for lectures 15-17
        s = prompt.split(" ")
       
        s = s[-1].split("-")
        print(s)
        n1, n2 = s[0], s[1]
        print(n1, n2)

        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = f"Using this text as context, write a challenging exam according to the user's specifications. Make sure to introduce your response as an exam for lectures {n1}-{n2}."

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        list_of_possible_contexts = []
        
        grand_summaries = []
        for i in range(int(n1),int(n2)+1):
            

            possible_contexts = index.query( 
            vector=query_response_data,
            filter={
            "lecture": {"$eq": str(i)},
            },
            top_k=1,
            include_values=False
            )

            # print(possible_contexts)
            multiple_matches = False
            context = possible_contexts["matches"][0]["id"]
            if "-" in context:
                multiple_matches = True
                context_split = context.split("-")
                context = context_split[0]
                ind = int(context_split[1]) 

        # print(possible_contexts["matches"])
        # print(context)
            # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context
            file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
            with open(file_path,encoding="utf8", errors="ignore") as f:
                summaries = []
                #classifier = pipeline("summarization")
                if multiple_matches:
                    span = 3000
                    content = f.read()
                    print(content)
                    num_words = len(content.split())
                    content = content.split()
                    
                    l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                    for section in l:
                        tokens = 3000 // len(l)
                        """
                        # TRY USING HUGGINFACE SUMMARIZER TO SUMMARIZE TEXT TO AVOID RATE LIMIT ERRORS
                        # MAKE TEST FIRST TO EVAL HOW LONG SUMMARIES ARE (SHOULD BE SHORT)
                        """
                        #print(len(section.split()))
                        #print("cool")
                        response_summary = summarize(section, 0.03)
                        #print(response_summary)
                        # summary_prompt = section + "\n" + "Summarize this section of text in detail." 
                        # response_summary = openai.ChatCompletion.create( 
                        # model=model,
                        # messages=[{"role": "user","content":section}],
                        # max_tokens=tokens,# try raising this number
                        # temperature=0.5,
                        # ) 
                        summaries.append(response_summary) # ['choices'][0]['message']['content'])
                else:
                    # response_summary = summarize(context, 0.03)
                    response_summary = context
                    print(context)
                    print("AHHH")
                    summaries.append(response_summary)
                final_summary = f"Summary of Lecture {i}" + "\n".join(summaries)
                grand_summaries.append(final_summary)
        grand_summary_final = "\n".join(grand_summaries)
        header = f"Using these summaries of lectures as context, write a challenging exam according to the user's specifications covering the concepts in these lectures. Make sure to introduce your response as a practice exam for lectures {n1}-{n2}."
        build_context = header + "\n" + grand_summary_final  + "\n"  + prompt
                    # print(final_summary)
                
        self.messages.append(dict(role = "user", content = build_context))
        try:
            response = openai.ChatCompletion.create( 
                model=model,
                messages = self.messages,
                max_tokens=1000,
                temperature=0.5,
                ) 
        except:
            time.sleep(1.5) # this could minimum be 0.006
            response = openai.ChatCompletion.create( 
                        model=model,
                        messages = self.messages,
                        max_tokens=1000,
                        temperature=0.5,
                        ) 

        self.messages[-1] = (dict(role = "user", content = prompt))
        message = response['choices'][0]['message']['content']
        return message

    def locate_query(self, prompt, model="gpt-4"): # try gpt-4-32k if this runs out of tokens
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a response for the query. If the context seems irrelevant to the query, ignore the context proceed to respond normally."
        # counter = -1
        # meta_num = 0
        # lec_specific = False
        # split_prompt = prompt.split()
        # if "lecture" or "lec" in prompt.split():

        #     lec_specific = True
        #     for p in split_prompt:
        #         counter += 1
        #         print(counter)
        #         print(prompt.split())
        #         if p == "lecture" or p == "lec":
        #             meta_num = str(split_prompt[counter + 1].strip())
        #             print(meta_num)
        #             meta_num = re.sub("[^0-9]", "", meta_num)
        #             print("hai")
        #             print(meta_num)
        #             break

        

        query_response = openai.Embedding.create( # related text (large)
        model= "text-embedding-ada-002",
        input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        # if lec_specific:
        #     print("yuh")
        #     possible_contexts = index.query(
        #     vector=query_response_data,
        #     top_k=3,
        #     filter={
        #     "lecture": {"$eq": meta_num},
        #     },
        #     include_values=False
        #     )
        



        possible_contexts = index.query(
            vector=query_response_data,
            top_k=3,
            include_values=False
            )

        context = possible_contexts["matches"][0]["id"]
        context_copy = possible_contexts["matches"][0]["id"]
        #print(possible_contexts)
        if "-" in context:
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1]) 
            


        #print(possible_contexts["matches"])
        file_path = os.path.join(os.path.dirname(__file__), self.path+"/"+context)
        with open(file_path,encoding="utf8", errors="ignore") as f:
            if "-" in context_copy:
                span = 3000
                content = f.read()
                num_words = len(content.split())
                content = content.split()

                l = [" ".join(content[i:i+span]) for i in range(0, num_words, span)]
                content = l[ind - 1]
            else:

                content = f.read()
            header = f"Using this transcript of {context} as context, tell the user in which lecture # they can find their requested information and in what context."
            build_context = header + "\n" + content + "\n" + prompt
            self.messages.append(dict(role = "user", content = build_context))
            try:
                #print(ind)
                #print(len(self.messages[1]['content'].split()))
                response = openai.ChatCompletion.create( 
                    model=model,
                    messages = self.messages,
                    max_tokens=500,
                    temperature=0.7,
                    ) 
            except:
                #print("checking")

                print(self.messages)
                #print(len(self.messages[1]['content'].split()))


                # time.sleep() does not work. Instead, we have three options:
                    # chunk data into smaller pieces (~3000-4000 or less)
                    # if except is triggered, clear everything before current build_context and continue
                    # store message history before current build context elsewhere temporarily, run response, and add message history back when build_context is removed
                        # This is the best idea. Will work for follow up too bc follow up doesn't require large build contexts

                    # ^ This did not work. Next solution:
                    # 1. Chunk data into smaller pieces (~2000) <--- Seems like better solution until I can request a rate limit increase
                    # 2. Summarize build contexts with classifier
                #print(len(self.messages[1]['content'].split()))
                time.sleep(6)
                temp_msgs = [self.messages[0], self.messages[-1]]
                response = openai.ChatCompletion.create( 
                    model=model,
                    messages = temp_msgs,
                    max_tokens=100,
                    temperature=0.7,
                    ) 

            self.messages[-1] = (dict(role = "user", content = prompt))
            message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
            return message
                

        # else:
        #     context = f.read()
        #     build_context = header + "\n" + context + "\n" + prompt
        #     self.messages.append(dict(role = "user", content = build_context))
        #     try:
        #         response = openai.ChatCompletion.create( 
        #             model=model,
        #                 messages = self.messages,
        #                 max_tokens=1000,
        #                 temperature=0.5,
        #                 ) 
        #     except:
        #         time.sleep(0.5) # this could minimum be 0.006
        #         response = openai.ChatCompletion.create( 
        #                 model=model,
        #                 messages = self.messages,
        #                 max_tokens=1000,
        #                 temperature=0.5,
        #                 )

            # self.messages[-1] = (dict(role = "user", content = prompt))
            # message = response['choices'][0]['message']['content'] #response.choices[0].text.strip()
            # return message


    def main(self, prompt, query_type):
        # Impending Changes
        # 1: Remove summarize_query() function. lecture_notes_query() satisfies its use case and beyond.
        # 2: Add notes_query(). Takes general purpose notes on a lecture.
        
        # prompt = input("Enter GPT Prompt: ")
        # if prompt[(len(prompt) - 2):] == "-f":
                
        #     message = self.follow_up_query(prompt[:(len(prompt) - 2)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 3):] == "-qu":
                
        #     message = self.query_quiz(prompt[:(len(prompt) - 3)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 3):] == "-su":
        #     message = self.summarize_query(prompt[:(len(prompt) - 3)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 3):] == "-no":
        #     message = self.notes_query(prompt[:(len(prompt) - 3)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 3):] == "-lo":
        #     message = self.lecture_notes_query(prompt[:(len(prompt) - 3)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 2):] == "-t":
        #     message = self.test_query(prompt[:(len(prompt) - 2)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 3):] == "-fi":
        #     message = self.locate_query(prompt[:(len(prompt) - 3)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # elif prompt[(len(prompt) - 2):] == "-g" or prompt[(len(prompt) - 2):] != "-g":
        #     message = self.query(prompt[:(len(prompt) - 2)])
        #     self.messages.append(dict(role = "assistant", content = message))
        # return message
        if query_type == "follow up":
                
            message = self.follow_up_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "quiz":
                
            message = self.query_quiz(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "summary":
            message = self.summarize_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "notes":
            message = self.notes_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "outline":
            message = self.lecture_notes_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "exam":
            message = self.test_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "locate":
            message = self.locate_query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        elif query_type == "general":
            message = self.query(prompt)
            self.messages.append(dict(role = "assistant", content = message))
        return message




# maybe add tags "-g means general query" (all prompts are gonna be assumed -g), "-f" for follow up, and "-qu" for quiz

# cs_2110 = Course("CS2110", "Cornell", openai.api_key)
# cs_2110.embed_data()
#cs_105_A = Course("CS105-A", "Stanford", openai.api_key) # Test Case #1: CS 105 @ Stanford
#cs_105_A.embed_data()
# cs_105 = Course("CS105", "Stanford", openai.api_key) # Test Case #1: CS 105 @ Stanford
#econ_251 = Course("ECON251", "Yale", openai.api_key)
#econ_251.embed_data()
# econ_251.main()
#ibm_ai_essentials = Course("AI-Essentials", "IBM", openai.api_key)
#ibm_ai_essentials.embed_data()
#ibm_ai_essentials.main()
#cs_229 = Course("CS229", "Stanford", openai.api_key)
#cs_229.embed_data()

# gg_140 = Course("GG140", "Yale", openai.api_key)
# gg_140.embed_data()
#cs_105.embed_data()
#message1 = cs_105.query("What is a binary number?")
#message2 = cs_105.query("what is an example of a data privacy breach?")
#message3 = cs_105.query_quiz("Please provide a quiz for Lecture 26.1")
# cs_105.main()
#print(message)
#print(message2)
#print(message3)
# cs_105.embed_data() # only use once
#print(cs_105.check_record_stats())
# open-ai api key: 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