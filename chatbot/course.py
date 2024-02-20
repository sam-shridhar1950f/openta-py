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
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import io
# 33290b8c-00ff-432a-9042-ba9bb087eebb
# Note To Self: 8000 tokens is about 6500 words, so use as modulo for calculating chunks

# "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" "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" # "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" # old openai api keys
# openai.api_key = "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"
# pinecone.init(api_key="190dabf8-0c0b-4690-8733-0be7fcecdb34") # init pinecone db

load_dotenv()
# S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_NAME = "openta-bucket"
# ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
ACCESS_KEY = "AKIAY27SSCRNPVHS3LXT"
# SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
SECRET_KEY = "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"


class Course:

    def __init__(self, course_title, school, api_key, path="", pinecone_index="", messages=[]):
        self.course_title = course_title  # Title of course
        self.school = school  # Institution that offers the course
        self.path = "courses/"+school+"/"+course_title
        self.api_key = api_key
        self.pinecone_index = school.lower() + "-" + course_title.lower()
        self.messages = [{
            "role": "system",
            "content": f"You are a teaching assistant for the course {course_title} at {school}"
        }]
        if self.pinecone_index in pinecone.list_indexes():
            pass
        else:
            # name of index needs to be lowercase
            pinecone.create_index(self.pinecone_index,
                                  dimension=1536, metric="cosine")

    def __str__(self):
        return f"{self.course_title}{self.school})"

    def embed_data(self):
        for filename in os.listdir(self.path):  # iterating over files
            fp = self.path + "/" + filename
            stripped_fp = self.path + "/"
            print(filename)
            shortened_responses = []
            # open(stripped_fp+"myfilexs_jmQt5D0s.txt") as f2:
            with open(fp, encoding="utf8", errors="ignore") as f:
                content = f.read()
                halved = False
                try:
                    response = openai.Embedding.create(  # related text (large)
                        model="text-embedding-ada-002",
                        input=[content])
                except:
                    print("error")
                    halved = True
                    num_words = len(content.split())
                    # divisor = (num_words // 6000) + 1 # do this logic tomorrow
                    span = 3000
                    words = content.split()
                    l = [" ".join(words[i:i+span])
                         for i in range(0, num_words, span)]

                    for text in l:
                        response_section = openai.Embedding.create(  # shortened
                            model="text-embedding-ada-002",
                            input=[text])
                        shortened_responses.append(response_section)

                index = pinecone.Index(self.pinecone_index)
                # print(filename[7:])
                c = 0
                if halved:
                    for resp in shortened_responses:
                        c += 1

                        resp_data = resp["data"][0]["embedding"]

                        # response_data = response_data_half1 + response_data_half2
                        index.upsert(
                            [(filename + "-" + str(c), resp_data, {"lecture": filename[7:]})])
                        # index.upsert([(filename + "-" + "2", response_data_half2, {"lecture": filename[7:]})])

                else:
                    response_data = response["data"][0]["embedding"]
                    index = pinecone.Index(self.pinecone_index)
                    index.upsert(
                        [(filename, response_data, {"lecture": filename[7:]})])

    def check_record_stats(self):
        index = pinecone.Index(self.pinecone_index)
        return index.describe_index_stats()

    def query(self, prompt, model="gpt-4"):  # try gpt-4-32k if this runs out of tokens
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a brief response for the query. If the context seems irrelevant to the query, ignore the context and proceed to respond normally."

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
            input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        possible_contexts = index.query(
            vector=query_response_data,
            top_k=3,
            include_values=False
        )

        context = possible_contexts["matches"][0]["id"]
        context_copy = possible_contexts["matches"][0]["id"]
        # print(possible_contexts)
        if "-" in context:
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1])  # 4-1: 1 is index, 4 is lecture

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # print(possible_contexts["matches"])
        file_path = os.path.join(os.path.dirname(
            __file__), self.path+"/"+context)
        if "-" in context_copy:
            span = 3000
            num_words = len(content.split())
            content = content.split()

            l = [" ".join(content[i:i+span])
                 for i in range(0, num_words, span)]
            content = l[ind - 1]
        build_context = header + "\nQuery: \"" + prompt + \
            "\"\n\n\nFOCUS ON ANSWERING THE QUERY AND ONLY THE QUERY. The following lecture may be useful in ANSWERING THE QUERY: " + content
        print("\n" + build_context + "\n")
        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )
        except:
            print(self.messages)

            # time.sleep() does not work. Instead, we have three options:
            # chunk data into smaller pieces (~3000-4000 or less)
            # if except is triggered, clear everything before current build_context and continue
            # store message history before current build context elsewhere temporarily, run response, and add message history back when build_context is removed
            # This is the best idea. Will work for follow up too bc follow up doesn't require large build contexts

            # ^ This did not work. Next solution:
            # 1. Chunk data into smaller pieces (~2000) <--- Seems like better solution until I can request a rate limit increase
            # 2. Summarize build contexts with classifier
            # print(len(self.messages[1]['content'].split()))
            time.sleep(6)
            temp_msgs = [self.messages[0], self.messages[-1]]
            response = openai.ChatCompletion.create(
                model=model,
                messages=temp_msgs,
                max_tokens=1000,
                temperature=0.5,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
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
        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a quiz according to the user's specifications (if there are no specifications, use general rules). Use normal text, avoid markdown and ### signs:"

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
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

        print(possible_contexts)

        # print(possible_contexts)
        multiple_matches = False
        context = possible_contexts["matches"][0]["id"]
        print(context)
        if "-" in context:
            multiple_matches = True
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1])

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context

        build_context = header + "\n" + content + "\n" + prompt
        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )
        except:
            time.sleep(0.5)  # this could minimum be 0.006
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
        return message

    def summarize_query(self, prompt, model="gpt-4-turbo-preview"):
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

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
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

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context

        build_context = header + "\n" + content  # + "\n" + prompt
        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )
        except:
            time.sleep(0.5)  # this could minimum be 0.006
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
        return message

    def lecture_notes_query(self, prompt, model="gpt-4-turbo-preview"):
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        # print(lec_num)
        index = pinecone.Index(self.pinecone_index)
        header = f"Using this transcript of lecture {lec_num} as context,write a summarizing and concise lecture outline according to the user's specifications. Make sure to introduce your response as an outline for lecture {lec_num}."

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
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

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context

        build_context = header + "\n" + content + "\n" + prompt
        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )
        except:
            time.sleep(0.5)  # this could minimum be 0.006
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
        return message

    def notes_query(self, prompt, model="gpt-4-turbo-preview"):
        words = prompt.split(" ")
        lec_num = ""
        for word in words:
            if bool(re.search(r'\d', word)):
                lec_num = word
        if lec_num[-1].isdigit() != True:
            lec_num = lec_num[:-1]
        index = pinecone.Index(self.pinecone_index)
        header = f"Using this text as context, write specific, detailed notes according to the user's specifications. Make sure to introduce your response as notes for lecture {lec_num}."

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
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

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context

            build_context = header + "\n" + content + "\n" + prompt
            self.messages.append(dict(role="user", content=build_context))
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=self.messages,
                    max_tokens=1000,
                    temperature=0.5,
                )
            except:
                time.sleep(0.5)  # this could minimum be 0.006
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=self.messages,
                    max_tokens=1000,
                    temperature=0.5,
                )

            self.messages[-1] = (dict(role="user", content=prompt))
            # response.choices[0].text.strip()
            message = response['choices'][0]['message']['content']
            return message

    def prompt_selector(self):
        pass

    def follow_up_query(self, prompt, model="gpt-4"):
        header = "Allow the user to follow up on the previous query and response: "
        f"Respond to the following follow up question. If it references specific content of {self.school} {self.course_title}, still answer the question directly and to the best of your knowledge."
        build_context = header + "\n" + prompt
        self.messages.append(dict(role="user", content=build_context))
        response = openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            max_tokens=1000,
            temperature=0.5,
        )
        self.messages[-1] = (dict(role="user", content=prompt))
        # replace self.messages[-1] with prompt
        # self.messages[-1] = prompt
        # do this for every query type
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
        return message

    # user requests sample exam for lectures [x,y], method returns exam with answers
    def test_query(self, prompt, model="gpt-4-turbo-preview"):
        s = prompt.split(" ")

        s = s[-1].split("-")
        print(s)
        n1, n2 = s[0], s[1]
        print(n1, n2)

        index = pinecone.Index(self.pinecone_index)
        header = f"Using this text as context, write a challenging exam according to the user's specifications. Make sure to introduce your response as an exam for lectures {n1}-{n2}."

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
            input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        list_of_possible_contexts = []

        grand_summaries = []
        for i in range(int(n1), int(n2)+1):

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

            try:
                file_path = f"{self.school}/{self.course_title}/{context}"
                s3_client = boto3.client(
                    's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
                file_content = io.BytesIO()
                print(S3_BUCKET_NAME)
                print(ACCESS_KEY)
                print(SECRET_KEY)
                print(file_path)
                s3_client.download_fileobj(
                    S3_BUCKET_NAME, file_path, file_content)
                print("s3 worked")
                file_content.seek(0)
                content = file_content.read().decode("windows-1252", errors="ignore")
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    raise

        # print(possible_contexts["matches"])
        # print(context)
            # if there are multiple lectures for one quiz, generate ~2000-2500 token summary for each section, concatenate summaries into one body, and pass as context

                summaries = []
                # response_summary = summarize(context, 0.03)
                response_summary = content
                print(context)
                print("AHHH")
                summaries.append(response_summary)
                final_summary = f"Summary of Lecture {i}" + \
                    "\n".join(summaries)
                grand_summaries.append(final_summary)
        grand_summary_final = "\n".join(grand_summaries)
        header = f"Using these summaries of lectures as context, write a challenging exam according to the user's specifications covering the concepts in these lectures. Make sure to introduce your response as a practice exam for lectures {n1}-{n2}."
        build_context = header + "\n" + grand_summary_final + "\n" + prompt
        # print(final_summary)

        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )
        except:
            time.sleep(1.5)  # this could minimum be 0.006
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=1000,
                temperature=0.5,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        message = response['choices'][0]['message']['content']
        return message

    # try gpt-4-32k if this runs out of tokens
    def locate_query(self, prompt, model="gpt-4"):
        index = pinecone.Index(self.pinecone_index)
        header = "Using this lecture transcript as context, provide a response for the query. If the context seems irrelevant to the query, ignore the context proceed to respond normally."

        query_response = openai.Embedding.create(  # related text (large)
            model="text-embedding-ada-002",
            input=[prompt])

        query_response_data = query_response["data"][0]["embedding"]

        possible_contexts = index.query(
            vector=query_response_data,
            top_k=3,
            include_values=False
        )

        context = possible_contexts["matches"][0]["id"]
        context_copy = possible_contexts["matches"][0]["id"]
        if "-" in context:
            context_split = context.split("-")
            context = context_split[0]
            ind = int(context_split[1])

        try:
            file_path = f"{self.school}/{self.course_title}/{context}"
            s3_client = boto3.client(
                's3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
            file_content = io.BytesIO()
            print(S3_BUCKET_NAME)
            print(ACCESS_KEY)
            print(SECRET_KEY)
            print(file_path)
            s3_client.download_fileobj(
                S3_BUCKET_NAME, file_path, file_content)
            print("s3 worked")
            file_content.seek(0)
            content = file_content.read().decode("windows-1252", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        header = f"Using this transcript of {content} as context, tell the user in which lecture they can find their requested information and in what context."
        build_context = header + "\n" + content + "\n" + prompt
        self.messages.append(dict(role="user", content=build_context))
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                max_tokens=500,
                temperature=0.7,
            )
        except:
            print(self.messages)
            # print(len(self.messages[1]['content'].split()))

            # time.sleep() does not work. Instead, we have three options:
            # chunk data into smaller pieces (~3000-4000 or less)
            # if except is triggered, clear everything before current build_context and continue
            # store message history before current build context elsewhere temporarily, run response, and add message history back when build_context is removed
            # This is the best idea. Will work for follow up too bc follow up doesn't require large build contexts

            # ^ This did not work. Next solution:
            # 1. Chunk data into smaller pieces (~2000) <--- Seems like better solution until I can request a rate limit increase
            # 2. Summarize build contexts with classifier
            # print(len(self.messages[1]['content'].split()))
            time.sleep(6)
            temp_msgs = [self.messages[0], self.messages[-1]]
            response = openai.ChatCompletion.create(
                model=model,
                messages=temp_msgs,
                max_tokens=100,
                temperature=0.7,
            )

        self.messages[-1] = (dict(role="user", content=prompt))
        # response.choices[0].text.strip()
        message = response['choices'][0]['message']['content']
        return message

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
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "quiz":

            message = self.query_quiz(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "summary":
            message = self.summarize_query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "notes":
            message = self.notes_query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "outline":
            message = self.lecture_notes_query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "exam":
            message = self.test_query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "locate":
            message = self.locate_query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        elif query_type == "general":
            message = self.query(prompt)
            self.messages.append(dict(role="assistant", content=message))
        return message

# maybe add tags "-g means general query" (all prompts are gonna be assumed -g), "-f" for follow up, and "-qu" for quiz

# cs_2110 = Course("CS2110", "Cornell", openai.api_key)
# cs_2110.embed_data()
# cs_105_A = Course("CS105-A", "Stanford", openai.api_key) # Test Case #1: CS 105 @ Stanford
# cs_105_A.embed_data()
# cs_105 = Course("CS105", "Stanford", openai.api_key) # Test Case #1: CS 105 @ Stanford
# econ_251 = Course("ECON251", "Yale", openai.api_key)
# econ_251.embed_data()
# econ_251.main()
# ibm_ai_essentials = Course("AI-Essentials", "IBM", openai.api_key)
# ibm_ai_essentials.embed_data()
# ibm_ai_essentials.main()
# cs_229 = Course("CS229", "Stanford", openai.api_key)
# cs_229.embed_data()

# gg_140 = Course("GG140", "Yale", openai.api_key)
# gg_140.embed_data()
# cs_105.embed_data()
# message1 = cs_105.query("What is a binary number?")
# message2 = cs_105.query("what is an example of a data privacy breach?")
# message3 = cs_105.query_quiz("Please provide a quiz for Lecture 26.1")
# cs_105.main()
# print(message)
# print(message2)
# print(message3)
# cs_105.embed_data() # only use once
# print(cs_105.check_record_stats())
# open-ai api key: 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
