DB_FILE = "database.sqlite3"
BATCH_SIZE = 1000
SIMILARITY_CUTOFF = 0.67

####IMPORTS
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
import sqlite3

con = sqlite3.connect(DB_FILE)
cur = con.cursor()
import time
import math

import tensorflow as tf #сравнение массивов из таблицы crawled вопросов со стандартными вопросами

cosine_similarity = tf.keras.losses.CosineSimilarity(axis=0)

#TODO with tf.device('/cpu:0'): benchmark for embeddings  #для быстродействия

####LOAD QUESTIONS
t = time.time()
cur.execute("select id, text from staticquestions") #текст и айди из статиквопросов
static_questions_db = cur.fetchall()
static_questions_db_texts = []
for i in range(len(static_questions_db)):
    static_questions_db_texts.append(static_questions_db[i][1]) #сделал массив из вопросов
static_questions_db_embeddings = embed(static_questions_db_texts)
static_questions_db_embeddings_norm = tf.math.l2_normalize(static_questions_db_embeddings, 1) #привел к каким то числам
t2 = time.time()
print("Loaded static questions in " + str((t2 - t)))


####MAIN LOGIC функция offset- отступ
def process_batch(offset, count):
    t = time.time()
    cur.execute("select id, questionText from Questions where status = 'CRAWLED' ORDER BY ID LIMIT "
                 + str(count) + " OFFSET " + str(offset))
    chunk_db = cur.fetchall() #состоит из массива содержаших айди и Question Text
    chunk_db_texts = []
    for i in range(len(chunk_db)):
        chunk_db_texts.append(chunk_db[i][1][:255])  # take first 255 characters of text
    print(chunk_db_texts)
    chunk_embeddings = embed(chunk_db_texts)
    chunk_embeddings_norm = tf.math.l2_normalize(chunk_embeddings, 1) #перевел текст в массив с цифрами
    similarity = tf.matmul(chunk_embeddings_norm, static_questions_db_embeddings_norm, transpose_b=True)
    similarity = tf.math.abs(similarity)
    similarity_max = tf.reduce_max(similarity, axis=1)
    similarity_max_index = tf.math.argmax(similarity, axis=1)

    results = []
    for i in range(len(similarity_max_index)):
        sim = similarity_max[i]
        if sim >= SIMILARITY_CUTOFF:
            results.append((chunk_db[i][0], static_questions_db[similarity_max_index[i]][0], float(sim)))
            # print("{0} ~ {1} ~ {2}".format(chunk_db_texts[i], static_questions_db_texts[similarity_max_index[i]],sim))
    cur.executemany("INSERT INTO UseSimilarityV1s VALUES (null, ?,?,?, date('now'), date('now'))", results)
    con.commit() #выполнить запрос
    t2 = time.time()
    print("Processed " + str(len(chunk_db)) + " questions in " + str((t2 - t)) + ". Offset: " + str(offset))


####MAIN LOOP
cur.execute("select count(*) from Questions where status = 'CRAWLED'")
count = cur.fetchone()[0]
iterations = math.ceil(count / BATCH_SIZE)
print("Processing {0} questions in {1} iterations".format(count, iterations))
for i in range(0, iterations):
    process_batch(BATCH_SIZE * i, BATCH_SIZE)

####SHUTDOWN
con.close()
