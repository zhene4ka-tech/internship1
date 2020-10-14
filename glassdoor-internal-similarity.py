import csv

DB_FILE = "E:\\tmys\\glassdoor-scrape\\database.sqlite3"
SIMILARITY_CUTOFF = 0.67
BATCH_SIZE = 200
SUBMATRIX_LIMIT = 3000
SPLIT_LIMIT_BATCHES = 750

####IMPORTS
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
import sqlite3

con = sqlite3.connect(DB_FILE)
cur = con.cursor()
import time
import math

import tensorflow as tf

cosine_similarity = tf.keras.losses.CosineSimilarity(axis=0)

split_num = 0
processed_ids = []
processed_embed_matrices = [[]]
processed_counts = []
processed_texts = []
submatrix_index = 0


def reset_data():
    global processed_ids, processed_embed_matrices, processed_counts, processed_texts, submatrix_index
    processed_ids = []
    processed_embed_matrices = [[]]
    processed_counts = []
    processed_texts = []
    submatrix_index = 0


reset_data()


def process_batch(offset, limit):
    global submatrix_index
    # t = time.time()
    # get next question
    cur.execute(
        "select id, questionText from Questions where status = 'CRAWLED' and "
        "not exists(select id from UseSimilarityV1s where question_id = Questions.id) ORDER BY ID LIMIT {0} OFFSET {1}".format(
            limit, offset))
    db_all = cur.fetchall()
    db_texts = []
    for i in range(len(db_all)):
        db_texts.append(db_all[i][1][:255])  # take first 255 characters of text
    # print(db)
    # print(db[1])
    # create embedding
    db_embed_all = tf.math.l2_normalize(embed(db_texts), 1)
    for i in range(len(db_texts)):
        # first question, just append
        if len(processed_embed_matrices[0]) == 0:
            processed_ids.append(db_all[i][0])
            processed_embed_matrices[0].append(db_embed_all[i])
            processed_counts.append(1)
            processed_texts.append(db_texts[i])
        else:
            found = False
            for j in range(len(processed_embed_matrices)):
                # find similarity matrix
                similarity = tf.matmul([db_embed_all[i]], processed_embed_matrices[j], transpose_b=True)
                similarity = tf.math.abs(similarity)
                similarity_max = tf.reduce_max(similarity, axis=1)
                similarity_max_index = int(tf.math.argmax(similarity, axis=1))
                # print(similarity_max)
                if similarity_max >= SIMILARITY_CUTOFF:
                    # print(similarity_max_index)
                    # print("found match {0} =TO= {1}".format(db[1], processed_texts[similarity_max_index]))
                    processed_counts[j * SUBMATRIX_LIMIT + similarity_max_index] += 1  # increase counter
                    found = True
                    break  # we found acceptable match in this submatrix, do not go further

            if not found:
                # print("no match {0}".format(db[1]))
                if len(processed_embed_matrices[submatrix_index]) >= SUBMATRIX_LIMIT:
                    # create a new submatrix for stroing results
                    submatrix_index = submatrix_index + 1
                    processed_embed_matrices.append([])
                processed_ids.append(db_all[i][0])
                processed_embed_matrices[submatrix_index].append(db_embed_all[i])
                processed_counts.append(1)
                processed_texts.append(db_texts[i])


def save():
    print("saving intermediate results")
    csvfile = open('glassdoor-internal-similarity-split{0}.csv'.format(split_num), 'w', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    for i in range(len(processed_ids)):
        writer.writerow([processed_ids[i], processed_texts[i], processed_counts[i]])
    csvfile.close()


try:
    cur.execute("select count(*) from Questions where status = 'CRAWLED' and "
                "not exists(select id from UseSimilarityV1s where question_id = Questions.id)")
    count = cur.fetchone()[0]
    iterations = math.ceil(count / BATCH_SIZE)
    for i in range(0, iterations): #TODO SKIP FIRST BATCH
        t = time.time()
        process_batch(BATCH_SIZE * i, BATCH_SIZE)
        t2 = time.time()
        print(
            "Processed {3} results in {0}. Matrices: {1}. Index: {2}".format(t2 - t, len(processed_embed_matrices), i,
                                                                             BATCH_SIZE))
        if i % 10 == 0:
            save()
        if i > 0 and i % SPLIT_LIMIT_BATCHES == 0:
            # create new file
            save()
            split_num += 1
            reset_data()

except KeyboardInterrupt:
    save()
except:
    raise

save()
