#Internship Assigment1

## How install

##Fork Github repository (make private and share with AG-Teammate)

Download the database (ask AG for a link)

Install dependencies (requirements.txt) in the project

Copy the database and truncate UseSimilarityV1s table

Run map-glassdoor-to-native.py

The script should insert entries in UseSimilarityV1s table

Debug the script line-by-line to understand the algorithm

Improve the script:

Line 46 simply takes the first 255 symbols of the questionText.
Some users put more content into this field
Analyze the field contents and suggest how to improve the script
Develop the improvements and create a pull request to the original repository on Github
Answer the following questions:
What percentage of crawled questions has been mapped to the static questions?
How to improve parsing of the questionText column?
How can we use the data in the Comments table?
