
# Threat Level Midnight

## Overview of Project
2020 - the beginning of the Pandemic. Unprecedented restrictions of movements and lockdowns. What do most people do when they are stuck at home? 

Watch "The Office". 

According to Nielsen, "The Office" had more than 57 billion minutes streamed in 2020 ([source](https://www.nielsen.com/us/en/insights/article/2021/tops-of-2020-nielsen-streaming-unwrapped/)). Why is the show so popular even after nearly a decade from its original airing? In recent years, many podcasts have popped up trying to answer this question. These podcasts track the history of the show and take deep dives into various episodes to figure out the answer, but we are turning to data science to unravel the mystery.

What can data analysis, natural language processing, and machine learning from the script tell us about why "The Office" is so popular?

## Data Source and Questions to Answer
### Data Source: 
[transcripts.foreverdreaming.org](https://transcripts.foreverdreaming.org/viewtopic.php?f=574&t=25301&sid=55341a4d23dec85533d960b6ff9edc2a)
This website hosts transcripts of many popular TV shows and movies.

### Questions to Answer:
1) Who are the characters of "The Office"?   
2) What was each character's overall sentiment throughout the show?   
3) Can we generate text for one or more characters with NLP Machine Learning?

## Data Pipeline
- Scrape script from the website (API, Python)
- ETL and store in a SQL database (NLP, Python, SQL)
- Query the data and create javascript arrays (Javascript, Python, SQL)
- Visualize on webpage (Javascript)
- Predict sentences spoken by character(s) (NLP, RNN, Python)



## References
[Text Generation with RNN](https://www.tensorflow.org/text/tutorials/text_generation)
