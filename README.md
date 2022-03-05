
# Threat Level Midnight

## Overview of Project
2020 - the beginning of the Pandemic. Unprecedented restrictions of movements and lockdowns. What do most people do when they are stuck at home? 

Watch "The Office". 

According to Nielsen, "The Office" had more than 57 billion minutes streamed in 2020 ([source](https://www.nielsen.com/us/en/insights/article/2021/tops-of-2020-nielsen-streaming-unwrapped/)). Why is the show so popular even after nearly a decade from its original airing? In recent years, many podcasts have popped up trying to answer this question. These podcasts track the history of the show and take deep dives into various episodes to figure out the answer, but we are turning to data science to unravel the mystery.

What can data analysis, natural language processing, and machine learning from the script tell us about why "The Office" is so popular?

### Data Sources: 
- [transcripts.foreverdreaming.org](https://transcripts.foreverdreaming.org/viewtopic.php?f=574&t=25301&sid=55341a4d23dec85533d960b6ff9edc2a)
This website hosts transcripts of many popular TV shows and movies.
- [ratinggraph.com](https://www.ratingraph.com/tv-shows/the-office-ratings-17546/#episodes) This website host ratings of tv shows and movies 
- [Wikipedia](https://en.wikipedia.org/wiki/The_Office_(American_TV_series)) Where we found episode details such as writer, director and air date.

<!-- ### Questions to Answer:
1) Who are the characters of "The Office"?   
2) Which character had the most lines throughout the show?
3) What was each character's overall sentiment throughout the show?   
4) Can we generate text for one or more characters with NLP Machine Learning? -->

## Having Fun With Data Science 

We can use data analytics and machine learning to do really serious and important things in the field of health care, we can help our businesses increase profit margins, or we can revolutionize molecular biology. But... it can also be something to play with, to dig into hobby topics, and be used to find beauty in ordinary things. 


When deciding to use the script of The Office as the data source for this project, having fun was top of mind. Working with written word also presented unique opportunites and challenges. The big challenge to tackle... could we use the data to traine a machine learning model to generate characters lines? Or even generate a scene from The Office?


### Getting the Data
[Transcripts Forever Dreaming](https://transcripts.foreverdreaming.org/about/) is a fan run site where individuals have taken the time to transcribe many television shows. The site has been around since the "days of dial-up modems." The scripts were simple enough to scrape from the HTML as it was dated, but having humans transcribe the shows introduced a higher volume of typos and clean up to be done. 

### Does it speak?
In our pursuit of reincarnating the our favorite characters from the show, wanted to create an interactive text generator that would be recognizable to fans of the show. We tested a few different different models, but ultimately decided to use a Recurring Neural Network (RNN) model. In addition to producing the best results, the RNN model was easier to use and computationally less expensive than other models we tested. When it was time to train the model, we chose Michael Scott, the main character. He had the most lines of any other character in the show and we thought if anyone had a personality that was big enough to shine through, it would be his. The result is an interactive text generator that, while imperfect, is undeniably Michael Scott, the lovably awkward Regional Manager of Dunder Mifflin, Scranton.

 
### What can we see? 
Once the script was cleaned and structured by season, episode, and character we were able to apply a sentiment analysis to each line in the script. We opted to use vaderSentiment over textblob. While both 

### And Scene 
The final challenge, could we make a computer generated scene for The Office? Checkout the scene at the bottom of the website and let us know what you think?

## Data Pipeline
- Scrape script from the websites (BeautifulSoup, Browser)
- Data Cleaning (pandas, numpy)
- Data Analysis (pandas, vaderSentiment)
- Visualize on webpage (Florish and Tableau)
- Predict sentences spoken by character(s) (NLP, RNN, Python)



## References
[Text Generation with RNN](https://www.tensorflow.org/text/tutorials/text_generation)

