# Hostname2keywords

Extract keywords from hostnames (E.g. 24h, dantri, king, xnxx,...)


# Handling hostnames

Given hostnames has **absolutely no extra information**. So in order to handle this problem, I used these hostnames as keywords for searching and crawling data from google. There was (of course) some issues with google anti-crawling methods so I did some tricks like rotating proxies and switching user agents back and forth to get rid of annoying captcha and temporary ban of google every 150 or 200 requests to their server. Free proxies are also available on [proxynova](https://www.proxynova.com/).  There's definitely a huge list of user agents that you can get on [this shortcut](https://developers.whatismybrowser.com/useragents). I did write the script for crawling proxies from proxynova. They are all available in [this google_scraper repository](https://github.com/vudat1710/google_scraper) (remember to change the settings to crawl faster when using proxy crawler). Also setting some delay and number of concurrent requests when using google search crawler is highly recommend for the bot to avoid being banned by google. Crawled fields include keyword (hostname), url, description, and title.
## Getting more information

From my analysis with google data I crawled. There's no consistency in search results, especially if you doing the search using different IP or different browsers (I'm not too sure about it but the crawled data is definitely inconsistent when I tested on a different machine). And also from my analysis, the search descriptions on google results usually do not represent what the actual content of the website really is. So I have to find more information to do my research. Soon I realize that in every website's html response, the **head** part usually contains useful text in the **title** html tag. There also a **meta** tag with **class keywords** sometimes contains topics of the website but with Vietnamese websites, these words just repetitive and not very useful in detecting the content of that website. The script for crawling website title is available in [this repository](https://github.com/vudat1710/web_url_crawler)

## Merging data

In order to create the merged data in `data` folder, you should use langdetect to get the language of the **description** field and **web_title** field. Then if the language of description is also the language of web_title, merge them into a new field called **merged_text**


# Keyword Extraction

There's numerous ways to extract keywords from text. However, most of them are only compatible with long text while text data crawled from the above method is generally short (1 or 2 sentences) mixed of English and Vietnamese. Of course, there are some methods that could be applied for short text like [Yake](http://yake.inesctec.pt/apidocs/#/available%20methods) or [Rake](https://github.com/csurfer/rake-nltk), but the results produced are somewhat random and not reliable. From my experience in applying these methods to this data, I found out that TextRank produced the most reasonable results. You can read more about this approach from [this Kaggle article](https://www.kaggle.com/john77eipe/textrank-for-keyword-extraction-by-python) or much more details in Vietnamese in [this thesis](http://data.uet.vnu.edu.vn/jspui/bitstream/123456789/1078/1/k21_Nguyen%20Vu%20Chi%20loan.pdf)
This repo provides you with text rank implementations on both Vietnamese and English. You can apply it with more than 1-gram using the parameter `ngram=x`. For Vietnamese (`text_rank_vn.py`) you can use the pretrained POS tagger model of [underthesea](https://github.com/undertheseanlp/underthesea) or [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) using the param `use_vncorenlp` in `analyze` function.


# Strong keywords extraction

The result from the above method is decent but still have lots of distracting words so in this stage, I used some gensims pretrained language models for detecting representative words out of a word list ([English model](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and [Vietnamese model](https://github.com/sonvx/word2vecVN)). Assume that all words in a list representing the same topic for a website, I compare word similarity of a word in the given list to every other words from the same list using pretrained model and cosine similarity. Then I compute average score for each word and rank all words from highest to lowest score, extract 2 to 3 highest score words from them to be the **strong keywords**. This method is applied on both **merged_text** field and **web_title** field then merge these 2 results to achieve the final keywords list for each given hostname.

The result achieved is exciting, many words could be the representative words for the given hostname. Using this method could be beneficial especially in recommendation problems or some kinds of analyzing users' behavior. Of course this tool should be coded in more efficient way with better interpretation, but I think this is a good way to analyze a website so we can automatically answer to the question: "What is the actual content of this website?"