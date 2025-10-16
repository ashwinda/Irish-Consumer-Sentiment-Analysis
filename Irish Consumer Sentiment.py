import requests, time, random, pandas as pd, re
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
nltk.download('stopwords')

HEADERS = {'User-Agent': 'Mozilla/5.0'}
BRANDS = {
    'SuperValu': 'https://www.trustpilot.com/review/supervalu.ie',
    'Avoca': 'https://www.trustpilot.com/review/avoca.com',
    'Lily O\'Brien\'s': 'https://www.trustpilot.com/review/www.lilyobriens.ie'
}

def fetch_page(url):
    try: r = requests.get(url, headers=HEADERS, timeout=15); return r.text if r.status_code==200 else None
    except: return None

def parse_reviews(html):
    soup = BeautifulSoup(html,'html.parser'); reviews=[]
    for card in soup.select('article'):
        text = card.select_one('p'); text=text.get_text(strip=True) if text else ''
        date = card.select_one('time'); date=date.get('datetime') if date and date.get('datetime') else None
        rating_tag = card.find(attrs={'data-service-review-rating': True})
        rating = int(rating_tag['data-service-review-rating']) if rating_tag else None
        if text: reviews.append({'text':text,'date':date,'rating':rating})
    return reviews

analyzer = SentimentIntensityAnalyzer()
def score_reviews(revs):
    scored=[]
    for r in revs:
        s=analyzer.polarity_scores(r['text'])
        r.update(s); scored.append(r)
    return scored

all_results=[]
for brand,url in BRANDS.items():
    brand_reviews=[]
    for page in range(1,4):
        html=fetch_page(url+f'?page={page}')
        if html: brand_reviews.extend(parse_reviews(html))
        time.sleep(random.uniform(1.2,2.5))
    scored=score_reviews(brand_reviews)
    for s in scored: s['brand']=brand
    all_results.extend(scored)

df=pd.DataFrame(all_results)
df['date']=pd.to_datetime(df['date'],errors='coerce')
df.to_csv('trustpilot_reviews_scored.csv',index=False)
print('Saved trustpilot_reviews_scored.csv')

# Aggregate and plot
summary=df.groupby('brand')['compound'].mean().reset_index()
summary.rename(columns={'compound':'avg_sentiment'},inplace=True)
plt.figure(figsize=(8,4))
plt.bar(summary['brand'], summary['avg_sentiment'], color='green')
plt.title('Average sentiment by brand'); plt.ylabel('VADER compound'); plt.show()

# Wordcloud per brand
stop_words=set(stopwords.words('english'))
for brand in df['brand'].unique():
    texts=' '.join(df[df['brand']==brand]['text'].tolist())
    wc=WordCloud(width=800,height=400,stopwords=stop_words,collocations=False).generate(texts)
    plt.figure(figsize=(12,6)); plt.imshow(wc, interpolation='bilinear'); plt.axis('off'); plt.title(f'Wordcloud — {brand}'); plt.show()