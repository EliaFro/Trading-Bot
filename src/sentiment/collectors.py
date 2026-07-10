"""
Sentiment data collectors.

NewsCollector works WITHOUT any API keys (public CryptoCompare news endpoint
+ CoinDesk/Cointelegraph RSS). Reddit/Twitter collectors activate only when
credentials are supplied; without keys they return empty lists gracefully.

Every collector returns a list of dicts:
    {'text': str, 'source': str, 'timestamp': datetime, 'url': str}
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
USER_AGENT = {'User-Agent': 'crypto-sentiment-bot/2.0 (research)'}

RSS_FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss',
}
CRYPTOCOMPARE_NEWS = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN'


async def _fetch_text(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, headers=USER_AGENT) as resp:
            if resp.status == 200:
                return await resp.text()
            logger.warning(f"{url} returned HTTP {resp.status}")
    except Exception as e:
        logger.warning(f"Fetch failed {url}: {e}")
    return None


class NewsCollector:
    """Keyless crypto news headlines from public feeds."""

    def __init__(self, api_keys: Optional[Dict] = None):
        self.api_keys = api_keys or {}

    async def collect_news(self, limit: int = 50) -> List[Dict]:
        items: List[Dict] = []
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            results = await asyncio.gather(
                self._collect_cryptocompare(session, limit),
                *[self._collect_rss(session, name, url, limit)
                  for name, url in RSS_FEEDS.items()],
                return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"News source failed: {result}")
            else:
                items.extend(result)
        # newest first, cap total
        items.sort(key=lambda x: x['timestamp'], reverse=True)
        return items[:limit * 2]

    async def _collect_cryptocompare(self, session, limit: int) -> List[Dict]:
        url = CRYPTOCOMPARE_NEWS
        key = self.api_keys.get('cryptocompare')
        if key:
            url += f'&api_key={key}'
        try:
            async with session.get(url, headers=USER_AGENT) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception as e:
            logger.warning(f"CryptoCompare news failed: {e}")
            return []
        items = []
        for article in (data.get('Data') or [])[:limit]:
            items.append({
                'text': f"{article.get('title', '')}. {article.get('body', '')[:300]}",
                'source': 'cryptocompare',
                'timestamp': datetime.fromtimestamp(
                    article.get('published_on', 0), tz=timezone.utc),
                'url': article.get('url', ''),
                'categories': article.get('categories', ''),
            })
        return items

    async def _collect_rss(self, session, name: str, url: str,
                           limit: int) -> List[Dict]:
        text = await _fetch_text(session, url)
        if not text:
            return []
        items = []
        try:
            root = ET.fromstring(text)
            for item in root.iter('item'):
                title = item.findtext('title') or ''
                description = item.findtext('description') or ''
                pub = item.findtext('pubDate') or ''
                try:
                    from email.utils import parsedate_to_datetime
                    ts = parsedate_to_datetime(pub)
                except Exception:
                    ts = datetime.now(timezone.utc)
                items.append({
                    'text': f"{title}. {description[:300]}",
                    'source': name,
                    'timestamp': ts,
                    'url': item.findtext('link') or '',
                    'categories': '',
                })
                if len(items) >= limit:
                    break
        except ET.ParseError as e:
            logger.warning(f"RSS parse failed for {name}: {e}")
        return items


class RedditCollector:
    """Reddit posts from crypto subreddits. Requires praw + credentials;
    without them, collect_posts returns []. (Optional in v1.)"""

    SUBREDDITS = ['CryptoCurrency', 'Bitcoin', 'ethereum', 'solana']

    def __init__(self, client_id: str = None, client_secret: str = None,
                 user_agent: str = 'crypto_sentiment_bot/1.0'):
        self.enabled = bool(client_id and client_secret)
        self._reddit = None
        if self.enabled:
            try:
                import praw
                self._reddit = praw.Reddit(client_id=client_id,
                                           client_secret=client_secret,
                                           user_agent=user_agent)
            except ImportError:
                logger.warning("praw not installed — Reddit collector disabled")
                self.enabled = False

    async def collect_posts(self, limit: int = 100) -> List[Dict]:
        if not self.enabled:
            return []

        def _fetch():
            posts = []
            per_sub = max(limit // len(self.SUBREDDITS), 5)
            for sub in self.SUBREDDITS:
                try:
                    for post in self._reddit.subreddit(sub).hot(limit=per_sub):
                        posts.append({
                            'text': f"{post.title}. {(post.selftext or '')[:300]}",
                            'source': f'reddit/{sub}',
                            'timestamp': datetime.fromtimestamp(
                                post.created_utc, tz=timezone.utc),
                            'url': post.url,
                            'score': post.score,
                        })
                except Exception as e:
                    logger.warning(f"Reddit r/{sub} failed: {e}")
            return posts

        return await asyncio.to_thread(_fetch)


class TwitterCollector:
    """Twitter/X recent search. Requires a bearer token; otherwise no-op."""

    QUERY = '(bitcoin OR ethereum OR solana OR crypto) -is:retweet lang:en'

    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token
        self.enabled = bool(bearer_token)

    async def collect_tweets(self, limit: int = 100) -> List[Dict]:
        if not self.enabled:
            return []
        url = ('https://api.twitter.com/2/tweets/search/recent'
               f'?query={self.QUERY}&max_results={min(limit, 100)}'
               '&tweet.fields=created_at')
        headers = {'Authorization': f'Bearer {self.bearer_token}', **USER_AGENT}
        try:
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning(f"Twitter API HTTP {resp.status}")
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.warning(f"Twitter collection failed: {e}")
            return []
        items = []
        for tweet in data.get('data', []):
            try:
                ts = datetime.fromisoformat(
                    tweet['created_at'].replace('Z', '+00:00'))
            except Exception:
                ts = datetime.now(timezone.utc)
            items.append({'text': tweet.get('text', ''), 'source': 'twitter',
                          'timestamp': ts, 'url': ''})
        return items
