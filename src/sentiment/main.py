#!/usr/bin/env python3
"""
Sentiment Analysis Service - Main Entry Point
Runs as a separate service to collect and analyze market sentiment
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import signal
import json
import redis
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/sentiment_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

class SentimentService:
    """Main sentiment analysis service"""
    
    def __init__(self):
        self.running = False
        self.redis_client = None
        self.collectors = {}
        self.analyzer = None
        self.aggregator = None
        
        # Configuration
        self.config = self._load_config()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        """Load configuration from environment and files"""
        return {
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'symbols': os.getenv('TRADING_SYMBOLS', 'BTC,ETH,SOL').split(','),
            'update_interval': int(os.getenv('SENTIMENT_UPDATE_INTERVAL', '300')),
            'apis': {
                'reddit': {
                    'client_id': os.getenv('REDDIT_CLIENT_ID'),
                    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
                    'user_agent': os.getenv('REDDIT_USER_AGENT', 'crypto_sentiment_bot/1.0')
                },
                'twitter': {
                    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN')
                },
                'news': {
                    'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
                    'newsapi': os.getenv('NEWSAPI_KEY')
                }
            }
        }
    
    def _initialize_components(self):
        """Initialize all sentiment analysis components"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(
                self.config['redis_url'],
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Import components (delayed import to avoid circular dependencies)
            from src.sentiment.collectors import RedditCollector, TwitterCollector, NewsCollector
            from src.sentiment.analyzer import CryptoSentimentAnalyzer
            from src.sentiment.aggregator import SentimentAggregator
            
            # Initialize sentiment analyzer
            self.analyzer = CryptoSentimentAnalyzer()
            logger.info("Initialized sentiment analyzer")
            
            # Initialize collectors
            if self.config['apis']['reddit']['client_id']:
                self.collectors['reddit'] = RedditCollector(
                    client_id=self.config['apis']['reddit']['client_id'],
                    client_secret=self.config['apis']['reddit']['client_secret'],
                    user_agent=self.config['apis']['reddit']['user_agent']
                )
                logger.info("Initialized Reddit collector")
            
            if self.config['apis']['twitter']['bearer_token']:
                self.collectors['twitter'] = TwitterCollector(
                    bearer_token=self.config['apis']['twitter']['bearer_token']
                )
                logger.info("Initialized Twitter collector")
            
            news_apis = {}
            if self.config['apis']['news']['cryptocompare']:
                news_apis['cryptocompare'] = self.config['apis']['news']['cryptocompare']
            if self.config['apis']['news']['newsapi']:
                news_apis['newsapi'] = self.config['apis']['news']['newsapi']
            
            if news_apis:
                self.collectors['news'] = NewsCollector(news_apis)
                logger.info("Initialized news collector")
            
            # Initialize aggregator
            self.aggregator = SentimentAggregator(self.analyzer)
            logger.info("Initialized sentiment aggregator")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the sentiment analysis service"""
        logger.info("Starting Sentiment Analysis Service...")
        self.running = True
        
        # Start collection tasks
        tasks = [
            asyncio.create_task(self._collection_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in sentiment service: {e}")
        finally:
            await self.shutdown()
    
    async def _collection_loop(self):
        """Main collection and analysis loop"""
        while self.running:
            try:
                start_time = datetime.now()
                logger.info("Starting sentiment collection cycle...")
                
                # Collect data from all sources
                social_posts = []
                news_articles = []
                
                # Collect Reddit posts
                if 'reddit' in self.collectors:
                    try:
                        reddit_posts = await self.collectors['reddit'].collect_posts(limit=100)
                        social_posts.extend(reddit_posts)
                        logger.info(f"Collected {len(reddit_posts)} Reddit posts")
                    except Exception as e:
                        logger.error(f"Reddit collection error: {e}")
                
                # Collect tweets
                if 'twitter' in self.collectors:
                    try:
                        tweets = await self.collectors['twitter'].collect_tweets(limit=100)
                        social_posts.extend(tweets)
                        logger.info(f"Collected {len(tweets)} tweets")
                    except Exception as e:
                        logger.error(f"Twitter collection error: {e}")
                
                # Collect news
                if 'news' in self.collectors:
                    try:
                        news = await self.collectors['news'].collect_news(limit=50)
                        news_articles.extend(news)
                        logger.info(f"Collected {len(news)} news articles")
                    except Exception as e:
                        logger.error(f"News collection error: {e}")
                
                # Aggregate sentiment
                if social_posts or news_articles:
                    sentiment_scores = await self.aggregator.aggregate_sentiment(
                        social_posts,
                        news_articles,
                        time_window=timedelta(hours=24)
                    )
                    
                    # Process and store results
                    await self._process_sentiment_scores(sentiment_scores)
                    
                    # Log summary
                    logger.info(f"Sentiment analysis complete for {len(sentiment_scores)} symbols")
                    for symbol, score in sentiment_scores.items():
                        logger.info(
                            f"{symbol}: sentiment={score.sentiment:.3f}, "
                            f"confidence={score.confidence:.2f}, volume={score.volume}"
                        )
                else:
                    logger.warning("No data collected in this cycle")
                
                # Calculate cycle duration
                cycle_duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Collection cycle completed in {cycle_duration:.1f} seconds")
                
                # Sleep until next cycle
                sleep_time = max(0, self.config['update_interval'] - cycle_duration)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _process_sentiment_scores(self, sentiment_scores: Dict):
        """Process and publish sentiment scores"""
        try:
            # Publish to Redis for real-time access
            pipeline = self.redis_client.pipeline()
            
            for symbol, score in sentiment_scores.items():
                # Store current sentiment
                key = f"sentiment:{symbol}"
                value = {
                    'sentiment': score.sentiment,
                    'confidence': score.confidence,
                    'volume': score.volume,
                    'timestamp': score.timestamp.isoformat(),
                    'sources': dict(score.metadata.get('sources', {}))
                }
                
                pipeline.setex(
                    key,
                    self.config['update_interval'] * 2,  # TTL = 2x update interval
                    json.dumps(value)
                )
                
                # Store in time series
                ts_key = f"sentiment:history:{symbol}"
                pipeline.zadd(
                    ts_key,
                    {json.dumps(value): score.timestamp.timestamp()}
                )
                
                # Trim old data (keep 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).timestamp()
                pipeline.zremrangebyscore(ts_key, 0, week_ago)
                
                # Publish event
                event = {
                    'type': 'sentiment_update',
                    'symbol': symbol,
                    'data': value
                }
                pipeline.publish('sentiment:updates', json.dumps(event))
            
            pipeline.execute()
            
            # Store in database for historical analysis
            await self._store_in_database(sentiment_scores)
            
        except Exception as e:
            logger.error(f"Error processing sentiment scores: {e}")
    
    async def _store_in_database(self, sentiment_scores: Dict):
        """Store sentiment scores in database"""
        # This would connect to your PostgreSQL/SQLite database
        # For now, just log that we would store it
        logger.debug(f"Would store {len(sentiment_scores)} sentiment scores in database")
    
    async def _health_check_loop(self):
        """Periodic health check"""
        while self.running:
            try:
                # Check Redis connection
                self.redis_client.ping()
                
                # Update health status
                health_data = {
                    'service': 'sentiment_analyzer',
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'collectors': list(self.collectors.keys()),
                    'symbols': self.config['symbols']
                }
                
                self.redis_client.setex(
                    'health:sentiment_analyzer',
                    60,  # 1 minute TTL
                    json.dumps(health_data)
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Clean up old sentiment data
                cleanup_time = datetime.now() - timedelta(days=30)
                logger.info(f"Cleaning up data older than {cleanup_time}")
                
                # Implement cleanup logic here
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def shutdown(self):
        """Gracefully shutdown the service"""
        logger.info("Shutting down Sentiment Analysis Service...")
        self.running = False
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Sentiment service shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False

async def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    # Create and start service
    service = SentimentService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
