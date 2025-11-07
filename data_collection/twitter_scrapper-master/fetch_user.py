#!/usr/bin/env python3
import asyncio
import sys
import os
from datetime import datetime
import json
import asyncio
import json
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

async def get_photo_image_url(username):
    url = f"https://x.com/{username}/photo"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector("img", timeout=10000)
            imgs = await page.query_selector_all("img")
            for img in imgs:
                src = await img.get_attribute("src")
                if src and "profile_images" in src:
                    await browser.close()
                    return src
        except PlaywrightTimeoutError:
            print(f"Timeout for {username}, skipping.")
        except Exception as e:
            print(f"Error for {username}: {e}")
        await browser.close()
        return None

# Fix Windows unicode output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import asyncio
import random
import hashlib
import time
import glob
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Set
from playwright.async_api import async_playwright, TimeoutError, Page
from pathlib import Path


COOKIES_FILE = os.path.join('./twitter_cookies.json')



# Rate limiting configuration
SCROLL_DELAY = 2  # Increased from 1 to 2 seconds between scrolls for better stability
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 2
TIMEOUT = 10000  # 10 seconds (increased from 3 seconds)

def parse_tweet_date(tweet_date_str: str) -> Optional[date]:
    """Parse tweet date string to date object for comparison."""
    if not tweet_date_str or tweet_date_str == "Unknown":
        return None
    
    try:
        # Common Twitter date formats
        date_formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format: 2024-01-15T10:30:45.123Z
            '%Y-%m-%dT%H:%M:%SZ',     # ISO format without microseconds
            '%Y-%m-%d %H:%M:%S',      # Simple format
            '%Y-%m-%d',               # Date only
            '%b %d, %Y',              # Jan 15, 2024
            '%B %d, %Y',              # January 15, 2024
            '%d %b %Y',               # 15 Jan 2024
            '%m/%d/%Y',               # 01/15/2024
        ]
        
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(tweet_date_str, date_format).date()
                return parsed_date
            except ValueError:
                continue
        
        # If none of the formats work, try to extract just the date part
        # Handle formats like "Jan 15" (assume current year)
        try:
            current_year = datetime.now().year
            parsed_date = datetime.strptime(f"{tweet_date_str} {current_year}", '%b %d %Y').date()
            return parsed_date
        except ValueError:
            pass
            
        print(f"Could not parse date format: {tweet_date_str}")
        return None
        
    except Exception as e:
        print(f"Error parsing date '{tweet_date_str}': {str(e)}")
        return None

def should_stop_at_date(tweet_date_str: str, stop_date) -> bool:
    """Check if we should stop scraping based on tweet date."""
    if not stop_date or not tweet_date_str:
        return False
    
    tweet_date = parse_tweet_date(tweet_date_str)
    if not tweet_date:
        return False
    
    # Stop if tweet date is on or before the stop date
    return tweet_date <= stop_date

async def safe_wait_for_selector(page: Page, selector: str, timeout: int = TIMEOUT, description: str = "element") -> bool:
    """Safely wait for a selector with proper error handling."""
    try:
        # Use a more reasonable timeout conversion
        timeout_seconds = max(timeout/1000, 10)  # At least 10 seconds
        await asyncio.wait_for(
            page.wait_for_selector(selector, timeout=timeout, state="attached"),
            timeout=timeout_seconds + 5  # Add 5 second buffer
        )
        return True
    except asyncio.TimeoutError:
        print(f"Timeout waiting for {description} (selector: {selector}) - {timeout}ms")
        return False
    except TimeoutError:
        print(f"Playwright timeout waiting for {description} (selector: {selector}) - {timeout}ms")
        return False
    except Exception as e:
        print(f"Error waiting for {description}: {str(e)}")
        return False

async def safe_browser_close(browser):
    """Safely close browser with timeout."""
    try:
        await asyncio.wait_for(browser.close(), timeout=5)
        print("Browser closed successfully")
    except asyncio.TimeoutError:
        print("Browser close timeout - force terminating")
    except Exception as e:
        print(f"Error closing browser: {str(e)}")

async def rate_limit_delay(delay: float = REQUEST_DELAY) -> None:
    """Add delay for rate limiting."""
    await asyncio.sleep(delay)

async def wait_for_profile_load(page: Page, username: str) -> bool:
    """Wait for profile to load with multiple fallback strategies."""
    try:
        # Strategy 1: Wait for tweets
        if await safe_wait_for_selector(page, 'article[data-testid="tweet"]', timeout=3000, description="tweets"):
            return True
        
        # Strategy 2: Wait for empty state
        if await safe_wait_for_selector(page, 'div[data-testid="emptyState"]', timeout=2000, description="empty state"):
            print(f"Profile {username} has no tweets or is empty")
            return True
            
        # Strategy 3: Check if profile header exists (private or protected account)
        if await safe_wait_for_selector(page, 'div[data-testid="UserName"]', timeout=3000, description="profile header"):
            print(f"Profile {username} loaded but may be private")
            return True
            
        print(f"Timeout waiting for profile {username} to load")
        return False
    except Exception as e:
        print(f"Error waiting for profile load: {str(e)}")
        return False

async def scrape_user_profile(page: Page, username: str) -> Dict[str, str]:
    """Scrape user profile information with improved error handling."""
    try:
        print(f"Navigating to profile page for @{username}...")
        await page.goto(f"https://twitter.com/{username}", wait_until="domcontentloaded", timeout=TIMEOUT)
        await rate_limit_delay()
        
        if not await safe_wait_for_selector(page, 'div[data-testid="UserName"]', description="profile"):
            print(f"Could not load profile for @{username}")
            return {"username": username, "bio": ""}
        
        # Get display name with multiple selector fallbacks
        display_name = ""
        name_selectors = [
            'div[data-testid="UserName"] span',
            'div[data-testid="UserName"] div span',
            'h1[data-testid="UserName"] span'
        ]
        
        for selector in name_selectors:
            try:
                name_element = page.locator(selector).first
                if await name_element.count() > 0:
                    display_name = await name_element.inner_text()
                    if display_name:
                        break
            except Exception as e:
                print(f"Error with name selector {selector}: {str(e)}")
                continue
        
        # Get bio with retry and multiple selectors
        bio = ""
        bio_selectors = [
            'div[data-testid="UserDescription"]',
            'div[data-testid="UserBio"]',
            'div[data-testid="UserProfessionalCategory"]'
        ]
        
        for attempt in range(MAX_RETRIES):
            for selector in bio_selectors:
                try:
                    bio_element = page.locator(selector)
                    if await bio_element.count() > 0:
                        bio = await bio_element.inner_text()
                        if bio:
                            break
                except Exception as e:
                    print(f"Error with bio selector {selector}: {str(e)}")
                    continue
            
            if bio:
                break
            await rate_limit_delay()
        
        return {
            "username": display_name or username,
            "bio": bio.strip() if bio else ""
        }

    except Exception as e:
        print(f"Error scraping profile: {str(e)}")
        return {"username": username, "bio": ""}

async def get_quoted_tweet_info(tweet_element) -> Optional[Dict[str, str]]:
    """Get information about a quoted tweet if present"""
    try:
        quoted_container = tweet_element.locator('div:has(> div[data-testid="tweet"])').last
        if await quoted_container.count() > 0:
            quoted_text = await quoted_container.locator('div[data-testid="tweetText"]').inner_text()
            name_element = quoted_container.locator('div[data-testid="User-Name"] div span').first
            quoted_username = await name_element.inner_text() if await name_element.count() > 0 else ""
            return {
                "quoted_content": quoted_text,
                "quoted_username": quoted_username
            }
    except Exception as e:
        print(f"Could not get quoted tweet info: {str(e)}")
    return None

async def get_main_tweet_content(tweet_element) -> str:
    """Get the main tweet content with improved selector robustness and timeout protection."""
    content_selectors = [
        'div[data-testid="tweetText"]',
        'div[lang]:not([data-testid])',
        'article div[lang]',
        'div[role="article"] div[lang]',
        'div[dir="auto"]:not([data-testid])',
        'span[lang]'
    ]
    
    for selector in content_selectors:
        try:
            elements = tweet_element.locator(selector)
            count = await asyncio.wait_for(elements.count(), timeout=3)
            if count > 0:
                texts = []
                for i in range(min(count, 3)):  # Limit to first 3 elements to avoid hanging
                    try:
                        element = elements.nth(i)
                        text = await asyncio.wait_for(element.inner_text(), timeout=2)
                        if text and len(text.strip()) > 0:
                            texts.append(text.strip())
                    except (asyncio.TimeoutError, Exception):
                        continue
                
                if texts:
                    return " ".join(texts)
        except (asyncio.TimeoutError, Exception):
            continue
    
    # Final fallback with timeout
    try:
        text = await asyncio.wait_for(tweet_element.inner_text(), timeout=3)
        if text and len(text.strip()) > 10:  # Avoid getting just metadata
            return text.strip()[:500]  # Limit length
    except (asyncio.TimeoutError, Exception):
        pass
    
    return ""

async def get_tweet_date(tweet_element) -> str:
    """Extract the date/time when the tweet was posted with timeout protection."""
    try:
        # Method 1: Try to get datetime from time element with timeout
        time_element = tweet_element.locator('time').first
        count = await asyncio.wait_for(time_element.count(), timeout=2)
        if count > 0:
            datetime_attr = await asyncio.wait_for(time_element.get_attribute('datetime'), timeout=2)
            if datetime_attr:
                return datetime_attr
    except (asyncio.TimeoutError, Exception):
        pass
    
    # Method 2: Try to get from title attribute of time element with timeout
    try:
        time_element = tweet_element.locator('time').first
        count = await asyncio.wait_for(time_element.count(), timeout=2)
        if count > 0:
            title_attr = await asyncio.wait_for(time_element.get_attribute('title'), timeout=2)
            if title_attr:
                return title_attr
    except (asyncio.TimeoutError, Exception):
        pass
    
    # Method 3: Try to get text content from time element with timeout
    try:
        time_element = tweet_element.locator('time').first
        count = await asyncio.wait_for(time_element.count(), timeout=2)
        if count > 0:
            time_text = await asyncio.wait_for(time_element.inner_text(), timeout=2)
            if time_text:
                return time_text.strip()
    except (asyncio.TimeoutError, Exception):
        pass
    
    # Method 4: Look for any timestamp in the tweet
    try:
        # Look for common date/time patterns in the tweet
        timestamp_selectors = [
            'a[href*="/status/"] time',
            'time[datetime]',
            '[data-testid*="time"]',
            '[aria-label*="time"]',
            '[title*="AM"]',
            '[title*="PM"]'
        ]
        
        for selector in timestamp_selectors:
            try:
                element = tweet_element.locator(selector).first
                if await element.count() > 0:
                    # Try datetime attribute first
                    datetime_attr = await element.get_attribute('datetime')
                    if datetime_attr:
                        return datetime_attr
                    
                    # Try title attribute
                    title_attr = await element.get_attribute('title')
                    if title_attr:
                        return title_attr
                    
                    # Try text content
                    text = await element.inner_text()
                    if text:
                        return text.strip()
            except Exception:
                continue
    except Exception:
        pass
    
    return ""

async def get_tweet_id(tweet_element) -> str:
    """Get a unique identifier for a tweet with improved robustness."""
    # Method 1: Try to get tweet status ID from URL
    try:
        links = tweet_element.locator('a[href*="/status/"]')
        count = await links.count()
        for i in range(count):
            try:
                href = await links.nth(i).get_attribute('href')
                if href and "/status/" in href:
                    tweet_id = href.split('/status/')[-1].split('?')[0]
                    if tweet_id.isdigit() and len(tweet_id) > 10:
                        return tweet_id
            except Exception:
                continue
    except Exception:
        pass
    
    # Method 2: Try to get datetime from time element
    try:
        time_element = tweet_element.locator('time').first
        if await time_element.count() > 0:
            datetime_attr = await time_element.get_attribute('datetime')
            if datetime_attr:
                return f"time_{datetime_attr}"
    except Exception:
        pass
    
    # Method 3: Try to get data attributes
    try:
        data_attrs = ['data-tweet-id', 'data-testid', 'data-item-id']
        for attr in data_attrs:
            value = await tweet_element.get_attribute(attr)
            if value:
                return f"{attr}_{value}"
    except Exception:
        pass
    
    # Method 4: Generate from content hash
    try:
        # Get a small portion of HTML for hashing (avoid memory issues)
        html_snippet = await tweet_element.inner_html()
        if html_snippet:
            # Take first 1000 chars to avoid memory issues
            content_hash = hashlib.md5(html_snippet[:1000].encode()).hexdigest()
            return f"hash_{content_hash}"
    except Exception:
        pass
    
    # Method 5: Last resort - use element position and content snippet hash
    try:
        # Create a more stable ID based on element properties
        element_text = await tweet_element.inner_text()
        if element_text and len(element_text.strip()) > 5:
            # Use first 100 chars of text to create stable hash
            text_snippet = element_text.strip()[:100]
            content_hash = hashlib.md5(text_snippet.encode()).hexdigest()[:12]
            return f"stable_{content_hash}"
    except Exception:
        pass
    
    # If we reach here, the element probably has no useful content
    # Return None to indicate this element should be skipped
    return None

async def is_repost(tweet_element) -> bool:
    """Check if tweet is a repost (retweet without comment) with improved detection."""
    try:
        # Method 1: Check for retweet indicator in social context
        social_selectors = [
            'div[data-testid="socialContext"]',
            'div[data-testid="tweet"] div[data-testid="socialContext"]'
        ]
        
        for selector in social_selectors:
            try:
                social_context = tweet_element.locator(selector)
                if await social_context.count() > 0:
                    social_text = await social_context.inner_text()
                    retweet_indicators = ["reposted", "retweeted", "retweet", "shared"]
                    if any(indicator in social_text.lower() for indicator in retweet_indicators):
                        print(f"Found retweet via social context: {social_text}")
                        return True
            except Exception:
                continue

        # Method 2: Check for retweet icon/action
        retweet_indicators = [
            'div[data-testid="retweetIcon"]',
            'div[data-testid="retweet"]',
            'div[aria-label*="retweet" i]',
            'div[aria-label*="repost" i]',
            'svg[data-testid="icon-retweet"]'
        ]
        
        for indicator in retweet_indicators:
            try:
                element = tweet_element.locator(indicator)
                if await element.count() > 0:
                    print(f"Found retweet via indicator: {indicator}")
                    return True
            except Exception:
                continue

        # Method 3: Check for retweet text patterns
        try:
            article_text = await tweet_element.inner_text()
            if article_text:
                retweet_phrases = [
                    "retweeted", "reposted", "retweet", "shared this",
                    "reposted this", "retweeted this", "shared a"
                ]
                for phrase in retweet_phrases:
                    if phrase in article_text.lower():
                        print(f"Found retweet via text pattern: {phrase}")
                        return True
        except Exception:
            pass

        # Method 4: Check for nested tweet structure
        try:
            # Look for quoted or nested tweets
            nested_selectors = [
                'div[data-testid="tweet"] div[data-testid="tweet"]',
                'article div[data-testid="tweet"]'
            ]
            
            for selector in nested_selectors:
                nested_tweet = tweet_element.locator(selector)
                if await nested_tweet.count() > 0:
                    print("Found retweet via nested structure")
                    return True
        except Exception:
            pass

    except Exception as e:
        print(f"Error checking repost status: {str(e)}")
    
    return False

async def get_retweet_info(tweet_element, page: Page) -> Optional[Dict[str, str]]:
    """Get information about a retweet WITHOUT bio fetching (to prevent hanging)."""
    try:
        # Get the original tweet content
        main_content = await get_main_tweet_content(tweet_element)
        
        # Get username of original tweet author
        username = ""
        username_selectors = [
            'div[data-testid="User-Name"] a',
            'div[data-testid="User-Name"] span:has-text("@")',
            'a[href*="/"]:not([href*="/status/"])'
        ]
        
        for selector in username_selectors:
            try:
                # Add timeout protection for element counting and access
                elements = tweet_element.locator(selector)
                count = await asyncio.wait_for(elements.count(), timeout=3)
                
                for i in range(min(count, 5)):  # Limit to first 5 elements
                    try:
                        element = elements.nth(i)
                        
                        # Try to get username from href with timeout
                        try:
                            href = await asyncio.wait_for(element.get_attribute('href'), timeout=2)
                            if href and '/' in href and not '/status/' in href:
                                potential_username = href.strip('/').split('/')[-1]
                                if potential_username and not potential_username.startswith('http'):
                                    username = potential_username
                                    break
                        except (asyncio.TimeoutError, Exception):
                            pass
                        
                        # Try to get username from text content with timeout
                        try:
                            text = await asyncio.wait_for(element.inner_text(), timeout=2)
                            if text and '@' in text:
                                username = text.strip('@').split()[0]
                                break
                        except (asyncio.TimeoutError, Exception):
                            pass
                            
                    except Exception:
                        continue
                        
                if username:
                    break
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Timeout/error in username extraction with selector {selector}: {str(e)}")
                continue

        # NO BIO FETCHING - leave empty as requested
        bio = ""
        print(f"Skipped bio fetching for @{username} (disabled to prevent hanging)")

        return {
            "retweet_content": "",  # Pure retweets have no additional content
            "retweet_username": username,
            "retweet_profile_bio": bio,  # Always empty - no bio fetching
            "retweet_main_content": main_content
        }

    except Exception as e:
        print(f"Could not get retweet info: {str(e)}")
        return None

async def scrape_tweets(page: Page, username: str, max_tweets: int = 100, max_retweets: int = 100, stop_date=None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Scrape tweets and retweets with improved efficiency and error handling."""
    tweets = []
    retweets = []
    processed_ids: Set[str] = set()
    
    # Initialize scroll_attempts at the beginning to avoid variable scope issues
    scroll_attempts = 0
    max_scroll_attempts = 50
    
    try:
        print(f"\nStarting to scrape tweets for user: {username} (max {max_tweets} tweets, {max_retweets} retweets)")
        if stop_date:
            print(f"🗓️  Will also stop when reaching tweets from {stop_date} or earlier")
        else:
            print("📅 No stop date set - will only stop at tweet limit or end of timeline")
        
        # Screenshots disabled for better performance
        
        # Navigate to profile
        await page.goto(f"https://twitter.com/{username}", wait_until="domcontentloaded", timeout=TIMEOUT)
        
        if not await wait_for_profile_load(page, username):
            print("Profile could not be loaded")
            return tweets, retweets

        print("Profile loaded successfully")
        
        # Scrolling variables
        last_height = await page.evaluate("document.body.scrollHeight")
        no_new_items_count = 0
        max_no_new_items = 3  # Restored from 2 to 3

        while scroll_attempts < max_scroll_attempts:
            scroll_attempts += 1
            
            try:
                # Wait for content to load
                await rate_limit_delay(SCROLL_DELAY)
                
                # Get all visible tweets with timeout protection
                try:
                    tweet_elements = await asyncio.wait_for(
                        page.locator('article[data-testid="tweet"]').all(),
                        timeout=5
                    )
                    print(f"[DEBUG] Found {len(tweet_elements)} tweet elements on scroll {scroll_attempts}")
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"[DEBUG] Timeout/error getting tweet elements on scroll {scroll_attempts}: {str(e)}")
                    tweet_elements = []
                
                # If no tweets found, wait and try again before giving up
                if not tweet_elements:
                    print(f"No tweets found on attempt {scroll_attempts}, waiting and retrying...")
                    await asyncio.sleep(3)  # Wait longer
                    
                    # Try a gentle scroll method instead of aggressive scrolling
                    try:
                        # Use smaller, gentler scroll
                        await asyncio.wait_for(
                            page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)"),
                            timeout=3
                        )
                        await asyncio.sleep(2)
                        
                        # Try again with timeout
                        tweet_elements = await asyncio.wait_for(
                            page.locator('article[data-testid="tweet"]').all(),
                            timeout=5
                        )
                        print(f"[DEBUG] After gentle retry: Found {len(tweet_elements)} tweet elements")
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"[DEBUG] Retry scroll/detection timeout: {str(e)}")
                        tweet_elements = []
                    
                    if not tweet_elements:
                        print("Still no tweets found, but continuing...")
                        no_new_items_count += 1
                        # Don't break here, continue to scroll more
                        if no_new_items_count >= max_no_new_items:
                            print("Multiple attempts failed, ending tweets scraping")
                            break
                        continue

                initial_count = len(tweets) + len(retweets)
                processed_in_batch = 0
                visible_tweet_ids = []  # Track tweets visible in current scroll

                for idx, tweet in enumerate(tweet_elements):
                    try:
                        # Check if tweet is actually visible in viewport
                        try:
                            is_visible = await asyncio.wait_for(
                                tweet.is_visible(),
                                timeout=1
                            )
                            if not is_visible:
                                continue  # Skip invisible tweets
                        except:
                            pass  # Continue processing if visibility check fails
                        
                        # Get unique tweet ID
                        tweet_id = await get_tweet_id(tweet)
                        
                        # Skip elements with no valid ID (probably no useful content)
                        if tweet_id is None:
                            continue
                        
                        # Track this tweet as visible in current batch
                        visible_tweet_ids.append(tweet_id)
                        
                        if tweet_id in processed_ids:
                            continue
                        
                        processed_ids.add(tweet_id)
                        processed_in_batch += 1
                        
                        # Check if it's a retweet only if we want retweets
                        is_retweet = False
                        if max_retweets > 0:
                            is_retweet = await is_repost(tweet)
                        
                        if is_retweet and max_retweets > 0:
                            # Check retweet limit
                            if len(retweets) >= max_retweets:
                                print(f"Reached maximum retweets limit ({max_retweets}), skipping further retweets")
                                continue
                                
                            # Screenshots completely removed to improve performance
                            print(f"Processing retweet #{len(retweets)+1} (ID: {tweet_id})")
                            
                            # Get retweet date with timeout
                            try:
                                print(f"Getting date for retweet {tweet_id}...")
                                retweet_date = await asyncio.wait_for(
                                    get_tweet_date(tweet),
                                    timeout=3
                                )
                                print(f"Date extracted for retweet {tweet_id}: {retweet_date}")
                            except asyncio.TimeoutError:
                                print(f"Timeout getting date for retweet {tweet_id}")
                                retweet_date = "Unknown"
                            except Exception as e:
                                print(f"Error getting date for retweet {tweet_id}: {str(e)}")
                                retweet_date = "Unknown"

                            # Get retweet info with timeout and debugging
                            try:
                                print(f"Getting retweet info for {tweet_id}...")
                                retweet_info = await asyncio.wait_for(
                                    get_retweet_info(tweet, page),
                                    timeout=15  # Longer timeout as this includes bio fetching
                                )
                                print(f"Retweet info extracted for {tweet_id}")
                            except asyncio.TimeoutError:
                                print(f"Timeout getting retweet info for {tweet_id}")
                                retweet_info = None
                            except Exception as e:
                                print(f"Error getting retweet info for {tweet_id}: {str(e)}")
                                retweet_info = None
                            
                            if retweet_info:
                                retweet_info["retweet_date"] = retweet_date
                                # Screenshots completely removed for better performance
                                retweets.append(retweet_info)
                                print(f"Successfully added retweet {len(retweets)} with date: {retweet_date}")
                            else:
                                print(f"Could not extract retweet info for {tweet_id}")
                            continue

                        # Check tweet limit
                        if len(tweets) >= max_tweets:
                            print(f"Reached maximum tweets limit ({max_tweets}), skipping further tweets")
                            continue

                        # Process as regular tweet with simpler handling
                        print(f"Processing tweet #{len(tweets)+1} (ID: {tweet_id})")
                        
                        # Get content with timeout protection
                        try:
                            print(f"Getting content for tweet {tweet_id}...")
                            content = await asyncio.wait_for(
                                get_main_tweet_content(tweet),
                                timeout=5
                            )
                            print(f"Content extracted for tweet {tweet_id}: {len(content) if content else 0} chars")
                        except asyncio.TimeoutError:
                            print(f"Timeout getting content for tweet {tweet_id}")
                            content = None
                        except Exception as e:
                            print(f"Error getting content for tweet {tweet_id}: {str(e)}")
                            content = None
                        
                        if content:
                            # Get tweet date with timeout
                            try:
                                print(f"Getting date for tweet {tweet_id}...")
                                tweet_date = await asyncio.wait_for(
                                    get_tweet_date(tweet),
                                    timeout=3
                                )
                                print(f"Date extracted for tweet {tweet_id}: {tweet_date}")
                            except asyncio.TimeoutError:
                                print(f"Timeout getting date for tweet {tweet_id}")
                                tweet_date = "Unknown"
                            except Exception as e:
                                print(f"Error getting date for tweet {tweet_id}: {str(e)}")
                                tweet_date = "Unknown"
                            
                            # Check if we should stop based on date
                            if should_stop_at_date(tweet_date, stop_date):
                                print(f"Stopping scraping with {len(tweets)} tweets collected.")
                                return tweets, retweets
                            
                            # Screenshots completely removed for better performance
                            tweet_data = {
                                "tweet_content": content,
                                "tweet_date": tweet_date
                            }

                            # Check for quoted tweet
                            try:
                                quoted_info = await get_quoted_tweet_info(tweet)
                                if quoted_info:
                                    tweet_data.update(quoted_info)
                            except Exception as e:
                                print(f"Error getting quoted tweet info: {str(e)}")

                            tweets.append(tweet_data)
                            print(f"Successfully added tweet {len(tweets)} with date: {tweet_date}")
                        else:
                            # Handle tweets without detectable content
                            print(f"Tweet {tweet_id} has no detectable content, skipping")

                    except Exception as e:
                        print(f"Error processing tweet element: {str(e)}")
                        continue

                # Check progress
                current_count = len(tweets) + len(retweets)
                print(f"Batch {scroll_attempts}: Processed {processed_in_batch} new items. Total: {len(tweets)} tweets, {len(retweets)} retweets")
                print(f"[DEBUG] Visible tweets in this batch: {len(visible_tweet_ids)}, New processed: {processed_in_batch}")
                
                # Additional debugging to track potential missed tweets
                if len(visible_tweet_ids) > processed_in_batch + 3:  # More than 3 already-processed tweets
                    print(f"[WARNING] High overlap detected: {len(visible_tweet_ids)} visible, {processed_in_batch} new. May indicate fast scrolling.")
                
                # Check if we've reached the limits
                tweets_limit_reached = len(tweets) >= max_tweets
                retweets_limit_reached = (max_retweets == 0) or (len(retweets) >= max_retweets)
                
                if tweets_limit_reached and retweets_limit_reached:
                    if max_retweets > 0:
                        print(f"Reached both limits: {len(tweets)} tweets (max: {max_tweets}), {len(retweets)} retweets (max: {max_retweets})")
                    else:
                        print(f"Reached tweet limit: {len(tweets)} tweets (max: {max_tweets}), retweets disabled")
                    break
                
                if current_count == initial_count:
                    no_new_items_count += 1
                    print(f"No new items found (attempt {no_new_items_count}/{max_no_new_items})")
                    
                    # If we had visible tweets but didn't process new ones, we might be scrolling too fast
                    if len(visible_tweet_ids) > 5:  # Many tweets visible but none new
                        print(f"[DEBUG] Many tweets visible ({len(visible_tweet_ids)}) but none new - possible fast scrolling issue")
                        # Wait a bit longer for content to settle
                        await asyncio.sleep(1)
                else:
                    no_new_items_count = 0

                if no_new_items_count >= max_no_new_items:
                    print("Reached end of timeline (no new items)")
                    break

                # Improved gentle scrolling to avoid missing tweets
                print(f"Scrolling for more content... (attempt {scroll_attempts})")
                
                try:
                    # Get current scroll position
                    current_scroll = await asyncio.wait_for(
                        page.evaluate("window.pageYOffset"),
                        timeout=3
                    )
                    
                    # Gentle incremental scrolling - scroll by smaller chunks
                    scroll_distance = min(
                        await asyncio.wait_for(page.evaluate("window.innerHeight"), timeout=3),
                        800  # Maximum 800px per scroll to avoid missing content
                    )
                    
                    # Perform 3 small scrolls instead of one big jump
                    for i in range(3):
                        await asyncio.wait_for(
                            page.evaluate(f"window.scrollBy(0, {scroll_distance // 3})"),
                            timeout=3
                        )
                        await asyncio.sleep(0.5)  # Short wait between micro-scrolls
                        
                        # Check for new tweets after each micro-scroll
                        try:
                            new_tweet_count = await asyncio.wait_for(
                                page.locator('article[data-testid="tweet"]').count(),
                                timeout=2
                            )
                            if new_tweet_count > len(tweet_elements):
                                print(f"[DEBUG] New tweets loaded during micro-scroll {i+1}: {new_tweet_count} vs {len(tweet_elements)}")
                                break  # Exit micro-scroll loop if new content appears
                        except:
                            pass
                    
                    # Wait for dynamic content to load - adaptive timing based on success
                    if processed_in_batch > 0:
                        # If we're finding new content, wait longer to ensure we don't miss anything
                        await asyncio.sleep(2.0)  # Longer wait when successfully collecting
                        print(f"[DEBUG] Extended wait due to successful collection ({processed_in_batch} new items)")
                    else:
                        # Standard wait when not finding much
                        await asyncio.sleep(1.5)
                    
                    # Check if we actually scrolled
                    new_scroll = await asyncio.wait_for(
                        page.evaluate("window.pageYOffset"),
                        timeout=3
                    )
                    
                    if abs(new_scroll - current_scroll) < 50:  # Less than 50px scrolled
                        print("[DEBUG] Small scroll detected, trying alternative method")
                        # If normal scroll didn't work, try keyboard scroll as fallback
                        try:
                            await asyncio.wait_for(page.keyboard.press('PageDown'), timeout=2)
                            await asyncio.sleep(1)
                        except (asyncio.TimeoutError, Exception):
                            pass
                    
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"Scroll operation timeout/error: {str(e)}")
                
                await rate_limit_delay(SCROLL_DELAY)

                # Check if page height changed (with timeout)
                try:
                    new_height = await asyncio.wait_for(
                        page.evaluate("document.body.scrollHeight"),
                        timeout=3
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"Page height check timeout: {str(e)}")
                    new_height = last_height  # Assume no change
                
                if new_height == last_height:
                    # Try one more aggressive scroll before giving up (with timeout)
                    try:
                        await asyncio.wait_for(
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight + 1000)"),
                            timeout=3
                        )
                        await asyncio.sleep(2)
                        new_height = await asyncio.wait_for(
                            page.evaluate("document.body.scrollHeight"),
                            timeout=3
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        print(f"Aggressive scroll timeout: {str(e)}")
                        new_height = last_height
                    
                    if new_height == last_height:
                        no_new_items_count += 1
                        print("Page height unchanged, may have reached end")
                    else:
                        last_height = new_height
                        print("Additional scroll worked, continuing...")
                else:
                    last_height = new_height

            except Exception as e:
                print(f"Error during scroll {scroll_attempts}: {str(e)}")
                no_new_items_count += 1

            # Emergency break conditions
            if no_new_items_count >= max_no_new_items:
                print("Stopping due to no new items")
                break

    except Exception as e:
        print(f"Error scraping tweets: {str(e)}")
        # Ensure scroll_attempts is initialized even in case of early error
        if 'scroll_attempts' not in locals():
            scroll_attempts = 0

    print(f"\nScraping completed after {scroll_attempts} scroll attempts!")
    print(f"Final results: {len(tweets)} tweets and {len(retweets)} retweets")
    return tweets, retweets

async def scrape_social_users(page: Page, username: str, user_type: str, max_users: int = 300) -> List[Dict[str, str]]:
    """Generic function to scrape followers or following with improved efficiency."""
    users = []
    try:
        # Navigate to the appropriate page
        url = f"https://twitter.com/{username}/{user_type}"
        print(f"Navigating to {url} (max: {max_users} users)")
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
        await rate_limit_delay()

        # Wait for content to load
        if not await safe_wait_for_selector(page, 'div[data-testid="cellInnerDiv"]', timeout=3000, description=f"{user_type} cells"):
            print(f"No {user_type} cells found")
            return users

        # Initialize tracking variables
        processed_usernames: Set[str] = set()
        no_new_users_count = 0
        max_no_new_users = 5  # Increased to get more followers/following
        scroll_attempts = 0
        max_scroll_attempts = 30

        while scroll_attempts < max_scroll_attempts:
            scroll_attempts += 1
            try:
                # Get all visible user cells
                cells = await page.locator('div[data-testid="cellInnerDiv"]').all()
                
                if not cells:
                    no_new_users_count += 1
                    if no_new_users_count >= max_no_new_users:
                        print(f"No more {user_type} cells found after multiple attempts")
                        break
                    await rate_limit_delay()
                    continue

                initial_count = len(users)
                processed_in_batch = 0

                for cell in cells:
                    try:
                        # Extract username
                        cell_username = await extract_username_from_cell(cell)
                        if not cell_username or cell_username in processed_usernames:
                            continue
                        
                        # Check if we've reached the user limit
                        if len(users) >= max_users:
                            print(f"Reached maximum {user_type} limit ({max_users}), stopping collection")
                            break
                            
                        processed_usernames.add(cell_username)
                        processed_in_batch += 1
                        
                        # Extract display name
                        display_name = await extract_display_name_from_cell(cell, cell_username)
                        
                        # Extract bio
                        bio = await extract_bio_from_cell(cell)
                        
                        # Create user data based on type
                        if user_type == "followers":
                            user_data = {
                                "follower_name": display_name or cell_username,
                                "follower_bio": bio
                            }
                        else:  # following
                            user_data = {
                                "following_name": display_name or cell_username,
                                "following_bio": bio
                            }
                        
                        users.append(user_data)
                        print(f"Added {user_type[:-1]} #{len(users)}: @{cell_username}" + (f" ({display_name})" if display_name else ""))
                        
                        if bio:
                            print(f"  Bio: {bio[:50]}..." if len(bio) > 50 else f"  Bio: {bio}")
                        
                    except Exception as e:
                        print(f"Error processing {user_type} cell: {str(e)}")
                        continue

                # Check progress and limits
                current_count = len(users)
                print(f"Batch {scroll_attempts}: Processed {processed_in_batch} new {user_type}. Total: {current_count}")
                
                # Check if we've reached the user limit
                if len(users) >= max_users:
                    print(f"Reached maximum {user_type} limit ({max_users}), stopping")
                    break
                
                if current_count == initial_count:
                    no_new_users_count += 1
                    print(f"No new {user_type} found (attempt {no_new_users_count}/{max_no_new_users})")
                else:
                    no_new_users_count = 0

                if no_new_users_count >= max_no_new_users:
                    print(f"Reached end of {user_type} list")
                    break

                # Scroll down
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await rate_limit_delay(SCROLL_DELAY)

            except Exception as e:
                print(f"Error during {user_type} scroll {scroll_attempts}: {str(e)}")
                no_new_users_count += 1

        print(f"Total {user_type} collected: {len(users)}")
        return users

    except Exception as e:
        print(f"Error scraping {user_type}: {str(e)}")
        return users

async def extract_username_from_cell(cell) -> str:
    """Extract username from a user cell."""
    username_selectors = [
        'a[role="link"][href*="/"]',
        'a[href*="/"]',
        'div[data-testid="User-Name"] a'
    ]
    
    for selector in username_selectors:
        try:
            links = cell.locator(selector)
            count = await links.count()
            for i in range(count):
                href = await links.nth(i).get_attribute('href')
                if href and '/' in href and not '/status/' in href:
                    potential_username = href.strip('/').split('/')[-1]
                    if potential_username and not potential_username.startswith('http'):
                        return potential_username
        except Exception:
            continue
    
    return ""

async def extract_display_name_from_cell(cell, username: str) -> str:
    """Extract display name from a user cell."""
    name_selectors = [
        'div[data-testid="User-Name"] div:first-child span span',
        'div[data-testid="User-Name"] div span',
        'div[data-testid="User-Name"] span'
    ]
    
    for selector in name_selectors:
        try:
            name_element = cell.locator(selector).first
            if await name_element.count() > 0:
                name = await name_element.inner_text()
                if name and name.strip() and not name.startswith('@'):
                    # Clean up name
                    name = name.strip().replace("·", "").strip()
                    if name != username:  # Don't return username as display name
                        return name
        except Exception:
            continue
    
    return ""

async def extract_bio_from_cell(cell) -> str:
    """Extract bio from a user cell."""
    bio_selectors = [
        'div[data-testid="UserDescription"]',
        'div[data-testid="UserBio"]',
        'div[data-testid="UserProfessionalCategory"]'
    ]
    
    for selector in bio_selectors:
        try:
            bio_element = cell.locator(selector)
            if await bio_element.count() > 0:
                bio = await bio_element.inner_text()
                if bio:
                    return bio.strip().replace("Follow", "").strip()
        except Exception:
            continue
    
    # Fallback: try to find any descriptive text
    try:
        text_elements = cell.locator('div[dir="auto"]')
        count = await text_elements.count()
        for i in range(count):
            text = await text_elements.nth(i).inner_text()
            if text and len(text) > 5 and not text.startswith('@') and 'Follow' not in text:
                return text.strip()
    except Exception:
        pass
    
    return ""

async def scrape_followers(page: Page, username: str, max_followers: int = 300) -> List[Dict[str, str]]:
    """Scrape followers using the generic social scraping function."""
    return await scrape_social_users(page, username, "followers", max_followers)

async def scrape_following(page: Page, username: str, max_following: int = 300) -> List[Dict[str, str]]:
    """Scrape following using the generic social scraping function."""
    return await scrape_social_users(page, username, "following", max_following)

async def scrape_twitter(username: str, max_tweets: int = 100, max_retweets: int = 100, max_followers: int = 1000, max_following: int = 1000, stop_date=None) -> Dict:
    result = {
        "user_profile": {"username": username, "bio": ""},
        # "following": [],
        # "followers": []
    }
    
    try:
        async with async_playwright() as p:
            # Detect if we have a display available
            has_display = os.environ.get('DISPLAY') is not None
            
            # Launch browser with appropriate settings
            launch_args = [
                '--disable-extensions',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-gpu' if not has_display else '',
            ]
            # Remove empty strings
            launch_args = [arg for arg in launch_args if arg]
            
            print(f"🖥️  Display available: {has_display} (DISPLAY={os.environ.get('DISPLAY', 'None')})")
            print(f"🚀 Browser args: {launch_args}")
            
            browser = await p.chromium.launch(
                headless=not has_display,  # Headless if no display, headed if display available
                args=launch_args
            )
            
            # Create context with larger viewport and modern user agent
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
            )
            
            # Load cookies
            if os.path.exists(COOKIES_FILE):
                try:
                    with open(COOKIES_FILE, "r") as f:
                        cookies = json.load(f)
                    await context.add_cookies(cookies)
                    print("Cookies loaded successfully")
                except Exception as e:
                    print(f"Error loading cookies: {str(e)}")
                    await browser.close()
                    return result
            else:
                print("No cookies file found. Please run login_manual.py first")
                await browser.close()
                return result
            
            try:
                # Create main page for profile info
                page = await context.new_page()
                page.set_default_timeout(60000)  # Set to 60 seconds instead of 30
                
                # First try to access Twitter directly
                print("\nAccessing Twitter...")
                try:
                    await page.goto("https://twitter.com", wait_until="domcontentloaded")
                except Exception as e:
                    print(f"Error accessing Twitter: {str(e)}")
                    await browser.close()
                    return result

                # Short wait for initial load
                await asyncio.sleep(1)
                
                # Check for login state using multiple indicators
                print("Verifying login status...")
                try:
                    # Wait a bit longer for the page to load completely
                    await asyncio.sleep(1)
                    
                    login_verified = False
                    
                    # Method 1: Check for login button (should not be present if logged in)
                    try:
                        login_button = page.locator('a[href="/login"]')
                        if await login_button.count() > 0:
                            print("Not logged in (login button found). Please run login_manual.py again.")
                            await browser.close()
                            return result
                    except Exception as e:
                        print(f"Could not check for login button: {str(e)}")
                    
                    # Method 2: Check for sign up button (should not be present if logged in)
                    try:
                        signup_button = page.locator('a[href="/i/flow/signup"]')
                        if await signup_button.count() > 0:
                            print("Not logged in (signup button found). Please run login_manual.py again.")
                            await browser.close()
                            return result
                    except Exception as e:
                        print(f"Could not check for signup button: {str(e)}")
                    
                    # Method 3: Try to find home timeline
                    try:
                        timeline = page.locator('div[data-testid="primaryColumn"]')
                        if await timeline.count() > 0:
                            print("Login verified - timeline found")
                            login_verified = True
                    except Exception as e:
                        print(f"Could not check for timeline: {str(e)}")
                    
                    # Method 4: Check for profile link
                    if not login_verified:
                        try:
                            profile_link = page.locator('a[data-testid="AppTabBar_Profile_Link"]')
                            if await profile_link.count() > 0:
                                print("Login verified - profile link found")
                                login_verified = True
                        except Exception as e:
                            print(f"Could not check for profile link: {str(e)}")
                    
                    # Method 5: Check for any authenticated content
                    if not login_verified:
                        try:
                            # Look for any authenticated content
                            authenticated_selectors = [
                                'div[data-testid="SideNav_AccountSwitcher_Button"]',
                                'div[data-testid="AppTabBar_Home_Link"]',
                                'div[data-testid="AppTabBar_Explore_Link"]',
                                'div[data-testid="AppTabBar_Notifications_Link"]'
                            ]
                            
                            for selector in authenticated_selectors:
                                element = page.locator(selector)
                                if await element.count() > 0:
                                    print(f"Login verified - authenticated element found: {selector}")
                                    login_verified = True
                                    break
                        except Exception as e:
                            print(f"Could not check for authenticated elements: {str(e)}")
                    
                    # Method 6: Check page title or URL for authentication
                    if not login_verified:
                        try:
                            current_url = page.url
                            if "twitter.com/home" in current_url or "x.com/home" in current_url:
                                print("Login verified - on home page")
                                login_verified = True
                        except Exception as e:
                            print(f"Could not check URL: {str(e)}")
                    
                    if not login_verified:
                        print("Could not verify login status with any method.")
                        print("This might be due to Twitter's anti-bot measures or page loading issues.")
                        print("Attempting to continue anyway...")
                        # Don't return here, continue with scraping attempt
                    else:
                        print("Login verified successfully")

                except Exception as e:
                    print(f"Error during login verification: {str(e)}")
                    print("Attempting to continue anyway...")
                    # Don't return here, continue with scraping attempt

                # Navigate directly to user's profile with retry logic
                print(f"\nNavigating to profile @{username}...")
                navigation_success = False
                for attempt in range(3):  # Try 3 times
                    try:
                        await page.goto(f"https://twitter.com/{username}", wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(2)  # Give page time to load
                        navigation_success = True
                        break
                    except Exception as e:
                        print(f"Navigation attempt {attempt + 1}/3 failed: {str(e)}")
                        if attempt < 2:  # Not the last attempt
                            print("Retrying navigation...")
                            await asyncio.sleep(5)  # Wait before retrying
                        
                if not navigation_success:
                    print(f"Error navigating to profile after 3 attempts")
                    await browser.close()
                    return result
                
                # Verify profile exists and is accessible
                try:
                    # Check for error messages
                    error_selectors = [
                        'div[data-testid="error-detail"]',
                        'div[data-testid="empty-state"]',
                        'div[data-testid="404-error"]'
                    ]
                    
                    for selector in error_selectors:
                        error_element = page.locator(selector)
                        if await error_element.count() > 0:
                            error_text = await error_element.inner_text()
                            print(f"Profile error: {error_text}")
                            await browser.close()
                            return result
                            
                    # Verify profile content is visible with retry logic
                    profile_accessed = False
                    for attempt in range(3):  # Try 3 times
                        try:
                            # Try multiple selectors to detect profile
                            profile_selectors = [
                                'div[data-testid="UserName"]',
                                'h2[aria-level="2"]',
                                'div[data-testid="UserDescription"]',
                                'article[data-testid="tweet"]',
                                'div[data-testid="primaryColumn"]'
                            ]
                            
                            for selector in profile_selectors:
                                try:
                                    await page.wait_for_selector(selector, timeout=3000)
                                    profile_accessed = True
                                    print(f"✅ Profile @{username} accessed successfully (detected via {selector})")
                                    break
                                except:
                                    continue
                            
                            if profile_accessed:
                                break
                            else:
                                if attempt < 2:  # Not the last attempt
                                    print(f"Profile detection attempt {attempt + 1}/3 failed, retrying...")
                                    await asyncio.sleep(2)
                        except Exception as e:
                            if attempt < 2:
                                print(f"Profile verification attempt {attempt + 1}/3 failed: {e}, retrying...")
                                await asyncio.sleep(2)
                    
                    # if not profile_accessed:
                        # Don't return - continue with scraping as profile might still be accessible
                        
                except Exception as e:
                    print(f"Error verifying profile: {str(e)}")
                    await browser.close()
                    return result

                # Get profile info
                print(f"Fetching profile info for @{username}...")
                result["user_profile"] = await scrape_user_profile(page, username)
                print(f"Profile info fetched: {result['user_profile']}")
                
                if not result["user_profile"]["bio"] and not result["user_profile"]["username"]:
                    print(f"Could not fetch profile info for @{username}")
                    await browser.close()
                    return result
                
                # Get tweets and retweets
                # print(f"\nFetching tweets and retweets for @{username}...")
                # tweets, retweets = await scrape_tweets(page, username, max_tweets, max_retweets, stop_date)
                # if tweets:
                #     result["tweets"] = tweets
                #     print(f"Found {len(tweets)} tweets")
                # else:
                #     print("No tweets found or error occurred")
                    
                # if retweets:
                #     result["retweets"] = retweets
                #     print(f"Found {len(retweets)} retweets")
                # else:
                #     print("No retweets found or error occurred")
                
                # Only scrape social data if limits are greater than 0
                # if max_followers > 0 or max_following > 0:
                #     # Create a new page for social data (followers/following)
                #     social_page = await context.new_page()
                #     social_page.set_default_timeout(60000)  # Set to 60 seconds
                    
                #     # Get followers if limit > 0
                #     if max_followers > 0:
                #         print(f"\nFetching followers for @{username}...")
                #         # followers = await scrape_followers(social_page, username, max_followers)
                #         if followers:
                #             result["followers"] = followers
                #             print(f"Found {len(followers)} followers")
                #         else:
                #             print("No followers found or error occurred")
                #     else:
                #         print(f"\nSkipping followers (limit set to 0)")
                    
                #     # Small delay between operations
                #     await asyncio.sleep(1)
                    
                #     # Get following if limit > 0
                #     if max_following > 0:
                #         print(f"\nFetching following for @{username}...")
                #         following = await scrape_following(social_page, username, max_following)
                #         if following:
                #             result["following"] = following
                #             print(f"Found {len(following)} following")
                #         else:
                #             print("No following found or error occurred")
                #     else:
                #         print(f"\nSkipping following (limit set to 0)")
                    
                #     await social_page.close()
                # else:
                #     print(f"\nSkipping followers and following (both limits set to 0)")
                
                # Scraping completed
                print("\nScraping completed successfully!")
                
            except Exception as e:
                print(f"Error during scraping: {str(e)}")
            finally:
                print("\nClosing browser...")
                await safe_browser_close(browser)
                    
    except Exception as e:
        print(f"Critical error: {str(e)}")
    
    # --- Save result as JSON file in scraped_profiles directory ---
    try:
        scraped_profiles_dir = os.path.join(os.path.dirname(__file__), 'scraped_profiles')
        os.makedirs(scraped_profiles_dir, exist_ok=True)
        json_path = os.path.join(scraped_profiles_dir, f"{username}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Scraped profile saved to {json_path}")
    except Exception as e:
        print(f"Error saving scraped profile: {str(e)}")
    # --- End save ---
    
    return result

# Screenshot functions removed for better performance
# All screenshot functionality has been disabled


def extract_list(result, keys):
    for k in keys:
        if k in result and isinstance(result[k], (list, tuple)):
            return result[k]
    return None

async def fetch_user(username, max_tweets=20, max_followers=100, max_following=100, show=20, stop_date=None):
    try:
        print(f"\nStarting fetch for @{username}")
        print(f"Requesting tweets: {max_tweets}, followers: {max_followers}, following: {max_following}")

        result = await scrape_twitter(
            username=username,
            max_tweets=max_tweets,
            max_retweets=0,
            max_followers=max_followers,
            max_following=max_following,
            stop_date=stop_date
        )

        # Fetch profile image URL
        print(f"Fetching profile image for @{username} ...")
        image_url = await get_photo_image_url(username)
        result["profile_image_url"] = image_url

        # Extract lists
        followers = extract_list(result, ['followers', 'followers_list', 'followers_data', 'followers_users'])
        following = extract_list(result, ['following', 'following_list', 'friends', 'following_data', 'following_users'])
        tweets = extract_list(result, ['tweets', 'user_tweets', 'tweets_data'])

        # Profile summary
        profile = result.get('user_profile') or result.get('profile') or {}
        print("Profile summary:")
        if profile:
            print("  username:", profile.get('username') or profile.get('screen_name') or username)
            print("  name:", profile.get('name', ''))
            print("  bio:", profile.get('bio', ''))
            print("  followers_count:", profile.get('followers_count', 0))
            print("  following_count:", profile.get('following_count', 0))
        else:
            print("  (no profile data returned)")

        # Print utility
        def print_list(name, data):
            if not data:
                print(f"\n{name}: (not present in result)")
                return
            print(f"\n{name}: total returned = {len(data)}; showing up to {show}:")
            for i, item in enumerate(data[:show], start=1):
                if isinstance(item, dict):
                    u = item.get('username') or item.get('screen_name') or item.get('handle') or item.get('user') or item.get('name')
                    text = item.get('text') if 'text' in item else ''
                    extra = []
                    if 'id' in item: extra.append(f"id={item.get('id')}")
                    if 'followers_count' in item: extra.append(f"foll={item.get('followers_count')}")
                    tag = f" ({', '.join(extra)})" if extra else ""
                    print(f"  {i}. {u or text}{tag}")
                    if text: print(f"     -> {text}")
                else:
                    print(f"  {i}. {item}")
            if len(data) > show:
                print(f"  ... (+{len(data)-show} more)")

        print_list("Tweets", tweets)
        print_list("Followers", followers)
        print_list("Following", following)

        # Save JSON
        output_folder = os.path.join(os.path.dirname(__file__), "scraped_profiles")
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, f"{username}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nProfile saved to {filepath}")
        return True

    except Exception as e:
        print(f"Failed to fetch @{username}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def is_scraped(username, output_folder):
    path = os.path.join(output_folder, f"{username}.json")
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < 10:
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # consider as processed if there's useful content or a skipped marker
        if isinstance(data, dict) and (data.get("skipped") or data.get("profile_image_url") or data.get("user_profile") or data.get("tweets")):
            return True
        return True  # fallback: file exists and non-empty => treated as processed
    except Exception:
        return False
    
async def main():
    import json

    with open("github_twitter_usernames.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    usernames = [entry["Twitter Username"] for entry in data]
    for username in usernames:
        print(username)

    #usernames = usernames[:3]  
    stop_date = datetime(2025, 1, 1).date()  # optional: stop at this date

    output_folder = os.path.join(os.path.dirname(__file__), "scraped_profiles")
    os.makedirs(output_folder, exist_ok=True)
    for u in usernames:
        if is_scraped(u, output_folder):
            print(f"Already scraped or attempted {u}, skipping.")
            continue
        await fetch_user(u, max_tweets=10, max_followers=100, max_following=100, show=5, stop_date=stop_date)

if __name__ == "__main__":
    asyncio.run(main())
