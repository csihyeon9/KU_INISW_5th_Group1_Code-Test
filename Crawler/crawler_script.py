import os
import argparse
import datetime
import json
import xml.etree.ElementTree as ET
from dateutil.relativedelta import relativedelta
from urllib import parse
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import pandas as pd


def crawl_query_by_unit(query, save_dir, begin, end, mode, days=None):
    partial_end = initialize_partial_end(end, mode)
    
    while partial_end >= begin:
        partial_begin = update_partial_begin(partial_end, mode, days)
        partial_begin_str = partial_begin.strftime('%Y.%m.%d')
        partial_end_str = partial_end.strftime('%Y.%m.%d')
        
        print(f"Starting crawl for '{query}' from {partial_begin_str} to {partial_end_str}")
        
        save_as_json = os.path.join(save_dir, f"{query}_{partial_begin_str}-{partial_end_str}.json")
        save_as_xml = os.path.join(save_dir, f"{query}_{partial_begin_str}-{partial_end_str}.xml")
        
        if os.path.exists(save_as_json) and os.path.exists(save_as_xml):
            print(f"Data for {query} already exists for this date range. Skipping...")
        else:
            crawl(query=query, begin=partial_begin_str, end=partial_end_str,
                  save_as_json=save_as_json, save_as_xml=save_as_xml)
        
        partial_end = update_partial_end(partial_end, mode, days)


def initialize_partial_end(end, mode):
    if mode == 'weekly':
        return end - datetime.timedelta(days=end.weekday()) + datetime.timedelta(days=6)
    elif mode == 'monthly':
        return end.replace(day=1) + relativedelta(months=1) - datetime.timedelta(days=1)
    elif mode == 'interval':
        return end


def update_partial_begin(partial_end, mode, days=None):
    if mode == 'weekly':
        return partial_end - datetime.timedelta(days=6)
    elif mode == 'monthly':
        return partial_end.replace(day=1)
    elif mode == 'interval':
        return partial_end - datetime.timedelta(days=days-1)


def update_partial_end(partial_end, mode, days=None):
    if mode == 'weekly':
        return partial_end - datetime.timedelta(days=7)
    elif mode == 'monthly':
        return partial_end.replace(day=1) - datetime.timedelta(days=1)
    elif mode == 'interval':
        return partial_end - datetime.timedelta(days=days)


def crawl(query, begin, end, save_as_json, save_as_xml, sort=0, field=1, delay=0.5, timeout=30, page_limit=50):
    data = []
    current_index = 1
    ua = UserAgent()

    while current_index <= page_limit * 10:
        url = make_url(query, sort, field, begin, end, current_index)
        print(f"Crawling page {current_index // 10 + 1} for query '{query}'")
        
        try:
            html = urlopen(Request(url, headers={'User-Agent': ua.random}), timeout=timeout)
            bsobj = BeautifulSoup(html, 'html.parser')
        except Exception as e:
            print(f"Error fetching page: {e}")
            break

        news_links = [link['href'] for link in bsobj.find_all('a', href=True)
                      if 'https://news.naver.com/main/read' in link['href']]
        
        for link in news_links:
            try:
                news_data = fetch_news_data(link, ua, delay, timeout)
                if news_data:
                    data.append(news_data)
            except Exception as e:
                print(f"Error fetching news data: {e}")

        current_index += 10

    save_to_json(data, save_as_json)
    save_to_xml(data, save_as_xml)


def make_url(query, sort, field, begin, end, page):
    url = f"https://search.naver.com/search.naver?&where=news&query={parse.quote(query)}"
    url += f"&sort={sort}&field={field}&ds={begin}&de={end}"
    url += f"&nso=so:r,p:from{begin.replace('.', '')}to{end.replace('.', '')}"
    url += f"&start={page}"
    return url


def fetch_news_data(url, ua, delay, timeout):
    time.sleep(delay)
    try:
        html = urlopen(Request(url, headers={'User-Agent': ua.random}), timeout=timeout)
        bsobj = BeautifulSoup(html, 'html.parser')
        
        title = bsobj.select_one('h3#articleTitle').get_text(strip=True)
        article = bsobj.select_one('#articleBodyContents').get_text(strip=True)
        date = bsobj.select_one('.t11').get_text(strip=True)
        date = datetime.datetime.strptime(date, '%Y.%m.%d. %H:%M')

        return {"url": url, "title": title, "article": article, "date": date.isoformat()}
    except Exception as e:
        print(f"Error parsing news article at {url}: {e}")
        return None


def save_to_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Data saved to {save_path}")


def save_to_xml(data, save_path):
    root = ET.Element("NewsData")
    for entry in data:
        item = ET.SubElement(root, "NewsItem")
        for key, value in entry.items():
            ET.SubElement(item, key).text = str(value)
    
    tree = ET.ElementTree(root)
    tree.write(save_path, encoding="utf-8", xml_declaration=True)
    print(f"Data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True, help="Keyword to search")
    parser.add_argument('--begin', required=True, help="Start date in YYYY.MM.DD format")
    parser.add_argument('--end', required=True, help="End date in YYYY.MM.DD format")
    parser.add_argument('--save_dir', default="data", help="Directory to save results")
    parser.add_argument('--mode', required=True, choices=["weekly", "monthly", "interval"], help="Crawling mode")
    parser.add_argument('--days', type=int, default=7, help="Interval days if mode is 'interval'")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    begin_date = datetime.datetime.strptime(args.begin, "%Y.%m.%d")
    end_date = datetime.datetime.strptime(args.end, "%Y.%m.%d")

    crawl_query_by_unit(query=args.query, save_dir=args.save_dir, begin=begin_date, end=end_date, mode=args.mode, days=args.days)
