# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import scrapy
from w3lib.html import remove_tags
from subtitle_crawler.items import SubtitleCrawlerItem

class SubTitleSpider(scrapy.Spider):
    name = "subtitle"
    allowed_domains = ["zimuku.net"]
    start_urls = [
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=20",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=21",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=22",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=23",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=24",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=25",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=26",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=27",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=28",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=29",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=30",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=31",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=32",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=33",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=34",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=35",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=36",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=37",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=38",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=39",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=40",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=41",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=42",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=43",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=44",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=45",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=46",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=47",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=48",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=49",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=50",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=51",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=52",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=53",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=54",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=55",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=56",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=57",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=58",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=59",
    ]

    def parse(self, response):
        hrefs = response.selector.xpath('//div[contains(@class, "persub")]/h1/a/@href').extract()
        for href in hrefs:
            url = response.urljoin(href)
            request = scrapy.Request(url, callback=self.parse_detail)
            yield request

    def parse_detail(self, response):
        url = response.selector.xpath('//li[contains(@class, "dlsub")]/div/a/@href').extract()[0]
        print "processing: ", url
        request = scrapy.Request(url, callback=self.parse_file)
        yield request

    def parse_file(self, response):
        body = response.body
        item = SubtitleCrawlerItem()
        item['url'] = response.url
        item['body'] = body
        return item
