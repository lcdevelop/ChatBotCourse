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
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=900",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=901",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=902",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=903",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=904",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=905",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=906",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=907",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=908",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=909",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=910",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=911",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=912",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=913",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=914",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=915",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=916",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=917",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=918",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=919",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=920",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=921",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=922",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=923",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=924",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=925",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=926",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=927",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=928",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=929",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=930",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=931",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=932",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=933",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=934",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=935",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=936",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=937",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=938",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=939",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=940",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=941",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=942",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=943",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=944",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=945",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=946",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=947",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=948",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=949",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=950",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=951",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=952",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=953",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=954",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=955",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=956",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=957",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=958",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=959",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=960",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=961",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=962",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=963",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=964",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=965",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=966",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=967",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=968",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=969",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=970",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=971",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=972",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=973",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=974",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=975",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=976",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=977",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=978",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=979",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=980",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=981",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=982",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=983",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=984",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=985",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=986",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=987",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=988",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=989",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=990",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=991",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=992",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=993",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=994",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=995",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=996",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=997",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=998",
            "http://www.zimuku.net/search?q=&t=onlyst&ad=1&p=999",
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
