# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os
from subprocess import call

DOWNLOAD_CMD = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'save.sh')

class SubtitleCrawlerPipeline(object):
    '''
    Download Subtitle with curl
    '''
    def process_item(self, item, spider):
        url = item['url']
        call([DOWNLOAD_CMD, url])
        return item
