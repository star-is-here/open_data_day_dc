#you can use this guide to scrape data from a webpage: http://docs.python-guide.org/en/latest/scenarios/scrape/
# scraping news headlines and descriptions from NIST's webpage

from lxml import html
import requests

page = requests.get('http://www.nist.gov/allnews.cfm?s=01-01-2014&e=12-31-2014')
tree = html.fromstring(page.content)

#list of news headlines and descriptions
headlines = tree.xpath('//div[@class="select_portal_module_wrapper"]/a/strong/text()')
descriptions = tree.xpath('//div[@class="select_portal_module_wrapper"]/p/text()')
news = zip(headlines,descriptions)

print news