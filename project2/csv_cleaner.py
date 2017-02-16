from html.parser import HTMLParser

"""
For initial code
http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
"""


class HTMLCleaner(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def data(self):
        return ' '.join(self.fed)

    @staticmethod
    def clean(text):
        cleaner = HTMLCleaner()
        cleaner_text = text.replace('</3', '')  # edge case for broken heart
        cleaner.feed(cleaner_text)
        return cleaner.data()
