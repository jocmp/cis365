from html.parser import HTMLParser


class HtmlCleaner(HTMLParser):
    # Class to parse through text and remove HTML tags
    # `Initial sample code <http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python>`_

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def _data(self):
        return ' '.join(self.fed)

    @staticmethod
    def clean(text):
        ''' Returns copy of text stripped of HTML tags '''
        cleaner = HtmlCleaner()
        replaced_text = text.replace('</3', '')  # edge case for 'broken heart'
        cleaner.feed(replaced_text)
        return cleaner._data()
