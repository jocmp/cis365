from constants import Constants
from html.parser import HTMLParser

"""
For initial code
http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
"""


class CsvCleaner(HTMLParser):
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
    def clean():
        cleaner = CsvCleaner()
        dirty_csv = open(Constants.UNCLEAN_DATA, 'r')
        clean_csv = open(Constants.PARSED_DATA, 'w')
        for line in dirty_csv:
            line = line.replace('</3', '')  # edge case for broken heart
            cleaner.feed(line)
            clean_csv.write(cleaner.data())
            cleaner = CsvCleaner()

        dirty_csv.close()
        clean_csv.close()


if __name__ == '__main__':
    CsvCleaner.clean()
