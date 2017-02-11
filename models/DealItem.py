class DealItem:

    def __init__(self, row):
        self.id = row[0]
        self.deal_id = row[1]
        self.title_dealitem = row[2]
        self.coupon_text1 = row[3]
        self.coupon_text2 = row[4]
        self.begin_time = row[5]
        self.end_time = row[6]


#    def loadFromCsv(self, readerCSV):





