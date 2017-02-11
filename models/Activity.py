import copy

class Activity:

    def __init__(self, row):
        self.id = row[0]
        self.userId = row[1]
        self.dealitem_id = row[2]
        self.deal_id = row[3]
        self.quantity = row[4]
        self.market_price = row[5]
        self.team_price = row[6]
        self.create_time = row[7]


