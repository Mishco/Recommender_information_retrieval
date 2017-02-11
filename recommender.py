import numpy as np
import pandas as pd
import re
import re
import json
import csv

from datetime import datetime
from elasticsearch import Elasticsearch
import requests

es = Elasticsearch(['localhost:9200'])

def testElastic():
    doc = {
        'author': 'kimchy',
        'text': 'Elasticsearch: cool. bonsai cool.',
        'timestamp': datetime.now(),
    }
    res = es.index(index="test-index", doc_type='tweet', id=1, body=doc)
    print(res['created'])

    response = es.search(
        index='social-*',
        body={
            "query": {
                "match": {
                    "message": "myProduct"
                }
            },
            "aggs": {
                "top_10_states": {
                    "terms": {
                        "field": "state",
                        "size": 10
                    }
                }
            }
        }
    )
    print(response)

def main():

    # before start
    # testElastic()

    # Read csv into
    users_train = pd.read_csv('train_activity.csv')
    items_train = pd.read_csv('train_dealitems.csv', sep=',')
    test_deal   = pd.read_csv('test_deal_details.csv')
    drain_deal  = pd.read_csv('train_deal_details.csv')
    test_dealitem = pd.read_csv('test_dealitems.csv')
    drain_dealitem =  pd.read_csv('train_dealitems.csv')

    # ElasticWrapper
    # es

    # Insert into elastic
    # DealItem and activities

    res = es.index(index="deals", doc_type="deal", body=json.load(drain_deal))
    print(res['created'])




if __name__ == '__main__':
        main()