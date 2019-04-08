# Recommender information retrieval

Information retrieval (VINF) - recommender school project.

## Technology stack

* Python 3.5
* Pandas, Numpy, Sklearn libraries
* Elastic Search

## About project

Recommend activit or product for users.

## Code example

```python
 df = pd.read_csv('train_activity.csv', sep=',', names=dheader)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print ('Number of users = ' + str(n_users) + ' | Number of items = ' + str(n_items))

    train_data, test_data = cv.train_test_split(df, test_size=0.25)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
```

## How to use it

* Install python on local machine
* Download data files (.csv)
* Run one of the following script without any parameters

```bash
python3 test.py
```

```bash
python3 main.py
```

## Sources

* Datasource is old dataset from slovak server [zlava dna](https://www.zlavadna.sk/)
