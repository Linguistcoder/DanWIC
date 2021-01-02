import pandas as pd
import numpy as np
import random

data = pd.read_csv('data_all_senses.csv',
                   sep='@',
                   encoding='windows-1252',
                   index_col=False,
                   )


def create_testset(data):
    """Split all the data into test, development and training sets"""
    grouped = data.groupby('form')
    unique_forms = sorted(list(set(data['form'])))

    test = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])
    devel = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])
    train = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])
    index_val = []

    for index,form in enumerate(unique_forms):
        group = grouped.get_group(form)
        group = group.drop_duplicates(subset=['sentence_1'])
        group = group.drop_duplicates(subset=['sentence_2'])

        for index, sentence in group['sentence_2'].iteritems():
            if sentence in list(group['sentence_1']) and sentence in list(group['sentence_2']):
                sidx = np.where(group['sentence_1'] == sentence)[0]
                idx = group.index.values[sidx]
                group = group.drop(idx)
                index_val.append(idx)


        if index%2==0:
            test = test.append(group.iloc[:6:2])
            devel = devel.append(group.iloc[1:6:2])

        else:
            test = test.append(group.iloc[1:6:2])
            devel = devel.append(group.iloc[:6:2])
        train = train.append(group.iloc[6:])

    return test, devel, train

test, devel, train = create_testset(data)

test.to_csv('test_all.csv',sep='@',encoding='windows-1252')
devel.to_csv('devel_all.csv',sep='@',encoding='windows-1252')


test_unique_counts = test['label'].value_counts()
devel_unique_counts = devel['label'].value_counts()

# ensures balanced number of labels
test_count = np.mean(test['label'])
test1 = test.groupby('label').sample(n=test_unique_counts.loc[1], random_state=1)

# ensures balanced number of labels
devel_count = np.mean(devel['label'])
devel1 = devel.groupby('label').sample(n=devel_unique_counts.loc[1], random_state=1)

test_final = test1.sample(n=1600, random_state=1).sort_values(by=['form'])
devel_final = devel1.sample(n=800, random_state=1).sort_values(by=['form'])

# add all instances to training
training = pd.concat([test, devel, train])
# remove those that are in test or development set
training = training[~training.isin(test_final)].dropna()
training = training[~training.isin(devel_final)].dropna().sort_values(by=['form'])

test_final.to_csv('test_final_all.csv',sep='\t',encoding='windows-1252',index=False)
devel_final.to_csv('devel_final_all.csv',sep='\t',encoding='windows-1252',index=False)
training.to_csv('training_final_all.csv',sep='\t',encoding='windows-1252',index=False)


