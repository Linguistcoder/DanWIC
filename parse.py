import spacy
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import re

nlp = spacy.load("da_core_news_sm")

def clean(string):
    # clean a string
    return re.sub('["\'\[\]]+', '', string)


def create_dataframe():
    """Return DanNet as pd.DataFrame. The tables have been outer joined."""
    # Load data from DanNet
    file_words = 'DanNet-2.2_csv/words.csv'
    file_wordsense = 'DanNet-2.2_csv/wordsenses.csv'
    file_synsets = 'DanNet-2.2_csv/synsets.csv'
    file_relations = 'DanNet-2.2_csv/relations.csv'

    words = pd.read_csv(file_words,
                        sep='@',
                        encoding='windows-1252',
                        names=['word_id', 'form', 'PoS'],
                        index_col=False,
                        )

    wordsenses = pd.read_csv(file_wordsense,
                             sep='@',
                             encoding='windows-1252',
                             names=['wordsense_id', 'word_id', 'synset_id', 'register'],
                             index_col=False
                             )

    synsets = pd.read_csv(file_synsets,
                          sep='@',
                          encoding='utf8',
                          names=['synset_id', 'label', 'gloss', 'ont'],
                          index_col=False,
                          skiprows=0
                          )

    relations = pd.read_csv(file_relations,
                            sep='@',
                            encoding='windows-1252',
                            names=['synset_id', 'name', 'name2', 'value', 'taxonomic', 'in_com'],
                            index_col=False
                            )
    dataset = words.merge(wordsenses.loc[:, ['wordsense_id', 'word_id', 'synset_id']], how='outer')
    synsets['synset_id'] = synsets['synset_id'].astype(str)
    dataset = dataset.merge(synsets.loc[:, ['synset_id', 'gloss', 'ont']], how='outer')

    # dataset = dataset[dataset.gloss != '(ingen definition)']

    return dataset


def unique_cols(df):
    # checks whether the colomns in a dataframe are unique
    return df.nunique() == len(df)


def form_in_sentence(sentence, form):
    """Checks that a word form is in a sentence"""
    sentence = [token.lemma_ for token in nlp(sentence)]
    form = nlp(form)[0].lemma_
    form_in_sent = form in sentence
    return form_in_sent

def sentences_in_data(dataframe):
    """Collects all the sentences for every row in DanNet table"""
    all_sentences = []
    _index = []
    for row in dataframe.itertuples():

        gloss = row.gloss
        sentences = ''
        if 'Brug:' in str(gloss):
            # get the example sentences by splitting after "Brug:"
            sent = gloss.split('Brug:')

            # more than one sentence
            if '||' in sent[1] or ';' in sent[1]:
                # split example usage into sentences
                sentences = sent[1].split('||')
                sentences = ';'.join(sentences).split(';')
                # add the sentence if the word form appears in the sentence
                sentences = [s.strip('") \'') for s in sentences if form_in_sentence(s,row.form)]
            else:
                # Only one sentence
                sentences = [sent[1].strip('") \'')] if form_in_sentence(sent[1],row.form) else ''
        n_sen = len(sentences)

        # if there are sentences (empty example usages are removed)
        if sentences:
            all_sentences.append([row.word_id,
                                  row.form,
                                  row.wordsense_id,
                                  row.synset_id,
                                  row.ont,
                                  '||'.join(sentences),
                                  n_sen,
                                  row.gloss])

    # Store sentences in dataframe.
    all_sentences = pd.DataFrame(all_sentences, columns=['word_id', 'form', 'wordsense_id',
                                                         'synset_id', 'ont', 'sentences', 'length',
                                                         'gloss'])

    print('Collected all sentences')
    return all_sentences


def group_data(data, group, forms, cluster=True):
    """Groups the data into word forms and collect all possible instances for that word form"""
    grouped = data.groupby(group)
    dataset = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])

    for form in list(forms)[1:]:
        group = grouped.get_group(form)

        # check no duplicates of synsets
        if unique_cols(group['gloss']) is False:
            group = group.drop_duplicates(subset=['gloss'])

        # check that there is at least two sentences in group
        if np.sum(group['length']) > 2:

            if cluster is True:
                group = cluster_senses(group)
            # sentences from same synset
            for n_row in range(group.shape[0]):
                if group['length'].iloc[n_row] > 1:
                    new_df = sentences_to_dataset(group.iloc[n_row])
                    dataset = pd.concat([dataset, new_df])
        # check that there are more than 1 synset for the word form group
        if group.shape[0] > 1:
            # sentences from different synsets
            new_new_df = dif_mean_to_data(group)
            dataset = pd.concat([dataset, new_new_df])
    print('Done')
    return dataset

def cluster_senses(df):
    """Naive clustering. """
    dub = df.groupby('ont')
    if dub.ngroups > 1:
        dfs = []
        for ont in dub.groups:
            group = dub.get_group(ont)
            if group.shape[0] > 1:
                sentences = ''
                wordsense_id = ''
                synset_id = ''
                for row in group.itertuples():
                    sentences += row.sentences+'||'
                    wordsense_id += str(row.wordsense_id) + '+'
                    synset_id += str(row.synset_id) + '+'
                row = dict({'word_id': [group['word_id'].iloc[0]],
                            'form': [group.iloc[0]['form']],
                            'wordsense_id': [wordsense_id.rstrip('+')],
                            'synset_id': [synset_id.rstrip('+')],
                            'ont': [ont],
                            'sentences': [sentences.rstrip('||')],
                            'length': [len(sentences.rstrip('||').split('||'))],
                            'gloss': [group['gloss'].iloc[0]]})
                row = pd.DataFrame(row)
            else:
                row = group.copy()
            dfs.append(row)
        new_df = pd.concat(dfs, axis=0)
        new_df = new_df.drop_duplicates(subset=['ont'], keep='first')
        return new_df
    else:
        return df

def sentences_to_dataset(group):
    """Return pandas dataframe with sentences from the same synset in group"""
    df = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])
    sentences = group['sentences'].split('||')
    count = 0
    for i in range(len(sentences)-1):
        df.loc[count] = [group['form'],
                         clean(sentences[i]),
                         clean(sentences[i + 1]),
                         1]

        count += 1
    return df


def dif_mean_to_data(group):
    """Return pandas dataframe with sentences from different synsets in group"""
    df = pd.DataFrame(columns=['form', 'sentence_1', 'sentence_2', 'label'])
    count = 0
    shape = group.shape[0]
    for i in range(shape - 1):
        for n in range(shape - i):
            if n == 0:
                continue
            if i + n > shape:
                break
            row_1 = group.iloc[i]
            row_2 = group.iloc[i + n]
            sentences_1 = row_1['sentences'].split('||')
            sentences_2 = row_2['sentences'].split('||')

            for sent1 in sentences_1:
                for sent2 in sentences_2:
                    df.loc[count] = [row_1['form'], clean(sent1), clean(sent2), 0]
                    count += 1

    return df


cleaned = create_dataframe()
data = sentences_in_data(cleaned)
forms = set(data['form'])

groups = group_data(data, 'form', forms, cluster=True)

groups.to_csv('data_clus_senses.csv', sep='@', encoding='windows-1252', index=False)


