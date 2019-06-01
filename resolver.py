import pandas as pd
from random import choice
from scipy import sparse
import numpy as np
from sklearn import linear_model
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import sys


#################################################
# ================ NAIVE SOLUTION ===============
#################################################

def naive_resolver(df):
    """
    :param df: The dataframe
    :return:
    """
    for ix in df[df.pos.str.contains("PRP")].pos.index:
        if pd.notnull(df["mention_ids"][ix]) and "PRP" in df["pos"][ix]:
            z = str(
                get_closest_prev_antecedent(df["mention_ids"], ix))
            df.at[ix, "mention_ids"] = str(
                get_closest_prev_antecedent(df["mention_ids"], ix))
    return df


def get_closest_prev_antecedent(d, ix, threshold=100):
    """
    :param d: dataframe for the data
    :param ix: current row index
    :param threshold: how far back should we check
    :return:
    """
    distance = 1

    while distance < threshold:
        if pd.isnull(d[ix - distance]):
            distance += 1
            continue
        else:
            # Splitting the columns with multiple labels
            # with a space.
            options = str(d[ix - distance]).split(" ")

            # Randomly choose between the possible options
            ch = choice(options)
            return str(ch)


def replace_indices(indices):
    return str(indices)[1:-1].replace(", ", " ")


###############################################
# ================ LOREG SOLUTION =============
###############################################

lmtzr = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
L2_STRENGTH = 1
MIN_FEATURE_COUNT = 0
feature_vocab = {}


def create_data(df):
    lst = []
    for doc in df['doc_id'].unique():
        doc_df = df.loc[df['doc_id'] == doc]
        prp_rows = doc_df.loc[(doc_df['pos'] == 'PRP') | (doc_df['pos'] == 'PRP$')]
        np_rows = doc_df.loc[doc_df['mention_ids'].notnull() & (doc_df['pos'] != 'PRP') & (doc_df['pos'] != 'PRP$')]
        for pid in prp_rows.index:
            for nid in np_rows.index:
                if nid > pid:
                    break
                np_wrd = df.at[nid, 'word']
                if any(char.isdigit() for char in np_wrd) \
                    or lmtzr.lemmatize(np_wrd) == 'it' \
                    or np_wrd in STOPWORDS:
                    continue    # Some filtering for Mention Detection
                match = 0
                if "entity_ids" in doc_df.columns:
                    same_mention = set(str(doc_df["entity_ids"][pid]).split(" ")) \
                                   & set(str(doc_df["entity_ids"][nid]).split(" "))
                    if same_mention:
                        match = 1
                lst.append(((pid, nid), match))
    return lst


def is_plural(word):
    lemma = lmtzr.lemmatize(word, 'n')
    return True if word is not lemma else False


def create_features(tup, df):
    feats = {}
    prp_wrd = df.at[tup[0], "word"]
    np_wrd = df.at[tup[1], "word"]

    # Tokens and lemmatized tokens unary feature.
    feats["PRP_WRD_LEMMA_{}".format(lmtzr.lemmatize(prp_wrd))] = 1
    feats["NP_WRD_LEMMA_{}".format(lmtzr.lemmatize(np_wrd))] = 1
    feats["PRP_WRD_{}".format(prp_wrd)] = 1
    feats["NP_WRD_{}".format(np_wrd)] = 1

    # Part of speech unary features.
    feats["PRP_POS_{}".format(df.at[tup[0], "pos"])] = 1
    feats["NP_POS_{}".format(df.at[tup[1], "pos"])] = 1

    # Binned Token Distances:
    token_dist = min((tup[0] - tup[1])//4, 100)
    feats["DISTANCE_{}".format(token_dist)] = 1

    # Number Agreement
    if is_plural(prp_wrd) and is_plural(np_wrd):
        feats["BOTH_PLURAL"] = 1

    if not is_plural(prp_wrd) and not is_plural(np_wrd):
        feats["BOTH_SINGULAR"] = 1

    return feats


def featurize(data, df):
    featured_data = []
    i = 0
    for tup, match in data:
        feats = create_features(tup, df)
        featured_data.append((tup, match, feats))
        i += 1
        sys.stdout.write("\rFeaturing: {}/{}".format(i, len(data)))
        sys.stdout.flush()
    print(" ")
    return featured_data


def process(data, df, training=False):
    global feature_vocab

    data = featurize(data, df)

    if training:
        fid = 0
        feature_doc_count = Counter()
        for label, match, feats in data:
            for feat in feats:
                feature_doc_count[feat] += 1

        for feat in feature_doc_count:
            if feature_doc_count[feat] >= MIN_FEATURE_COUNT:
                feature_vocab[feat] = fid
                fid += 1

    F = len(feature_vocab)
    D = len(data)
    X = sparse.dok_matrix((D, F))
    Y = np.zeros(D)

    i = 0
    for idx, (label, match, feats) in enumerate(data):
        for feat in feats:
            if feat in feature_vocab:
                X[idx, feature_vocab[feat]] = feats[feat]
        i += 1
        sys.stdout.write("\rAdding Feature: {}/{}".format(i, len(data)))
        sys.stdout.flush()
        Y[idx] = match
    print(" ")
    return X, Y


def evaluate(log_reg, trainX, trainY, devX, devY):
    print("\nTraining LoReg, Feature Count: {}".format(len(feature_vocab.keys())))
    log_reg.fit(trainX, trainY)
    training_accuracy = log_reg.score(trainX, trainY)
    development_accuracy = log_reg.score(devX, devY)
    print("LoReg Results:\n\tTrain acc: {}, dev acc: {}".format(training_accuracy, development_accuracy))


def modify_df_with_preds(df, dat, predY):
    i = 0
    for idx in range(len(predY)):
        is_matched = predY[idx]
        if is_matched:
            matched_data = dat[idx]
            prp_index = matched_data[0][0]
            np_index = matched_data[0][1]
            np_mention_id = df.at[np_index, "mention_ids"]
            df.at[prp_index, "mention_ids"] = np_mention_id
        i += 1
        sys.stdout.write("\rModding DF: {}/{}".format(i, len(predY)))
        sys.stdout.flush()
    print(" ")
    return df


##############################################
# ================ Parsing Tools =============
##############################################

def get_index(entity_id, ix_to_entity_id, entity_id_to_ix, sequence_flag=False):
    """
    This function generates the mapping between entity and test ids

    :param entity_id: Current entity id
    :param ix_to_entity_id: Current mapping from test to entity
    :param entity_id_to_ix: Current mappings from entity to test ids
    :param sequence_flag: Whether the previous word was the same entity
    :return: The test_id for the word
    """
    # If sequence, don't generate new id
    if sequence_flag:
        cur_ix = entity_id_to_ix[entity_id]
        ix = max(cur_ix)
        return ix, ix_to_entity_id, entity_id_to_ix
    # If existing entity, add to test id list for entity
    if entity_id in entity_id_to_ix.keys():
        cur_ix = entity_id_to_ix[entity_id]
        ix = max(list(ix_to_entity_id.keys())) + 1
        cur_ix.append(ix)
        entity_id_to_ix[entity_id] = cur_ix
        ix_to_entity_id[ix] = entity_id
        return ix, ix_to_entity_id, entity_id_to_ix
    # Else, create new entry for entity
    else:
        # If no entry has been created
        if len(ix_to_entity_id.keys()) == 0:
            ix = -1
        else:
            ix = max(list(ix_to_entity_id.keys()))
        ix_to_entity_id[ix + 1] = entity_id
        entity_id_to_ix[entity_id] = [ix + 1]
        return ix + 1, ix_to_entity_id, entity_id_to_ix


def get_mention_ids(df, return_ds=False):
    df.reset_index(inplace=True, drop="Index")

    ix_to_entity_id = {}
    entity_id_to_ix = {}

    prev_entities = []
    mention_ids = []

    for row_ix in df.index:
        label = df["entity_ids"][row_ix]
        if pd.isnull(label):
            prev_entities = []
            mention_ids.append("")
            continue
        else:
            entities = label.split(" ")
            indices = []
            for entity in entities:
                if entity in prev_entities:
                    ix, ix_to_entity_id, entity_id_to_ix = get_index(entity,
                                                             ix_to_entity_id,
                                                             entity_id_to_ix,
                                                             True)
                    indices.append(ix)
                else:
                    ix, ix_to_entity_id, entity_id_to_ix = get_index(entity,
                                                             ix_to_entity_id,
                                                             entity_id_to_ix)
                    indices.append(ix)
            mention_ids.append(replace_indices(indices))
            prev_entities = entities

    if return_ds:
        return ix_to_entity_id, entity_id_to_ix
    else:
        df["mention_ids"] = pd.Series(mention_ids)
        return df


#############################################################
# ================ ACCURACY CHECKING FUNCTIONS ==============
#############################################################

def check_common_preds(y_pred, y, ix_to_label):
    y_pred = set([ix_to_label[int(pred)] for pred in y_pred.split(" ")])
    y = set(y.split(" "))

    return len(y.intersection(y_pred)) > 0


def check_valid_record(df, y_pred, ix):
    r1 = df["entity_ids"][ix]
    r2 = y_pred[ix]
    pos = df["pos"][ix]

    prediction_limit = 3

    # Check conditions for valid prediction record
    if pd.isnull(r1) or pd.isnull(r2):
        return False
    if r1 == "" or r2 == "":
        return False
    if r1 is None or r2 is None:
        return False
    if r1 == "None" or r2 == "None":
        return False
    if "PRP" not in pos:
        return False
    if len(r2.split(" ")) > prediction_limit:
        return False
    else:
        return True


def check_accuracy(y_pred, df):
    """
    :param y_pred: pandas series for predicted mention_ids
    :param df: The data frame with original entity and mention ids
    :return: float: mean accuracy of the predictions
    """
    y = df["entity_ids"].copy()

    # build test_id to entity dictionaries and vice versa
    ix_to_label, label_to_ix = get_mention_ids(df, True)
    scores = []

    for ix in y_pred.index:
        if check_valid_record(df, y_pred, ix):
            scores.append(1.0 if check_common_preds(y_pred[ix], y[ix], ix_to_label)
                          else 0.0)
    return np.mean(scores)


def main():
    train_df = pd.read_csv("train.coref.data.txt", sep="\t")
    dev_df = pd.read_csv("dev.coref.data.txt", sep="\t")
    log_reg = linear_model.LogisticRegression(C=L2_STRENGTH)

    print("\nCreating LoReg Training Data...")
    train_dat = create_data(train_df)
    print("\tLength of data: {}".format(len(train_dat)))
    print("Creating LoReg Dev Data...")
    dev_dat = create_data(dev_df)
    print("\tLength of data: {}".format(len(dev_dat)))
    print("Featurizing Dev and Training data...")
    trainX, trainY = process(train_dat, train_df, True)
    devX, devY = process(dev_dat, dev_df)

    evaluate(log_reg, trainX, trainY, devX, devY)

    # TEST DATA PREDATION / WRITING
    print("\nEvaluating Test File...")
    test_df = pd.read_csv("test.coref.data.txt", sep="\t")
    print("Creating LoReg Test Data...")
    test_dat = create_data(test_df)
    print("\tLength of data: {}".format(len(test_dat)))
    print("Featurizing Test data...")
    testX, testY = process(test_dat, test_df)
    new_test_df = modify_df_with_preds(test_df, test_dat, log_reg.predict(testX))
    # Writing test CSV from DF
    new_test_df.to_csv("test.coref.data.txt", sep='\t', encoding='utf-8', index=False)
    print("Written as: test.coref.data.txt\n")

    result_df2 = modify_df_with_preds(dev_df, dev_dat, log_reg.predict(devX))
    print("Dev Accuracy (DataFrame Replaced): {:.4f}".format(check_accuracy(dev_df["mention_ids"], result_df2)))

if __name__ == "__main__":
    main()


