from scipy.spatial.distance import cosine


def corr(model, words):

    """
    :param model:
    :param words:
    :return:
    """

    words['cos_sim'] = words.apply(lambda row: 1 - cosine(model[row['w1']], model[row['w2']]), axis=1)
    return words['cos_sim'].corr(words['sim'], method='spearman')