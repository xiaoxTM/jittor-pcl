def subset(dataset,size):
    sub = [None] * size
    for idx, data in enumerate(dataset):
        if idx < size:
            sub[idx] = data
        else:
            break
    return sub
