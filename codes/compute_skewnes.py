def compute_skewness(epochs):
    
    from scipy.stats import skew
    keep_skewness=[]
    for i, epoch in enumerate(epochs):
        deriv=np.diff(epoch, axis=1)
        epoch_skewness=skew(np.abs(deriv),axis=1)
        keep_skewness.append(epoch_skewness)
    epochs.metadata['skewness'] = np.mean(keep_skewness,axis=1)
    return epochs
