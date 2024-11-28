import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, bootstrap
import seaborn as sns
from sklearn.metrics import roc_auc_score

def get_avg(img):
    """
    Calculate the mean of values in an image that are above a threshold.

    Args:
        img (numpy.ndarray): Input image array.

    Returns:
        float: Mean value of the pixels above the threshold.
    """
    
    list = img[img > 0.01]
    return(np.mean(list))


def get_10perc(img, reverse = False):
    """
    Calculate the average of the top or bottom 10% of pixel values in an image.

    Args:
        img (numpy.ndarray): Input image array.
        reverse (bool): Whether to calculate for the top 10% (default False).

    Returns:
        float: Mean of the top or bottom 10% values.
    """
    
    list = img[img > 0.01]
    list = sorted(list)
    if reverse:
        list = list[::-1]
    idx  = np.ceil(0.1*len(list))
    list = list[:int(idx)]
    return(np.mean(list))

    
def get_correlation(a, b, verbose = True):
    """
    Compute Pearson and Spearman correlation coefficients between two arrays.

    Args:
        a (numpy.ndarray): First input array.
        b (numpy.ndarray): Second input array.
        verbose (bool): Whether to print the results (default True).

    Returns:
        tuple: Pearson correlation, Pearson p-value, Spearman correlation, Spearman p-value.
    """
    
    pearson_cor, p_val1 = pearsonr(a, b)
    spear_cor,   p_val2 = spearmanr(a, b)

    if verbose:
        print(f'Pearson:  {pearson_cor:.4f}    p-val: {p_val1:.1e}')
        print(f'Spearman: {spear_cor:.4f}    p-val: {p_val2:.1e}')
                
    return pearson_cor, p_val1, spear_cor, p_val2


def bootstrap_correlation(a,b):
    """
    Perform bootstrapping to compute the confidence interval for Pearson correlation.

    Args:
        a (numpy.ndarray): First input array.
        b (numpy.ndarray): Second input array.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    
    data = (a,b)
    
    def get_pearson(a, b):
        return pearsonr(a, b)[0]
    
    res = bootstrap(data, get_pearson, paired=True, n_resamples=10000)
    
    if res.confidence_interval[0] < 0:
        CI_min = res.confidence_interval[1]
        CI_max = res.confidence_interval[0]
    else:
        CI_min = res.confidence_interval[0]
        CI_max = res.confidence_interval[1]
    
    return (CI_min, CI_max)


def print_correlation(x_vals, y_vals, parameter_name):
    """
    Print correlation coefficients and their confidence intervals for given data.

    Args:
        x_vals (numpy.ndarray): X-axis values.
        y_vals (numpy.ndarray): Y-axis values.
        parameter_name (str): Name of the parameter being analyzed.
    """
    
    all_x = x_vals.flatten()
    all_y = y_vals.flatten()
    
    print('---', parameter_name, '---')
    pearson, _, _, _ = get_correlation(all_x, all_y, verbose=False)
    pearson_CI       = bootstrap_correlation(all_x, all_y)
    print(f'r = {pearson:.3f} [{pearson_CI[0]:.3f},{pearson_CI[1]:.3f}]')
    
    
def plot_correlation(x_vals, y_vals, xlabel, ylabel, nums=[], delete=[]):
    """
    Plot correlation between two datasets, with optional highlighted points.

    Args:
        x_vals (numpy.ndarray): X-axis values.
        y_vals (numpy.ndarray): Y-axis values.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        nums (list): Indices of highlighted points.
        delete (list): Indices of points to exclude.
    """
    
    # Concatenate all values for final output
    all_x = x_vals.flatten()
    all_y = y_vals.flatten()
    
    # Remove selected indices
    if len(delete)>0:
        NUM = x_vals.shape[1]
        new_x, new_y = np.zeros((2,NUM-len(delete))), np.zeros((2,NUM-len(delete)))
        new_x[0,:] = np.delete(x_vals[0,:], delete)
        new_x[1,:] = np.delete(x_vals[1,:], delete)
        new_y[0,:] = np.delete(y_vals[0,:], delete)
        new_y[1,:] = np.delete(y_vals[1,:], delete)
        all_x = np.delete(all_x, np.append(delete,[i+NUM for i in delete]))
        all_y = np.delete(all_y, np.append(delete,[i+NUM for i in delete]))
        x_vals, y_vals = new_x, new_y
        
    pearson, _, _, _ = get_correlation(all_x, all_y, verbose=False)
    pearson_CI       = bootstrap_correlation(all_x, all_y)
    
    plt.figure()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    _ = sns.regplot(x=all_x, y=all_y, ci=None, color='black',line_kws=dict(linewidth=0.8))
    plt.scatter(x_vals[0,:],y_vals[0,:],c='red',label='PCa')
    plt.scatter(x_vals[1,:],y_vals[1,:],c='dodgerblue',label='Benign')
    
    for i in range(len(nums)):
        plt.scatter(x_vals[0,nums[i]],y_vals[0,nums[i]],c='black', alpha=1-i*0.15)
        plt.scatter(x_vals[1,nums[i]],y_vals[1,nums[i]],c='black', alpha=1-i*0.15)
    
    plt.legend()
    
    # Define text coordinates
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    text_y = y_min + 0.05 * (y_max - y_min) 
    text_x = x_min + 0.28 * (x_max - x_min)  

    t = plt.text(text_x, text_y, f'r = {pearson:.3f} [{pearson_CI[0]:.2f},{pearson_CI[1]:.2f}]', fontsize=12, color='black')
    t.set_bbox(dict(facecolor='whitesmoke', alpha=0.5, edgecolor='whitesmoke'))
    plt.show()


def bootstrap_auc(labels, scores, n_bootstraps=1000, rdn_state=42):
    """
    Perform bootstrapping to compute confidence intervals for the AUC.

    Args:
        labels (numpy.ndarray): Ground truth labels (binary classification).
        scores (numpy.ndarray): Predicted scores for each sample.
        n_bootstraps (int): Number of bootstrap samples (default 1000).
        rdn_state (int): Random seed for reproducibility (default 42).

    Returns:
        tuple: Lower and upper bounds of the 95% confidence interval for AUC.
    """
    
    rng = np.random.RandomState(rdn_state)  # For reproducibility
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, len(labels), len(labels))
        score_sample = scores[indices]
        label_sample = labels[indices]
        # Calculate AUC only if both classes are present in the sample
        if len(np.unique(label_sample)) == 2:
            bootstrapped_aucs.append(roc_auc_score(label_sample, score_sample))

    # Calculate 95% CI
    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    return ci_lower, ci_upper


def auc_p_val(labels, scores, auc_value, n_permutations=10000, rdn_state=42):
    """
    Perform a permutation test to compute the p-value for the AUC.

    Args:
        labels (numpy.ndarray): Ground truth labels (binary classification).
        scores (numpy.ndarray): Predicted scores for each sample.
        auc_value (float): Observed AUC value.
        n_permutations (int): Number of permutations for the test (default 10000).
        rdn_state (int): Random seed for reproducibility (default 42).

    Returns:
        float: P-value indicating the significance of the observed AUC.
    """
    
    rng = np.random.RandomState(rdn_state)  # For reproducibility
    permuted_aucs = []

    for _ in range(n_permutations):
        # Shuffle labels to break any relationship between scores and labels
        shuffled_labels = rng.permutation(labels)
        permuted_aucs.append(roc_auc_score(shuffled_labels, scores))

    # Calculate p-value
    # Count how many permuted AUCs are at least as extreme as the observed AUC
    p_value = (np.sum(np.array(permuted_aucs) >= auc_value) + 1) / (n_permutations + 1)
    return p_value


def get_auc(data_1, data_2, name, label_1='csPCa', label_2='Benign'):
    """
    Calculate and display AUC with confidence intervals for given data.

    Args:
        data_1 (numpy.ndarray): Input data of class 1.
        data_2 (numpy.ndarray): Input data of class 2.
        name (str): Name of the parameter being analyzed.
        label_1 (str): Name of data class 1.
        label_2 (str): Name of data class 2.
    """
    
    print(f"{name}")
    
    mean_1, mean_2, std_1, std_2 = np.mean(data_1), np.mean(data_2), np.std(data_1), np.std(data_2)
    print(f"{label_1}: {mean_1:.2f} +/- {std_1:.2f} ")
    print(f"{label_2}: {mean_2:.2f} +/- {std_2:.2f} ")
    
    # Flatten data and create labels
    scores = np.concatenate([data_1, data_2])
    if mean_2>mean_1: 
        labels = np.concatenate([np.zeros(len(data_1)), np.ones(len(data_2))])
    else:
        labels = np.concatenate([np.ones(len(data_1)), np.zeros(len(data_2))])

    # Calculate AUC
    auc_value           = roc_auc_score(labels, scores)
    ci_lower, ci_upper  = bootstrap_auc(labels, scores)
    
    print(f"AUC: {auc_value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}] ")
    
    # Permutation test for p-value
    p_value = auc_p_val(labels, scores, auc_value)
    print(f"P-value: {p_value:.5f}\n")