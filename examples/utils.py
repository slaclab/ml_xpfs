from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from scipy import special as sc
from scipy.stats import pearsonr
import numpy as np 

# custom keras callback using tqdm for epochs 
class TqdmCallbackFix(TqdmCallback):
    def _implements_train_batch_hooks(self): return True
    def _implements_test_batch_hooks(self): return True
    def _implements_predict_batch_hooks(self): return True

# Negative Log Likelihood for NB distribution 
def nll(beta, k, kbar):
    M = 1/(beta**2)
    return -(1/len(k))*np.nansum(sc.gammaln(M+k) - sc.gammaln(M) - k*np.log(kbar + M) + M*np.log((M)/(kbar + M)))
