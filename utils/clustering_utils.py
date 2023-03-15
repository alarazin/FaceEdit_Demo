import sys
sys.path.append("../GANLocalEditing")
from spherical_kmeans import MiniBatchSphericalKMeans
import numpy as np 
import cv2

def one_hot(a,n):
  b=np.zeros((a.size,n))
  b[np.arange(a.size), a]=1
  return b

def partial_unflat(x, orig_shape, N=None, H=None, W=None):
  assert len(x.shape)==2
  C=x.shape[1]
  if N is None:
    N,C,H,W=orig_shape
  if W is None:
    W=H
  assert N is not None and H is not None and W is not None
  return x.reshape((N,H,W,C)).transpose((0,3,1,2))


class MultiResolutionStore:
  def __init__(self, item=None ): #interpolation_mode='bilinear'
    self._data={}
    self._res=None
    if item is not None:
      self._res=item.shape[-1]
      self._data[self._res]=item
    #self.interpolation_mode=interpolation_mode

  def get(self, res=None, make=True):
    if res==None:
      res=self._res
    if res not in self and make:
      self.make(res)
    
    ret=self._data[res]
    return ret
  
  def __getitem__(self,res):
    return self.get(res,make=False)
  
  def resolutions(self):
    return (res for res in self._data.keys())
  
  def __repr__(self):
    return 'MultiResolutionStore {}: {}'.format(self._data[self._res].shape, list(self.resolutions()))

  def make(self, res):
    self._data[res]=self._resize(res)
  
  def _resize(self, res):
    assert type(res) is int
    return cv2.resize(self._data[self._res][0].transpose((1,2,0)), (res, res), interpolation=cv2.INTER_NEAREST)

#h_1024=cv2.resize(heatmaps._data[heatmaps._res][0].transpose((1,2,0)), (1024, 1024), interpolation=cv2.INTER_NEAREST)


class FactorCatalog:
  def __init__(self, k, random_state=0, factorization=None, **kwargs):
    if factorization is None:
      factorization=MiniBatchSphericalKMeans
    self._factorization=factorization(n_clusters=k, random_state=random_state, **kwargs)
    self.annotations={}
  
  def _preprocess(self, X):
    X_flat=X.copy()
    X_flat=X_flat.transpose((0,2,3,1)).reshape((-1,X.shape[1]))
    orig_shape=X.shape
    return X_flat, orig_shape
  
  def _postprocess(self, labels, X, raw, orig_shape):
    heatmaps=one_hot(labels, self._factorization.cluster_centers_.shape[0])
    heatmaps=partial_unflat(heatmaps, orig_shape, N=X.shape[0], H=X.shape[-1])
    #print(heatmaps.shape)
    #print(self.annotations.values())
    if raw:
      heatmaps=MultiResolutionStore(heatmaps)
      return heatmaps
    else:
      heatmaps=MultiResolutionStore(np.concatenate([np.sum(heatmaps[:,v], axis=1, keepdims=True) for v in self.annotations.values()], axis=1))
      labels=list(self.annotations.keys())
      return heatmaps, labels


  def fit_predict(self, X, raw=False):
    prep_X, orig_shape=self._preprocess(X)
    self._factorization.fit(prep_X)
    labels=self._factorization.labels_
    return self._postprocess(labels, X, raw, orig_shape)
  
  def predict(self, X, raw=False):
    prep_X, orig_shape=self._preprocess(X)
    labels=self._factorization.predict(prep_X)
    return self._postprocess(labels, X, raw, orig_shape)
  
  def __repr__(self):
    header='{} catalog:'.format(type(self._factorization))
    return '{}\n\t{}'.format(header, self.annotations)