import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
def plot_freq(f):
  max = 0
  for k in range(f.shape[0]):
    if (f[k] > max):
      max = f[k]
  for k in range(f.shape[0]):
    print('{0:5d} {1} {2}'.format(k, '+' * int(f[k]/max*50), f[k]))
def plot_orig_f(orig,start,n,file):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i])
    plt.title("original ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.savefig(file)
def plot_orig(orig,start,n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i])
    plt.title("original ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_orig_scale(s,orig,start,n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i]*s)
    plt.title("original ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_orig_scale_auto(orig,start,n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    max = np.array([0.0,0.0,0.0])
    for j in range(orig[start+i].shape[0]):
      for k in range(orig[start+i].shape[1]):
        if (orig[start+i,j,k,0] > max[0]):
          max[0] = orig[start+i,j,k,0]
        if (orig[start+i,j,k,1] > max[1]):
          max[1] = orig[start+i,j,k,1]
        if (orig[start+i,j,k,2] > max[2]):
          max[2] = orig[start+i,j,k,2]
    norm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig[start+i].shape[0]):
      for k in range(orig[start+i].shape[1]):
        if (max[0] > 0.0):
           norm[j,k,0] = orig[start+i,j,k,0]/max[0]
        if (max[1] > 0.0):
           norm[j,k,1] = orig[start+i,j,k,1]/max[1]
        if (max[2] > 0.0):
           norm[j,k,2] = orig[start+i,j,k,2]/max[2]
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(norm)
    plt.title("original ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_h(orig,dec,start,n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i])
    plt.title("original ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(dec[start+i])
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_h1(orig,dec,start,n):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i])
    plt.title("original ["+str(start+i)+"]")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(dec[start+i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_h1_f(orig,dec,start,n,file):
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(orig[start+i])
    plt.title("original ["+str(start+i)+"]")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(dec[start+i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.savefig(file)
def plot_h6(orig,dec,start,n):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    # display original feature 0-2
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(orig[start+i,:,:,0:3])
    plt.title("original(0-2) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 0-2
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(dec[start+i,:,:,0:3])
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original feature 3-5
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(orig[start+i,:,:,3:6])
    plt.title("original(3-5) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 3-5
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(dec[start+i,:,:,3:6])
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_h6_f(orig,dec,start,n,file):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    # display original feature 0-2
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(orig[start+i,:,:,0:3])
    plt.title("original(0-2) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 0-2
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(dec[start+i,:,:,0:3])
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original feature 3-5
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(orig[start+i,:,:,3:6])
    plt.title("original(3-5) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 3-5
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(dec[start+i,:,:,3:6])
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.savefig(file)
def plot_h6_mix_neg(orig,dec,start,n):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    # display original feature 0,2,4
    ax = plt.subplot(4, n, i + 1)
    norm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        norm[j,k,0] = (1-orig[start+i,j,k,0])
        norm[j,k,1] = (1-orig[start+i,j,k,2])
        norm[j,k,2] = (1-orig[start+i,j,k,4])
    plt.imshow(norm)
    plt.title("orig(0,2,4) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 0,2,4
    ax = plt.subplot(4, n, i + 1 + n)
    denorm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        denorm[j,k,0] = (1-dec[start+i,j,k,0])
        denorm[j,k,1] = (1-dec[start+i,j,k,2])
        denorm[j,k,2] = (1-dec[start+i,j,k,4])
    plt.imshow(denorm)
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original feature 1,3,5
    ax = plt.subplot(4, n, i + 1 + 2*n)
    norm1 = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        norm1[j,k,0] = (1-orig[start+i,j,k,1])
        norm1[j,k,1] = (1-orig[start+i,j,k,3])
        norm1[j,k,2] = (1-orig[start+i,j,k,5])
    plt.imshow(norm1)
    plt.title("orig(1,3,5) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction feature 1,3,5
    ax = plt.subplot(4, n, i + 1 + 3*n)
    denorm1 = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        denorm1[j,k,0] = (1-dec[start+i,j,k,1])
        denorm1[j,k,1] = (1-dec[start+i,j,k,3])
        denorm1[j,k,2] = (1-dec[start+i,j,k,5])
    plt.imshow(denorm1)
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_h6_mix_neg_emb(orig,dec,emb,start,n,file):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    #
    # display original feature 0,2,4
    #
    ax = plt.subplot(5, n, i + 1)
    norm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        norm[j,k,0] = (1-orig[start+i,j,k,0])
        norm[j,k,1] = (1-orig[start+i,j,k,2])
        norm[j,k,2] = (1-orig[start+i,j,k,4])
    plt.imshow(norm)
    plt.title("orig(0,2,4) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display reconstruction feature 0,2,4
    #
    ax = plt.subplot(5, n, i + 1 + n)
    denorm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        denorm[j,k,0] = (1-dec[start+i,j,k,0])
        denorm[j,k,1] = (1-dec[start+i,j,k,2])
        denorm[j,k,2] = (1-dec[start+i,j,k,4])
    plt.imshow(denorm)
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # diplay embedding
    #
    ax = plt.subplot(5, n, i + 1 + 2*n)
    embnorm = np.zeros((emb.shape[1],emb.shape[2],3))
    for j in range(emb.shape[1]):
      for k in range(emb.shape[2]):
        embnorm[j,k,0] = (1-emb[start+i,j,k,0])
        if (embnorm[j,k,0] < 0):
          embnorm[j,k,0] = 0
        if (embnorm[j,k,0] > 1):
          embnorm[j,k,0] = 1
        if (emb.shape[3] > 1):
          embnorm[j,k,1] = (1-emb[start+i,j,k,1])
          if (embnorm[j,k,1] < 0):
            embnorm[j,k,1] = 0
          if (embnorm[j,k,1] > 1):
            embnorm[j,k,1] = 1
        else:
          embnorm[j,k,1] = 0
        if (emb.shape[3] > 2):
          embnorm[j,k,2] = (1-emb[start+i,j,k,2])
          if (embnorm[j,k,2] < 0):
            embnorm[j,k,2] = 0
          if (embnorm[j,k,2] > 1):
            embnorm[j,k,2] = 1
        else:
          embnorm[j,k,2] = 0
    plt.imshow(embnorm)
    plt.title("embedding")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display original feature 1,3,5
    #
    ax = plt.subplot(5, n, i + 1 + 3*n)
    norm1 = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        norm1[j,k,0] = (1-orig[start+i,j,k,1])
        norm1[j,k,1] = (1-orig[start+i,j,k,3])
        norm1[j,k,2] = (1-orig[start+i,j,k,5])
    plt.imshow(norm1)
    plt.title("orig(1,3,5) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display reconstruction feature 1,3,5
    #
    ax = plt.subplot(5, n, i + 1 + 4*n)
    denorm1 = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        denorm1[j,k,0] = (1-dec[start+i,j,k,1])
        denorm1[j,k,1] = (1-dec[start+i,j,k,3])
        denorm1[j,k,2] = (1-dec[start+i,j,k,5])
    plt.imshow(denorm1)
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  if (file != 'XXX'):
    plt.savefig(file)
  else:
    plt.show()

def plot_h6_mix_neg_emb_g(orig,dec,emb,start,n):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    #
    # display original feature 0
    #
    ax = plt.subplot(5, n, i + 1)
    norm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        norm[j,k,0] = (1-orig[start+i,j,k,0])
        norm[j,k,1] = 1
        norm[j,k,2] = 1
    plt.imshow(norm)
    plt.title("orig ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display reconstruction feature 0
    #
    ax = plt.subplot(5, n, i + 1 + n)
    denorm = np.zeros((orig.shape[1],orig.shape[2],3))
    for j in range(orig.shape[1]):
      for k in range(orig.shape[2]):
        denorm[j,k,0] = (1-dec[start+i,j,k,0])
        denorm[j,k,1] = 1
        denorm[j,k,2] = 1
    plt.imshow(denorm)
    plt.title("reconstructed")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # diplay embedding
    #
    ax = plt.subplot(5, n, i + 1 + 2*n)
    embnorm = np.zeros((emb.shape[1],emb.shape[2],3))
    for j in range(emb.shape[1]):
      for k in range(emb.shape[2]):
        embnorm[j,k,0] = (1-emb[start+i,j,k,0])
        if (embnorm[j,k,0] < 0):
          embnorm[j,k,0] = 0
        if (embnorm[j,k,0] > 1):
          embnorm[j,k,0] = 1
        embnorm[j,k,1] = (1-emb[start+i,j,k,1])
        if (embnorm[j,k,1] < 0):
          embnorm[j,k,1] = 0
        if (embnorm[j,k,1] > 1):
          embnorm[j,k,1] = 1
        embnorm[j,k,2] = 1
    plt.imshow(embnorm)
    plt.title("embedding")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

def plot_x(x,y,lmod,gmod,start,n):
  # load local decoder
  local_m = keras.models.load_model(lmod)
  global_m = keras.models.load_model(gmod)
  l_emb = x[start:start+n,:,:,0:3]
  g_emb = x[start:start+n,:,:,3:5]
  rq_emb = x[start:start+n,:,:,5:7]
  y_sub = y[start:start+n]  
  plt.figure(figsize=(20, 8))
  for i in range(n):
    l_hist = local_m.decoder(l_emb[i].reshape((-1,3072)))
    g_hist = global_m.decoder(g_emb[i].reshape((-1,2048)))
    rq_hist = global_m.decoder(rq_emb[i].reshape((-1,2048)))
    #
    # display local histogram features 0,2,4
    #
    ax = plt.subplot(5, n, i + 1)
    
    norm = np.zeros((128,128,3))
    for j in range(128):
      for k in range(128):
        norm[j,k,0] = (1-l_hist[0,j,k,0])
        norm[j,k,1] = (1-l_hist[0,j,k,2])
        norm[j,k,2] = (1-l_hist[0,j,k,4])
    plt.imshow(norm)
    plt.title("LHist(0,2,4) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display global histogram
    #
    ax = plt.subplot(5, n, i + 1 + n)
    normg = np.zeros((128,128,3))
    for j in range(128):
      for k in range(128):
        normg[j,k,0] = (1-g_hist[0,j,k,0])
        normg[j,k,1] = 1
        normg[j,k,2] = 1
    plt.imshow(normg)
    plt.title("GHist ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display global histogram RQ
    #
    ax = plt.subplot(5, n, i + 1 + 2*n) 
    normrq = np.zeros((128,128,3))
    for j in range(128):
      for k in range(128):
        if (rq_hist[0,j,k,0]>0.001):
          normrq[j,k,0] = 1
        else:
          normrq[j,k,0] = 0
        normrq[j,k,1] = 0
        normrq[j,k,2] = 0
    plt.imshow(normrq)
    plt.title("RQ "+str(y_sub[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
def plot_x_rq(x,y,lmod,gmod,start,n):
  # load local decoder
  local_m = keras.models.load_model(lmod)
  global_m = keras.models.load_model(gmod)
  l_emb = x[start:start+n,:,:,0:3]
  g_emb = x[start:start+n,:,:,3:5]
  rq_hist = x[start:start+n,:,:,5:6]
  y_sub = y[start:start+n]
  plt.figure(figsize=(20, 8))
  for i in range(n):
    l_hist = local_m.decoder(l_emb[i].reshape((-1,3072)))
    g_hist = global_m.decoder(g_emb[i].reshape((-1,2048)))
    #
    # display local histogram features 0,2,4
    #
    ax = plt.subplot(5, n, i + 1)

    norm = np.zeros((128,128,3))
    for j in range(128):
      for k in range(128):
        norm[j,k,0] = (1-l_hist[0,j,k,0])
        norm[j,k,1] = (1-l_hist[0,j,k,2])
        norm[j,k,2] = (1-l_hist[0,j,k,4])
    plt.imshow(norm)
    plt.title("LHist(0,2,4) ["+str(start+i)+"]")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display global histogram 
    #
    ax = plt.subplot(5, n, i + 1 + n)
    normg = np.zeros((128,128,3))
    for j in range(128):
      for k in range(128):
        normg[j,k,0] = (1-g_hist[0,j,k,0])
        normg[j,k,1] = 1
        normg[j,k,2] = 1
    plt.imshow(normg)
    plt.title("GHist ["+str(start+i)+"]")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #
    # display global histogram RQ
    #
    ax = plt.subplot(5, n, i + 1 + 2*n) 
    normrq = np.zeros((32,32,3))
    for j in range(32):
      for k in range(32):
        if (rq_hist[i,j,k,0]>0.0001):
          normrq[j,k,0] = 1
        else:
          normrq[j,k,0] = 0
        normrq[j,k,1] = 0
        normrq[j,k,2] = 0
    plt.imshow(normrq)
    plt.title("RQ "+str(y_sub[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()
