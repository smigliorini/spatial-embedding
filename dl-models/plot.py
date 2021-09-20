import matplotlib.pyplot as plt
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
