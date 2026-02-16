import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from functions import laplacian, MDS


if __name__ == '__main__':

    k = 3 # number of nearest neighbors
    T = 5000 # Temperaturs parameter
    display_weighted = True
    display_unweighted = True

    # Initiallizing data
    D = np.array([
    [0,   65,   84,   87,  156,  199,   66,  142,  161],
    [65,    0,  150,  138,  185,  236,  126,   99,  120],
    [84,  150,    0,   16,  103,  139,   66,  113,  235],
    [87,  138,   16,    0,  100,   82,   64,  111,  181],
    [156,  185,  103,  100,    0,  155,  100,   59,  123],
    [199,  236,  139,   82,  155,    0,  144,  198,  280],
    [66,  126,   66,   64,  100,  144,    0,   77,  140],
    [142,   99,  113,  111,   59,  198,   77,    0,   71],
    [161,  120,  235,  181,  123,  280,  140,   71,    0]])

    labels = ['Denver', 'Fort Collins', 'Breckenridge', 'Copper Mtn', 'Steamboat', 'Aspen', 'Winter Park', 'Walden', 'Snowy Range']


    # Laplacian embedding weighted and unweighted
    eigval, eigvec = laplacian(X = D, weighted=False)

    eigvalw, eigvecw = laplacian(X = D, weighted=True)

    # 2D embeddings
    embedding = eigvec[:, 1:3]
    embeddingw = eigvecw[:, 1:3]

    # Plotting 
    plt.figure(figsize=(8,6))

    # flipping the first and second axes in order to replicated traditional maps more accurately
    if display_unweighted:
        plt.scatter(embedding[:,1], embedding[:,0], color='red', label='Unweighted', s=60)
        for i, city in enumerate(labels):
            plt.text(embedding[i,1]+.01, embedding[i,0]+.01, city)
    if display_weighted:
        plt.scatter(embeddingw[:,1], embeddingw[:,0], color='blue', label='Weighted', s=60)
        for i, city in enumerate(labels):
            plt.text(embeddingw[i,1]+.01, embeddingw[i,0]+.01, city)


    plt.xlabel('Eigenvector 2')
    plt.ylabel('Eigenvector 3')
    plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # MDS embedding
    X, _,_ = MDS(D)
    
    # flipping the first and second axes in order to replicated traditional maps more accurately
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 1], -X[:, 0])
    ax = plt.gca()
    ax.set_xlim([-100, 120])
    ax.set_ylim([-150, 150])
    plt.xlabel("Second Embedding Dimension")
    plt.ylabel("First Embedding Dimension")
    plt.title("2D MDS embedding of Ski Town Distances")
    for i, city in enumerate(labels):
        plt.text(X[i,1]+ 2, -X[i,0] + 2, city)
    plt.show()
