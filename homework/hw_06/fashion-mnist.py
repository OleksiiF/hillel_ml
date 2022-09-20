import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan


def get_umaped(train_data):
    train_data.label = train_data.label.astype(int)
    reducer = umap.UMAP(
        random_state=13,
        min_dist=0.0000205,
        n_components=2,
        n_neighbors=43
    )
    X = train_data[train_data.columns[1:].values].values
    scaled_mnist_data = StandardScaler().fit_transform(X)

    return reducer.fit_transform(scaled_mnist_data)


train_data = pd.read_csv('fashion-mnist_train.csv')
embedding = get_umaped(train_data)

plt.figure(figsize=(10, 10), dpi=128, num=1)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[
        sns.color_palette()[x]
        for x in train_data.label.map(
            {i: i for i in range(10)}
        )
    ],
    s=0.3,
    cmap='Spectral',
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the MNIST FASHION dataset', fontsize=12)
plt.savefig('embeddings.png')
plt.show()
plt.clf()
# STEP 2: HDBSCAN
labels = hdbscan.HDBSCAN().fit_predict(embedding)  # TODO tune hyper params

clustered = (labels >= 0)

plt.figure(figsize=(10, 10), dpi=128, num=2)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[
        sns.color_palette()[x]
        for x in train_data.label.map(
            {i: i for i in range(len(set(labels)))}
        )
    ],
    s=0.3,
    cmap='Spectral'
)
plt.title('HDBSCAN after UMAP', fontsize=12)
plt.savefig('hdbscan.png')
