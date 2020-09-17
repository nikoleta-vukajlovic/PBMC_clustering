import pandas as pd
import scipy.cluster.hierarchy as shc 
import scipy.spatial.distance as ssd
from os.path import join as join
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def save_labels(labels, path):
    
    df = pd.DataFrame(labels, columns=['labels'])
    df.to_csv(path_or_buf=path)


def my_dendrogram(*args, **kwargs):
    
    max_d = kwargs.pop('max_d', None)
    title = kwargs.pop('title', None)
    save_path = kwargs.pop('save_path', None)
    
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    plt.figure(figsize=(20, 10))
    data = shc.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title(f'{title} p={p}')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(data['icoord'], data['dcoord'], data['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, 
                             (x, y), 
                             xytext=(0, -5), 
                             textcoords='offset points', 
                             va='top',
                             ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
            
    plt.savefig(save_path)
    plt.close()
    return data


def elbow_agglo_params(Z, save_path):
    
    plt.figure(figsize=(15, 15))
    last = Z[:, 2]
    reverse_last = last[::-1]
    indexs = np.arange(1, len(last) + 1)
    plt.plot(indexs, reverse_last)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    reverse_acceleration = acceleration[::-1]
    plt.plot(indexs[:-2] + 1, reverse_acceleration)
    
    p = reverse_acceleration.argmax() + 2
    
    plt.title(f'{p} clusters')
    
    plt.savefig(save_path)
    plt.close()
    
    return shc.fcluster(Z, p, criterion='maxclust')

save_folder = './save'
p=10

samples = ['/sbgenomics/project-files/final_GSM3396161.csv', '/sbgenomics/project-files/final_GSM3396162.csv', '/sbgenomics/project-files/final_GSM3396163.csv',
           '/sbgenomics/project-files/final_GSM3396164.csv', '/sbgenomics/project-files/final_GSM3396165.csv',
           '/sbgenomics/project-files/final_GSM3396166.csv', '/sbgenomics/project-files/final_GSM3396167.csv',
           '/sbgenomics/project-files/final_GSM3396168.csv', '/sbgenomics/project-files/final_GSM3396169.csv',
           '/sbgenomics/project-files/final_GSM3396170.csv', '/sbgenomics/project-files/final_GSM3396171.csv',
           '/sbgenomics/project-files/final_GSM3396172.csv', '/sbgenomics/project-files/final_GSM3396173.csv',
           '/sbgenomics/project-files/final_GSM3396174.csv', '/sbgenomics/project-files/final_GSM3396175.csv',
           '/sbgenomics/project-files/final_GSM3396176.csv', '/sbgenomics/project-files/final_GSM3396177.csv',
           '/sbgenomics/project-files/final_GSM3396178.csv', '/sbgenomics/project-files/final_GSM3396179.csv',
           '/sbgenomics/project-files/final_GSM3396180.csv', '/sbgenomics/project-files/final_GSM3396181.csv',
           '/sbgenomics/project-files/final_GSM3396182.csv', '/sbgenomics/project-files/final_GSM3396183.csv',
           '/sbgenomics/project-files/final_GSM3396184.csv', '/sbgenomics/project-files/final_GSM3396185.csv'
          ]


# Hijerarhijsko klasterovanje (dist = cosine, jaccard) (methods = average, ward)

save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

distances = ['cosine', 'jaccard']
methods = ['average', 'ward']

for i, sample in enumerate(samples):
    
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index 

    for distance in distances:
        dist_matrix = ssd.pdist(df.values, metric=distance)

        for link in methods:
            Z = shc.linkage(dist_matrix, method=link)
            coph, _ = shc.cophenet(Z, dist_matrix)

            cluster_type = f'{distance}_{link}'
            title = f'data_{i}, shape:{df.shape}:\n{cluster_type}, coph={coph}'

            folder = join(save_pref[i], cluster_type)
            print(folder)
            Path(folder).mkdir(exist_ok=True, parents=True                     

            dendro_save = join(folder, 'dendrogram.jpg')
            my_dendrogram(
                Z,
                truncate_mode='lastp',
                p=p,
                show_contracted=True,
                title=title,
                save_path=dendro_save
            )

            elbow_save = join(folder, f'{cluster_type}_elbow.jpg')
            labels = elbow_agglo_params(Z, elbow_save)
           
            csv_path = join(folder, 'labels.csv')
            save_labels(labels, csv_path)

def dbscan_knee(X, save_path, metric):
    print('p1')
    neigh = NearestNeighbors(n_neighbors=3, metric=metric) 
    neighbors = neigh.fit(X) 
    print('p2')
    distances, indices = neighbors.kneighbors(X)  
    distances = np.sort(distances, axis=0) 
    distances = distances[:,1] 
    
    print('hello before Rotor')
    rotor = Rotor()
    data = np.hstack((np.array(range(df.shape[0])).reshape(-1, 1), distances.reshape(-1, 1)))
    rotor.fit_rotate(data)
    eps = distances[rotor.get_elbow_index()]
    print('p3')
    plt.plot(distances)
    plt.title(f'eps={eps}')
    plt.savefig(f'{save_path}/knee_{metric}.jpg')
    plt.close()
    
    return eps


# DBSCAN (min_points = 3, 4, 5)

save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

distances = ['cosine', 'jaccard']
methods = ['average', 'ward']

for i, sample in enumerate(samples):
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index
    
    for metric in ['jaccard']:
        
        eps = dbscan_knee(df, save_pref[i], metric)

        for min_pts in [3, 5]:
            
            cluster_type = f'{metric}_{min_pts}'
            folder = join(save_pref[i], cluster_type)
            print(folder)
            Path(folder).mkdir(exist_ok=True)

            print('dbscan')
            db = DBSCAN(eps=eps, min_samples=min_pts, metric=metric).fit(df)
            labels = db.labels_
            
            csv_path = join(folder, 'labels.csv')
            save_labels(labels, csv_path)

        print(f'{sample} Done!')


# KMeans (n = 2, 3, 4)

save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

distances = ['cosine', 'jaccard']
methods = ['average', 'ward']

for i, sample in enumerate(samples):

    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index
    
    for n in [2, 3, 4]:
        
        cluster_type = f'kmeans_{n}'
        folder = join(save_pref[i], cluster_type)
        print(folder)
        Path(folder).mkdir(exist_ok=True)

        kmeans = KMeans(n_clusters=n).fit(df)

        labels = kmeans.labels_
        save_labels(labels, join(folder, 'labels.csv'))


# Spektralno klasterovanje (n = 2, 3, 4)

save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

for i, sample in enumerate(samples):

    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index
    
    for n in [2, 3, 4]:
        
        cluster_type = f'discretize_{n}'
        folder = join(save_pref[i], cluster_type)
        Path(folder).mkdir(exist_ok=True)
        print(folder)
        
        spectral = SpectralClustering(n_clusters=n, 
                                    assign_labels='discretize', 
                                    random_state=0,
                                    affinity='nearest_neighbors').fit(df.values)
        
        labels = spectral.labels_
        save_labels(labels, join(folder, 'labels.csv'))


def tsne_visualisation(df, labels, n_components, save_path):
    n = len(set(labels))
    
    tsne = TSNE(n_components, 
#                 perplexity=30,  
                n_iter=300)
    tsne_data = pd.DataFrame(tsne.fit_transform(df))
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_data[0], 
               tsne_data[1],
                   c=labels,
               cmap=ListedColormap(sns.color_palette('Set1', n).as_hex()))
    
    plt.savefig(f'{save_path}_tsne_{n_components}.jpg')
    plt.close()


#agglomerative_clustering visualisation
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

distances = ['cosine', 'jaccard']
methods = ['average', 'ward']

for i, sample in enumerate(samples):
    
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index 
#     df = sns.load_dataset('iris')
#     df = df.drop(['species'], axis=1)

    for distance in distances:
        for link in methods:
            cluster_type = f'{distance}_{link}'
            title = f'data_{i}, shape:{df.shape}:\n{cluster_type}'

            folder = join(save_pref[i], cluster_type)
            label_path = join(folder, 'labels.csv')
            print(label_path)
            labels_df = pd.read_csv(label_path)
            labels = np.array(labels_df['labels'])
            tsne_path = join(folder, f'{cluster_type}')
            tsne_visualisation(df, labels, 2, tsne_path)
            print(f'saved {tsne_path}')


#KMeans visualisation
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]
n_clusters = [2, 3, 4]

for i, sample in enumerate(samples):
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index 

    for n in n_clusters:
        cluster_type = f'kmeans_{n}'

        folder = join(save_pref[i], cluster_type)
        label_path = join(folder, 'labels.csv')
        print(label_path)
        labels_df = pd.read_csv(label_path)
        labels = np.array(labels_df['labels'])
        tsne_path = join(folder, f'{cluster_type}')
        tsne_visualisation(df, labels, 2, tsne_path)
        print(f'saved {tsne_path}')

#DBSCAN visualisation
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]
metrics = ['jaccard']
min_neighbors = [3, 5]

for i, sample in enumerate(samples):
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index 

    for metric in metrics:

        for min_pts in min_neighbors:
            cluster_type = f'{metric}_{min_pts}'
            folder = join(save_pref[i], cluster_type)
            
            label_path = join(folder, 'labels.csv')
            print(label_path)
            labels_df = pd.read_csv(label_path)
            labels = np.array(labels_df['labels'])
            tsne_path = join(folder, f'{cluster_type}')
            tsne_visualisation(df, labels, 2, tsne_path)
            print(f'saved {tsne_path}')

#Spectral visualisation
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

for i, sample in enumerate(samples):

    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index
    
    for n in [2, 3, 4]:
        
        cluster_type = f'discretize_{n}'
        folder = join(save_pref[i], cluster_type)
        label_path = join(folder, 'labels.csv')
        print(label_path)
        labels_df = pd.read_csv(label_path)
        labels = np.array(labels_df['labels'])
        
        tsne_path = join(folder, f'{cluster_type}')
        tsne_visualisation(df, labels, 2, tsne_path)
        print(f'saved {tsne_path}')


groups = [['/sbgenomics/project-files/final_GSM3396161.csv', '/sbgenomics/project-files/final_GSM3396178.csv', '/sbgenomics/project-files/final_GSM3396183.csv', '/sbgenomics/project-files/final_GSM3396185.csv'], ['/sbgenomics/project-files/final_GSM3396169.csv', '/sbgenomics/project-files/final_GSM3396170.csv', '/sbgenomics/project-files/final_GSM3396172.csv', '/sbgenomics/project-files/final_GSM3396162.csv'], ['/sbgenomics/project-files/final_GSM3396163.csv', '/sbgenomics/project-files/final_GSM3396171.csv', '/sbgenomics/project-files/final_GSM3396174.csv', '/sbgenomics/project-files/final_GSM3396177.csv']]

for i,g in enumerate(groups):
    dfs = []
    for file in g:
        df = pd.read_csv(file)
        index = df.columns[1:]
        df = df.set_index('Index').transpose()
        df.index = index
        
        dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv(f'group_{i}.csv')


df = pd.read_csv('./group_1.csv', index_col=['Unnamed: 0'])
df.shape

for i in range(3):
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])
    print(f'group {i}: {df.shape}')

#aglomerative clustering by groups & TSNE
save_pref = [join('./groups/', f'group_{i}') for i in range(3)]
print(save_pref)

distances = ['cosine', 'jaccard']
methods = ['average', 'ward']

for i in range(3):
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])
    

    for distance in distances:
        dist_matrix = ssd.pdist(df.values, metric=distance)

        for link in methods:
            Z = shc.linkage(dist_matrix, method=link)
            coph, _ = shc.cophenet(Z, dist_matrix)

            cluster_type = f'{distance}_{link}'
            title = f'data_{i}, shape:{df.shape}:\n{cluster_type}, coph={coph}'

            folder = join(save_pref[i], cluster_type)
            print(folder)
            Path(folder).mkdir(exist_ok=True, parents=True)                    

            dendro_save = join(folder, 'dendrogram.jpg')
            my_dendrogram(
                Z,
                truncate_mode='lastp',
                p=p,
                show_contracted=True,
                title=title,
                save_path=dendro_save
            )

            elbow_save = join(folder, f'{cluster_type}_elbow.jpg')
            labels = elbow_agglo_params(Z, elbow_save)
           
            csv_path = join(folder, 'labels.csv')
            save_labels(labels, csv_path)
                               
            tsne_path = join(folder, f'{cluster_type}')
            tsne_visualisation(df, labels, 2, tsne_path)
            print(f'saved {tsne_path}')

#KMeans clustering by groups & TSNE
save_pref = [join('./groups/', f'group_{i}') for i in range(3)]
print(save_pref)

for i in range(3):
    
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])
   
    for n in [2, 3, 4]:
        
        cluster_type = f'kmeans_{n}'
        folder = join(save_pref[i], cluster_type)
        print(folder)
        Path(folder).mkdir(exist_ok=True)

        kmeans = KMeans(n_clusters=n).fit(df)

        labels = kmeans.labels_
        save_labels(labels, join(folder, 'labels.csv'))                               
        tsne_path = join(folder, f'{cluster_type}')
        tsne_visualisation(df, labels, 2, tsne_path)
        print(f'saved {tsne_path}')


#DBSCAN group clustering & TSNE
save_pref = [join('./groups/', f'group_{i}') for i in range(3)]
print(save_pref)

for i in range(3):
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])
    for metric in ['jaccard']:
        
        eps = dbscan_knee(df, save_pref[i], metric)
        print(f'group{i} got her eps')
        for min_pts in [3, 5]:
            
            cluster_type = f'{metric}_{min_pts}'
            folder = join(save_pref[i], cluster_type)
            print(folder)
            Path(folder).mkdir(exist_ok=True)

            print('dbscan')
            db = DBSCAN(eps=eps, min_samples=min_pts, metric=metric).fit(df)
            labels = db.labels_
            
            csv_path = join(folder, 'labels.csv')
            save_labels(labels, csv_path)
            tsne_path = join(folder, f'{cluster_type}')
            tsne_visualisation(df, labels, 2, tsne_path)
            print(f'saved {tsne_path}')


save_pref = [join('./groups/', f'group_{i}') for i in range(3)]
print(save_pref)
for i in range(3):
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])

    for n in [2, 3, 4]:
        
        cluster_type = f'discretize_{n}'
        folder = join(save_pref[i], cluster_type)
        Path(folder).mkdir(exist_ok=True)
        print(folder)
        
        spectral = SpectralClustering(n_clusters=n, 
                                    assign_labels='discretize', 
                                    random_state=0,
                                    affinity='nearest_neighbors').fit(df.values)
        
        labels = spectral.labels_
        save_labels(labels, join(folder, 'labels.csv'))
        tsne_path = join(folder, f'{cluster_type}')
        tsne_visualisation(df, labels, 2, tsne_path)
        print(f'saved {tsne_path}')

