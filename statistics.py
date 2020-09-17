import pandas as pd
import sklearn.metrics as metrics
from os.path import join as join

samples = ['/sbgenomics/project-files/final_GSM3396161.csv',
            '/sbgenomics/project-files/final_GSM3396162.csv', '/sbgenomics/project-files/final_GSM3396163.csv',
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
report = pd.DataFrame(columns=['dataset', 'metric', 'parameter', 'score'])

n_clusters = [2, 3, 4]
save_folder = './save'
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]

folders = ['kmeans_2', 'kmeans_3', 'kmeans_4', 'jaccard_ward', 'jaccard_average', 'cosine_ward', 'cosine_average', 'discretize_2', 'discretize_3', 'discretize_4']
report = pd.DataFrame(columns=['dataset', 'metric', 'parameter', 'score'])

save_pref = [join('./groups/', f'group_{i}') for i in range(3)]
print(save_pref)

for i in range(3):
    df = pd.read_csv(f'./group_{i}.csv', index_col=['Unnamed: 0'])
    
    sample_path = f'data-{i+1}'
    for f in folders:
        cluster_type = f
        folder = join(save_pref[i], cluster_type)
        print(folder)
        label_path = join(folder, 'labels.csv')

        df_labels = pd.read_csv(label_path)
        df_labels = df_labels.drop('Unnamed: 0', axis=1)
        print(df_labels.shape)

        score = metrics.silhouette_score(df, df_labels['labels'], metric='euclidean')

        report = report.append(
            { 'dataset': join('./groups/', f'group_{i}'), 'metric': 'euclidean', 'parameter': cluster_type, 'score': score}, 
            ignore_index=True
        )


report.to_csv('report_groups.csv')

folders = ['kmeans_2', 'kmeans_3', 'kmeans_4', 'jaccard_ward', 'jaccard_average', 'cosine_ward', 'cosine_average', 'jaccard_3', 'jaccard_5', 'discretize_2', 'discretize_3', 'discretize_4']

for i,sample in enumerate(samples):
    df = pd.read_csv(sample)
    index = df.columns[1:]
    df = df.set_index('Index').transpose()
    df.index = index 
    
    print(df.shape)

    sample_path = f'data-{i+1}'
    
    print(save_pref[i])

    for f in folders:
        cluster_type = f

        folder = join(save_pref[i], cluster_type)
        print(folder)
        label_path = join(folder, 'labels.csv')

        df_labels = pd.read_csv(label_path)
        df_labels = df_labels.drop('Unnamed: 0', axis=1)
        print(df_labels.shape)

        score = metrics.silhouette_score(df, df_labels['labels'], metric='euclidean')

        report = report.append(
            { 'dataset': sample_path, 'metric': 'euclidean', 'parameter': cluster_type, 'score': score}, 
            ignore_index=True
        )


report.to_csv('reports.csv')