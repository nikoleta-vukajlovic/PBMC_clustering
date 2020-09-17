#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from os.path import join as join


# In[39]:


df = pd.read_csv('reports.csv')
df.head()


# In[40]:


df = pd.concat([df.loc[df['dataset'] == f'data-{i+1}'].sort_values(by=['score'], ascending = False).head(3) for i in range(25)])
df.shape


# In[50]:


print(df.to_latex(columns=['dataset', 'parameter', 'score'], bold_rows=True, index=False, longtable=True))


# In[57]:


df_groups = pd.read_csv('report_groups.csv')

df_groups = pd.concat([df_groups.loc[df_groups['dataset'] == f'./groups/group_{i}'].sort_values(by=['score'], ascending = False).head(3) for i in range(3)])
df_groups.shape

print(df_groups.to_latex(columns=['dataset', 'parameter', 'score'], bold_rows=True, index=False, longtable=True))


# In[88]:


gsm_df = pd.read_csv('save/final_GSM3396183.csv/kmeans_2/labels.csv')
zeros = (gsm_df['labels'] == 1).count()

counts = gsm_df['labels'].value_counts().to_frame().transpose()
counts.head()


# In[5]:


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
save_folder = './save'


# In[6]:


folders = ['kmeans_2', 'kmeans_3', 'kmeans_4', 'jaccard_ward', 'jaccard_average', 'cosine_ward', 'cosine_average', 'jaccard_3', 'jaccard_5', 'discretize_2', 'discretize_3', 'discretize_4']
save_pref = [join(save_folder, sample[sample.rfind('/')+1:]) for sample in samples]
report = pd.DataFrame(columns=['dataset', 'parameter'])

for i,sample in enumerate(samples):
    sample_name = sample[sample.rfind('_')+1:-3]
    for f in folders:
        cluster_type = f

        folder = join(save_pref[i], cluster_type)
        
        label_path = join(folder, 'labels.csv')

        df_labels = pd.read_csv(label_path)
        df_labels = df_labels.drop('Unnamed: 0', axis=1)
        
        counts = df_labels['labels'].value_counts().to_frame().transpose()
        
        
        


# In[10]:


import os

def calculate_stats(df, save_path):
    labels = df['label'].unique()
    label_counts = df['label'].value_counts()
    sample_counts = df['sample'].value_counts()

    sample_counts.to_csv(os.path.join(save_path, 'cells_per_sample.csv'))
    label_counts.to_csv(os.path.join(save_path, 'cells_per_cluster.csv'))

    for label in labels:
        tmp = df.loc[df['label'] == label]['sample'].value_counts()
        tmp.to_csv(os.path.join(save_path, f'cluster{label}.csv'))

        
folders = ['kmeans_2', 'kmeans_3', 'kmeans_4', 'jaccard_ward', 'jaccard_average', 'cosine_ward', 'cosine_average', 'discretize_2', 'discretize_3', 'discretize_4']
methods = [3]

paths = [os.path.join('./', f'group_{i}.csv') for i in range(3)]
print(paths)

for i, path in enumerate(paths):
    
    df = pd.read_csv(path)
    stats_df = pd.DataFrame(columns=['sample', 'label', 'cluster'])
    stats_df['sample'] = df['Unnamed: 0'].apply(lambda x: x[:x.find('_')])
    for folder in folders:
        save_prefix = f'./groups/group_{i}/{folder}'
        labels_df = pd.read_csv(os.path.join(save_prefix, f'labels.csv'))
        stats_df['label'] = labels_df['labels']
        calculate_stats(stats_df, save_prefix)
        print(save_prefix)


# In[ ]:




