import cellxgene_census
from pathlib import Path
from autoimmunechecker import AutoimmuneChecker
import scanpy as sc
import os
import json

def download_sets_from_cellxgene(census):
    checker = AutoimmuneChecker(api_key="")
    
    census_datasets = (
        census["census_info"]["datasets"]
        .read(column_names=[
            "collection_name", 
            "dataset_title", 
            "dataset_id",
            #"dataset_h5ad_path", 
            #"dataset_total_cell_count",
            "soma_joinid"])
        .concat()
        .to_pandas()
    )
    
    setlist_path = "./sets/list.json"
    if os.path.exists(setlist_path):
        with open(setlist_path, "r") as f:
            setlist = json.load(f)
    else:
        setlist = {}
        
    download_sets_from_cellxgene_inner(census_datasets, setlist, checker)
    
    with open(setlist_path, "w") as f:
        json.dump(setlist, f, indent=4)
    checker.save_diseases()
    
def download_sets_from_cellxgene_inner(census_datasets, setlist, checker):
    count = 0
    for set in census_datasets.itertuples():
        id = set.dataset_id
        name = set.dataset_title
        to_path = f"./sets/{id}.h5ad"
        
        if id in setlist.keys():
            continue
        
        if count == 50:
            break
        count += 1
        
        print(f"Downloading dataset '{name}' ({id})")
        
        cellxgene_census.download_source_h5ad(id, to_path=to_path)
        setlist[id] = [set.collection_name, name]
        
        adata = sc.read_h5ad(to_path)
        
        diseases = adata.obs['disease'].unique().tolist()
        setlist[id].append(diseases)

        if (not any(checker.is_autoimmune_disease(x) for x in diseases)):
            print(f"Dataset '{name}' doesn't contain any autoimmune diseases")

with open("./sets/list.json", "r") as f:
    setlist = json.load(f)

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if filename != "list.json":
        print(filename in setlist.keys())
        
        adata = sc.read_h5ad(filepath)
        diseases = adata.obs['disease'].unique().tolist()
        if (not any(checker.is_autoimmune_disease(x) for x in diseases)):
            print(f"Dataset '{filename}' doesn't contain any autoimmune diseases")

def organize_by_metadata(adata):    
    for species in adata.obs['species'].unique():
        species_data = adata[adata.obs['species'] == species]
    
        for tissue in disease_data.obs['tissue'].unique():
            tissue_data = disease_data[disease_data.obs['tissue'] == tissue]
                    
            for disease in species_data.obs['disease'].unique():
                disease_data = species_data[species_data.obs['disease'] == disease]
                
                # Create directory path
                save_path = Path("./sets") / species / tissue / disease
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Save the data
                save_path /= f'{id}.h5ad'
                tissue_data.write(save_path)
                
                print(f"Finished writing to '{f'{id}.h5ad'}'")



""""""
        #organism_name = "Homo sapiens" if set.dataset_id in human_sets else "Mus musculus"
        #adata = cellxgene_census.get_anndata(census,
        #                            organism=organism_name,
        #                            obs_value_filter=f"dataset_id=='{set.dataset_id}'")
        
        # adata.write_h5ad(f"./sets/{set.dataset_id}")
        #if (adata.n_obs == 0):
        #    print(f"boo, '{set.dataset_name}' doesn't contain any observations")
        #    continue
        #else:
        #    print(f"hooray! '{set.dataset_name}' has observations!")
        #    break

import pickle

with open("lung_dataset_cell_counts","rb") as inp:
    lung_dataset_cell_counts = pickle.load(inp)

top_lung_datasets_ids = lung_dataset_cell_counts.index[:5].to_list()


#lung_obs = cellxgene_census.get_obs(
#    census, "homo_sapiens", value_filter="tissue_general == 'lung' and is_primary_data == True"
#)
"""
inp = open("lung_obs.pkl","rb")
lung_obs = pickle.load(inp)
inp.close()
lung_obs

census_datasets = (
    census["census_info"]["datasets"]
    .read(column_names=["collection_name", "dataset_title", "dataset_id", "soma_joinid"])
    .concat()
    .to_pandas()
)
census_datasets = census_datasets.set_index("dataset_id")
#census_datasets

dataset_cell_counts = pd.DataFrame(lung_obs[["dataset_id"]].value_counts())
dataset_cell_counts = dataset_cell_counts.rename(columns={0: "cell_counts"})
dataset_cell_counts = dataset_cell_counts.merge(census_datasets, on="dataset_id")

#dataset_cell_counts
"""