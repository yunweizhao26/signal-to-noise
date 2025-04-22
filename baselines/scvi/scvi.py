import scvi
import anndata
import numpy as np

def run_scvi(data, save_path, file_extension='', target_file=None):
    adata = anndata.AnnData(data)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata)
    model.train()
    x = model.get_normalized_expression(return_numpy=True)
    np.save(save_path + "scvi" + file_extension, x)
    
