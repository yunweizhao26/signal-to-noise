import cellxgene_census
import scanpy as sc
from scanpy_pipeline import run_pipeline

cellxgene_census.download_source_h5ad(
	"d8da613f-e681-4c69-b463-e94f5e66847f", 
	"test.h5ad"
)
adata = sc.read_h5ad("test.h5ad")
run_pipeline("test", adata, "Homo sapiens")