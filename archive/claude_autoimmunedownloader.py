import pandas as pd
import requests
from Bio import Entrez
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Optional

class AutoimmuneDatasetFinder:
    def __init__(self, email: str):
        """
        Initialize the dataset finder with your email (required for NCBI queries)
        """
        Entrez.email = email
        self.autoimmune_terms = [
            "multiple sclerosis",
            "rheumatoid arthritis",
            "systemic lupus erythematosus",
            "type 1 diabetes",
            "inflammatory bowel disease",
            "celiac disease",
            "psoriasis",
            "graves disease",
            "myasthenia gravis",
            "sjogren's syndrome",
            "autoimmune thyroiditis",
            "systemic sclerosis"
        ]
    
    def search_geo_datasets(self, max_results: int = 1000) -> List[Dict]:
        """
        Search GEO for single-cell RNA-seq datasets related to autoimmune diseases
        """
        datasets = []
        
        for disease in self.autoimmune_terms:
            query = f'({disease}[Title/Abstract]) AND ' \
                   f'("single cell RNA"[Title/Abstract] OR "scRNA-seq"[Title/Abstract]) AND ' \
                   f'"expression profiling by high throughput sequencing"[DataSet Type]'
            
            # Search GEO
            handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            print(f"prog: finished query for disease {disease}")
            
            if int(record["Count"]) > 0:
                for gds_id in record["IdList"]:
                    # Get detailed information
                    handle = Entrez.esummary(db="gds", id=gds_id)
                    summary = Entrez.read(handle)
                    handle.close()
                    
                    if summary:
                        entry = summary[0]
                        title = entry.get("title")
                        print(f"prog: downloaded dataset {title}")
                        dataset_info = {
                            "accession": entry.get("Accession"),
                            "title": title,
                            "disease": disease,
                            "platform": entry.get("platform"),
                            "samples": entry.get("samples"),
                            "gse": entry.get("GSE"),
                            "summary": entry.get("summary")
                        }
                        datasets.append(dataset_info)
            
            # Be nice to NCBI servers
            time.sleep(1)
            
            print(f"prog: finished dataset {disease}")
        
        return datasets
    
    def export_dataset_list(self, output_file: str = "autoimmune_scrna_datasets.csv"):
        """
        Gather datasets from all sources and export to CSV
        """
        # Collect datasets
        geo_datasets = self.search_geo_datasets()
        arrayexpress_datasets = self.search_arrayexpress()
        
        # Combine and create DataFrame
        all_datasets = geo_datasets + arrayexpress_datasets
        df = pd.DataFrame(all_datasets)
        
        # Add download availability column
        df["download_url"] = df.apply(self._generate_download_url, axis=1)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        return df
    
    def _generate_download_url(self, row: pd.Series) -> Optional[str]:
        """
        Generate download URLs for datasets
        """
        accession = row["accession"]
        if accession.startswith("GSE"):
            return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        elif accession.startswith("E-"):
            return f"https://www.ebi.ac.uk/arrayexpress/experiments/{accession}/files/"
        return None

"""
if __name__ == "__main__":
    finder = AutoimmuneDatasetFinder(email="your.email@institution.edu")
    datasets_df = finder.export_dataset_list()
    print(f"Found {len(datasets_df)} datasets")
    print("\nSample of datasets found:")
    print(datasets_df[["accession", "disease", "samples"]].head())
"""