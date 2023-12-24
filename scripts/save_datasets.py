import os

from aml.dataset import save_dataset,OpenmlDataset, XirisDataset
        
if __name__=="__main__":
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'

    source_dir = data_dir+r"\source"
    save_dir = data_dir+r"\compressed"
    
    xiris_src = source_dir+r"\xiris"
    openml_src = source_dir+r"\openml\openml_28.txt"
        
    save_dataset( OpenmlDataset, openml_src, save_dir )
    save_dataset( XirisDataset, xiris_src, save_dir )
