from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcTPSA
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Lipinski import *
from rdkit.Chem.AtomPairs import Torsions, Pairs
from rdkit.Chem import MACCSkeys 
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys 
import numpy as np
import pandas as pd
from tqdm import tqdm
from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from ogb.graphproppred import Evaluator

dataset = GraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')

split_idx = dataset.get_idx_split()
train_idx = np.array(split_idx["train"])
valid_idx = np.array(split_idx["valid"])
test_idx = np.array(split_idx["test"])

data = pd.read_csv(f"/nfs/home/l_hage26/dataset/ogbg_molhiv/dataset/dataset/dataset/ogbg_molhiv/mapping/mol.csv.gz".replace("-", "_"))
smiles = data["smiles"]
outcome = data.set_index ("smiles").drop(["mol_id"], axis = 1)

smi = data.smiles
mol = [Chem.MolFromSmiles(x) for x in smi]

######Morgan Fingerprint

bit_MFP = np.array([AllChem.GetMorganFingerprintAsBitVect(x,radius = 3, nBits=3*1024) for x in tqdm(mol)], dtype= np.int8)
bit_frame = pd.DataFrame(bit_MFP)
bit_frame = pd.concat([data.HIV_active,bit_frame], axis=1)
bit_frame.head()



######RDKit-FP

rdkbi = {}
rdfp = np.array([Chem.RDKFingerprint(x, maxPath = 5, bitInfo=rdkbi)for x in tqdm(mol)], dtype = np.int8)
rdfp = pd.DataFrame(rdfp)


####MACCS-Keys 


Mac = [MACCSkeys.GenMACCSKeys(x) for x in tqdm(mol)]
Mac = np.array(Mac, dtype=np.int8)
Mac_frame_woHIV= pd.DataFrame(Mac)  


##MACCS-Key + Morgan

Mac_Morg_rd_frame = pd.concat([bit_frame, Mac_frame_woHIV, rdfp], axis=1)


###Creating Correlation-Matrix and eliminating all values above 90% correlationo

fingerprint = Mac_Morg_rd_frame
corr = np.corrcoef(np.transpose(fingerprint))
corr = pd.DataFrame(corr)
mask = np.triu(np.ones_like(corr, dtype=bool))
tri_df = corr.mask(mask)

evaluator = Evaluator(name = "ogbg-molhiv")
Parameters = [None]
np.random.seed(42)
for n in (Parameters):
    
    to_drop = [x for x in tri_df.columns if any(abs(tri_df[x])>0.9)]
    reduced_fingerprint = fingerprint.drop(fingerprint.columns[to_drop], axis = 1)
    
    train_frame = pd.DataFrame(reduced_fingerprint.iloc[train_idx,:])
    test_frame = pd.DataFrame(reduced_fingerprint.iloc[test_idx,:])
    val_frame = pd.DataFrame(reduced_fingerprint.iloc[valid_idx,:])

    val_x = val_frame.iloc[:,1:]
    val_y = val_frame.iloc[:,0]
    test_x = test_frame.iloc[:,1:]
    test_y = test_frame.iloc[:,0]
    
    score_val = []
    score_test = []
    test_erg = []
    val_erg = []
    
    Epochs = 10
    for i in range(Epochs):
        inactive_mol = train_frame.loc[train_frame['HIV_active'] == 0]
        active_mol = train_frame.loc[train_frame['HIV_active']==1]
        inactive_short = inactive_mol.sample(frac= 0.3, random_state=i)
        short_list = pd.concat([active_mol, inactive_short], axis = 0)
        short_list = short_list.sample(frac=1.0, random_state=i)
        train_x = short_list.iloc[:,1:]
        train_y = short_list.iloc[:,0]
        
        rf = RandomForestClassifier(n_estimators=500,
                                    random_state=i, 
                                    n_jobs = 4, 
                                    min_samples_leaf=2,
                                    criterion="entropy",
                                    min_impurity_decrease=0,
                                    warm_start= True,
                                    max_features = "auto",
                                    max_depth = None,
                                    min_samples_split=10
                                    )
        rf.fit(train_x, train_y)
       
        y_hat = rf.predict_proba(val_x)[:,1]
        erg = roc_auc_score(val_y, y_hat)
        val_erg.append(erg)
        
        y_hat_test = rf.predict_proba(test_x)[:,1]
        erg_test = roc_auc_score(test_y,y_hat_test)
        test_erg.append(erg_test)
        
       
        val_y = np.array(val_y)
        input_dict_val = {"y_true":val_y.reshape(val_y.shape[0],1),"y_pred":y_hat.reshape(y_hat.shape[0],1)}
        score_val.append(evaluator.eval(input_dict_val)[dataset.eval_metric])
        
       
        test_y = np.array(test_y)
        input_dict = {"y_true":test_y.reshape(test_y.shape[0],1),"y_pred":y_hat_test.reshape(y_hat_test.shape[0],1)}
        score_test.append(evaluator.eval(input_dict)[dataset.eval_metric])
        
        
        print("\n Rand:",i, "Param:",n,"\n Val:\n","%.4f"%erg)
        print("Test:\n","%.4f"%erg_test)  
       
    print("===========================================================================")
    print("Val:\n", "%.4f"%np.mean(val_erg), "Std.-Abweichung:", "%.4f"%np.std(val_erg))
    print("Test:\n", "%.4f"%np.mean(test_erg), "Std.-Abweichung:", "%.4f"%np.std(test_erg))
    print("===========================================================================")

print("Validation--Final results:\n",np.mean(score_val),"+/-", np.std(score_val))
print("Test--Final results:\n", np.mean(score_test), "+/-", np.std(score_test))




   
