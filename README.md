# Deep reinforcement learning for de novo synthesis of eye drops
When developing eye drops, it is necessary to take into account not only the biological activity of the drug, but also a number of additional properties that affect the effectiveness of the drug. In this work, we present a novel pipeline based on reinforcement learning for the de novo synthesis of ophthalmic drugs with desired properties. We have shown that molecules generated by our method have higher values ​​of corneal permeability, melanin binding, and low toxicity compared to molecules generated by the model FREED++.
![frame1](https://github.com/AnastasiaVepreva/ophthalmic_drugs/blob/b89e84f13e3b592e7979ff391d52a5c271c350ae/Frame%201.png)
## Setup Python environment
```
# Install python environment
conda env create -f environment.yml
# Activate environment
conda activate freedpp
```
## Usage
```
python main.py     --exp_root ../experiments
 --alert_collections ../alert_collections.csv     --fragments ../zinc_crem.json
 --receptor ../COX-2.pdbqt     --vina_program ./env/qvina02     --starting_smile "O=C(O)C(*)c1c(*)c(*)c(*)c(*)c1(*)"
 --fragmentation crem     --num_sub_proc 12     --n_conf 1     --exhaustiveness 1     --save_freq 10
--epochs 100     --commands "train,sample"     --reward_version soft     --box_center "27.116,24.090,14.936"
--box_size "9.427,10.664,10.533"     --seed 150     --name freedpp
 --objectives "DockingScore,Corneal,Melanin,Irritation"     --weights "1.0,1.0,1.0,1.0"
# You can choose objectives for reward function, i.g., --objectives "DockingScore"
```
