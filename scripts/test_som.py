from minisom import MiniSom    
import os

import matplotlib.pyplot as plt

from aml.dataset import load_dataset
import numpy as np

comp_dir = os.getcwd()+r"\data\compressed"
    
dss2 = load_dataset( comp_dir, date=-1, hint="xiris")

data = np.vstack( [ dss2[i]["X"] for i in dss2.keys()])
target = np.hstack( [ dss2[i]["y"].ravel() for i in dss2.keys()])

imgs = dss2["DED-h"]["I"]

# Initialization and training
n_neurons = 5
m_neurons = 5
som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, 
              neighborhood_function='triangle', random_seed=0)

#som.pca_weights_init(data)
som.train_random(data, 1000, verbose=True)  # random training


plt.figure()

plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

# Plotting the response for each pattern in the iris dataset
# different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']

un,inv = np.unique(target,return_inverse=True)


for cnt in range(len(target)):

    w = som.winner(data[cnt])  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[inv[cnt]], markerfacecolor='None',
             markeredgecolor=colors[inv[cnt]], markersize=12, markeredgewidth=2)

plt.show()

import matplotlib.gridspec as gridspec

labels_map = som.labels_map(data, [target[t] for t in inv])

fig = plt.figure(figsize=(9, 9))

the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)

for position in labels_map.keys():
    
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    plt.subplot(the_grid[n_neurons-1-position[1],
                         position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names.values(), bbox_to_anchor=(3.5, 6.5), ncol=3)
plt.savefig('resulting_images/som_seed_pies.png')
plt.show()
