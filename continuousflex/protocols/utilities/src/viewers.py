# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import os
import matplotlib.pyplot as plt
plt.style.use("bmh")
import numpy as np
import RamachanDraw
import continuousflex.protocols.utilities.src as src
import warnings


def molecule_viewer(mol, names=None):
    """
    Matplotlib Viewer for Molecules
    :param mol: Molecule, list of Molecule
    :param names: list of names
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    legend=[]
    if isinstance(mol, list):
        for i in range(len(mol)):
            coord=  mol[i].coords
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2])
            if names is not None:
                legend.append(names[i])
    else:
        for j in range(mol.n_chain):
            coord = mol.get_chain(j)
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2])
        if names is not None:
            legend.append(names)
    ax.legend(legend)

def chimera_molecule_viewer(mol):
    """
    Chimera Viewer for Molecules
    :param mol: Molecule, list of Molecule
    """
    if isinstance(mol, list):
        cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \""
        for i in range(len(mol)):
            mol[i].save_pdb( "mol"+str(i)+".pdb")
            cmd+= "open mol"+str(i)+".pdb ; "
        cmd+="hide atoms ; show cartoons\""
    else:
        mol.save_pdb("mol.pdb")
        cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \"open mol.pdb ; hide atoms ; show cartoons\""
    os.system(cmd)

def image_viewer(img):
    """
    Matplotlib viewer for Image
    :param img: Image to show
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.data, cmap='gray')

def volume_viewer(vol):
    """
    Matplotlib viewer for Volume
    :param vol: Volume to show
    """
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    volume= vol.data
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

def fit_viewer(fit, save=None):
    """
    Matplotlib Vierwer for fitting statistics
    :param fit: FlexibleFitting
    :param save: path (str) where to save the figure (PNG)
    """
    if isinstance(fit.fit, list):
        fits = fit.fit
    else:
        fits = [fit.fit]
    fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    for i in range(len(fits)):
        ax[0, 0].plot(fits[i]['U'])
        ax[0, 1].plot(fits[i]['U_potential'])
        ax[0, 2].plot(fits[i]['U_biased'])
        ax[1, 0].plot(np.array(fits[i]['K']) + np.array(fits[i]['U']))
        ax[1, 0].plot(fits[i]['U'])
        ax[1, 0].plot(fits[i]['K'])
        ax[1, 1].plot(fits[i]['C'])
        ax[1, 2].plot(fits[i]['CC'])
        ax[0, 3].plot(fits[i]['T'])
        if 'RMSD' in fits[i]:
                ax[1, 3].plot(fits[i]['RMSD'])
    ax[0, 0].set_title('U')
    ax[0, 1].set_title('U_potential')
    ax[0, 2].set_title('U_biased')
    ax[1, 0].set_title('H=U+K')
    ax[1, 1].set_title('Criterion')
    ax[1, 2].set_title('CC')
    ax[0, 3].set_title('Temperature')
    ax[1, 3].set_title('RMSD')
    # fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

def fit_potentials_viewer(fit, save=None):
    if isinstance(fit.fit, list):
        fits = fit.fit[0]
    else:
        fits = fit.fit
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i in fit.params["potentials"]:
        if "U_"+i in fits:
            ax.plot(fits["U_"+i], label=i)
    ax.legend()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)


def chimera_fit_viewer(mol, target):
    """
    Chimera vierwer for fitting results
    :param mol: initial Molecule
    :param target: target Density
    """
    mol.save_pdb("mol.pdb")
    target.save_mrc("vol.mrc")
    cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \"open mol.pdb ; open vol.mrc ; volume #2 level 0.7 ; volume #2 transparency 0.7 ; hide atoms ; show cartoons\""
    os.system(cmd)

def ramachandran_viewer(file, save=None):

    if save is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RamachanDraw.plot(file, show=False, save=True, out=save)
            plt.close("all")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RamachanDraw.plot(file)