=====================
ContinuousFlex plugin
=====================

This plugin provides the latest Scipion protocols for cryo-EM continuous conformational flexibility/heterogeneity analysis of biomolecular complexes.


Requirements
------------

You will need to use `3.0 <https://github.com/I2PC/scipion/releases>`_ version of Scipion to be able to run these protocols.
If you need help installing Scipion3, please refer to the Scipion Documentation `here <https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html>`__


Installation
------------

To install the plugin, you have two options:

a) Stable version

	Install Scipion3 and use the plugin manager to install the plugin.

b) Developer's version

   * download repository

   .. code-block::

      git clone https://github.com/scipion-em/scipion-em-continuousflex.git
      git checkout devel

   * install

   .. code-block::

      scipion3 installp -p path_to_scipion-em-continuousflex --devel

continuousflex sources will be downloaded automatically with the plugin.


Note: Xmipp and Chimerax plugins should be installed (from Scipion3 plugin manager) to run continuousflex protocols.
You should also consider having VMD on your system for visualization.
We assume that VMD is installed on your system in "/usr/local/lib/vmd".
If VMD is installed but does not work, you may run the command "scipion3 config" and look for VMD_HOME in the config file (the config file is usually at ~/scipion3/config/scipion.conf)

Note: Matlab with its image processing toolbox is optional. It will only be needed if missing-wedge correction using Monte Carlo or volume denoising using BM4D are to be used
We assume that Matlab is installed on your system in "~/programs/Matlab".
If Matlab is installed but does not work, you may run the command "scipion3 config" and look for MATLAB_HOME in the config file (the config file is usually at ~/scipion3/config/scipion.conf)

Supported versions
------------------

versions > 3.3.0

Protocols
---------

* **HEMNMA**: Hybrid Electron Microscopy Normal Mode Analysis method to interpret heterogeneity of a set of single particle cryo-EM images in terms of continuous macromolecular conformational transitions, based on normal mode analysis [1-3]
* **StructMap**: Structural Mapping method to interpret heterogeneity of a set of single particle cryo-EM maps in terms of continuous conformational transitions, based on normal mode analysis [4]
* **HEMNMA-3D**: Extension of HEMNMA to continuous conformational variability analysis of macromolecules in cryo-ET subtomograms (in vitro and in situ) [5]
* **TomoFlow**: Method for analyzing continuous conformational variability of macromolecules in cryo-ET subtomograms (in vitro and in situ) based on 3D dense optical flow [6]
* **NMMD**: Software to perform cryo-EM flexible fitting using a combination of Normal Mode (NM) analysis and Molecular Dynamics (MD) simulations  implemented in GENESIS [7]
* **DeepHEMNMA**: A deep learning extension of HEMNMA  [8]
* **MDSPACE**: Approach for extracting atomic-resolution landscapes of continuous conformational variability of biomolecular complexes from cryo electron microscopy (cryo-EM) single particle images based on a new 3D-to-2D flexible fitting method, which uses molecular dynamics (MD) simulation and is embedded in an iterative conformational-landscape refinement scheme. [11]

Notes:

* The plugin additionally provides the test data and automated tests of the protocols in Scipion 3. The following two types of tests of HEMNMA and HEMNMA-3D can be produced by running, in the terminal, "scipion3 tests continuousflex.tests.test_workflow_HEMNMA" and “scipion3 tests continuousflex.tests.test_workflow_HEMNMA3D”, respectively: (1) tests of the entire protocol with the flexible references coming from an atomic structure and from an EM map; and (2) test of the alignment module (test run using 5 MPI threads). The automated tests of the TomoFlow method are also available and can be run using scipion3 tests continuousflex.tests.test_workflow_TomoFlow. 
* The automated tests of GENESIS provide an example of cryo-EM flexible fitting of an atomic model into a 3D density map using NMMD for CHARMM and C-Alpha Go model. The tests can be produced by running "scipion3 tests continuousflex.tests.test_workflow_GENESIS" (you need at least 2 MPI cores for these tests).
* HEMNMA additionally provides tools for synthesizing noisy and CTF-affected single particle cryo-EM images with flexible or rigid biomolecular conformations, for several types of conformational distributions, from a given atomic structure or an EM map. One part of the noise is applied on the ideal projections before and the other after the CTF, as described in [9-10].
* HEMNMA-3D additionally provides tools for synthesizing noisy, CTF and missing wedge affected cryo-ET tomograms and single particle subtomograms with flexible or rigid biomolecular conformations, for several types of conformational distributions, from a given atomic structure or an EM map. One part of the noise is applied on the ideal projections before and the other after the CTF, as described in [9-10].
* A reproduction of some utility codes with their corresponding licenses are contained in this plugin for subtomogram averaging, missing wedge correction, denoising and data reading. These codes are not used in the methods above, but they are made optional for data preprocessing and visualization.
* DeepHEMNMA automated test generates a small set of images; then, it runs HEMNMA to prepare data for the neural network training; finally, it trains the network and performs the inference. The test can be run using "scipion3 tests continuousflex.tests.test_workflow_Deep_HEMNMA.TestDeepHEMNMA1". 


References
----------

[1] Jin Q, Sorzano CO, de la Rosa-Trevin JM, Bilbao-Castro JR, Nunez-Ramirez R, Llorca O, Tama F, Jonic S: Iterative elastic 3D-to-2D alignment method using normal modes for studying structural dynamics of large macromolecular complexes. Structure 2014, 22:496-506. `[Open-access] <http://www-ext.impmc.upmc.fr/~jonic/Papers/HEMNMA.pdf>`__

[2] Jonic S: Computational methods for analyzing conformational variability of macromolecular complexes from cryo-electron microscopy images. Curr Opin Struct Biol 2017, 43:114-121. `[Link] <http://dx.doi.org/10.1016/j.sbi.2016.12.011>`__ `[Author’s version] <http://www-ext.impmc.upmc.fr/~jonic/Papers/CurrentOpinionStructBiol_Jonic_2017.pdf>`__

[3] Harastani M, Sorzano CO, Jonic S: Hybrid Electron Microscopy Normal Mode Analysis with Scipion. Protein Sci 2020, 29:223-36. `[Open-access] <https://onlinelibrary.wiley.com/doi/epdf/10.1002/pro.3772>`__

[4] Sanchez Sorzano CO, Alvarez-Cabrera AL, Kazemi M, Carazo JM, Jonic S: StructMap: Elastic Distance Analysis of Electron Microscopy Maps for Studying Conformational Changes. Biophys J 2016, 110:1753-1765. `[Open-access] <http://www-ext.impmc.upmc.fr/~jonic/Papers/StructMap.pdf>`__

[5] Harastani M, Eltsov M, Leforestier A, Jonic S: HEMNMA-3D: Cryo Electron Tomography Method Based on Normal Mode Analysis to Study Continuous Conformational Variability of Macromolecular Complexes. Front Mol Biosci 2021, 8:663121. `[Open-access] <https://www.frontiersin.org/articles/10.3389/fmolb.2021.663121/abstract>`__

[6] Harastani M, Eltsov M, Leforestier A, Jonic S: TomoFlow: Analysis of continuous conformational variability of macromolecules in cryogenic subtomograms based on 3D dense optical flow. J Mol Biol 2021,167381. `[Author’s version] <https://hal.archives-ouvertes.fr/hal-03452809>`__ `[Journal] <https://doi.org/10.1016/j.jmb.2021.167381>`__

[7] Vuillemot R, Miyashita O, Tama F, Rouiller I, Jonic S, NMMD: Efficient Cryo-EM Flexible Fitting Based on Simultaneous Normal Mode and Molecular Dynamics atomic displacements. J Mol Biol 2022, 167483. `[Author’s version] <https://hal.archives-ouvertes.fr/hal-03577246>`__ `[Journal] <https://doi.org/10.1016/j.jmb.2022.167483>`__

[8] Hamitouche I and Jonic S (2022), DeepHEMNMA: ResNet-based hybrid analysis of continuous conformational heterogeneity in cryo-EM single particle images. Front. Mol. Biosci. 9:965645. `[Author’s version] <https://hal.archives-ouvertes.fr/hal-03750789/document>`__ `[Journal] <https://www.frontiersin.org/articles/10.3389/fmolb.2022.965645/full>`__

[9] C.O.S. Sorzano, S. Jonic, R. Núñez-Ramírez, N. Boisset, J.M. Carazo: Fast, robust, and accurate determination of transmission electron microscopy contrast transfer function. Journal of Structural Biology 2007, 160: 249-262. `[Journal] <https://doi.org/10.1016/j.jsb.2007.08.013>`__

[10] Jonic S, Sorzano CO, Thevenaz P, El-Bez C, De Carlo S, Unser M: Spline-based image-to-volume registration for three-dimensional electron microscopy. Ultramicroscopy 2005, 103:303-317. `[Journal] <https://www.sciencedirect.com/science/article/pii/S0304399105000173>`__

[11] Vuillemot R, Mirzaei A, Harastani M, Hamitouche I, Fréchin L, Klaholz BP, Miyashita O, Tama F, Rouiller I, Jonic S. MDSPACE: Extracting continuous conformational landscapes from cryo-EM single particle datasets using 3D-to-2D flexible fitting based on Molecular Dynamics simulation. Journal of Molecular Biology. 2023 Jan 10:167951. `[Journal] <https://www.sciencedirect.com/science/article/abs/pii/S0022283623000074>`__

Citation
----------
Harastani, M., Vuillemot, R., Hamitouche, I., Moghadam, N. B., & Jonic, S. (2022). ContinuousFlex: Software package for analyzing continuous conformational variability of macromolecules in cryo electron microscopy and tomography data. Journal of Structural Biology, 214(4), 107906. `[Journal] <https://doi.org/10.1016/j.jsb.2022.107906>`__

Contact:
----------

All questions regarding the software can be sent through submitting an issue on the Github page or addressed to `[Contact] <continuousflex@gmail.com>`__

# scipion-em-continuousflex
