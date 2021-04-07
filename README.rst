=====================
ContinuousFlex plugin
=====================

This plugin provides the latest Scipion protocols for cryo-EM continuous conformational flexibility/heterogeneity analysis of biomolecular complexes.


Installation
------------

You will need to use `3.0 <https://github.com/I2PC/scipion/releases>`_ version of Scipion to be able to run these protocols. To install the plugin, you have two options:

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


Note: Xmipp and Chimerax plugins should be installed (from Scipion3 plugin manager) to run continuousflex protocols. You should also consider having VMD on your system for visualization.

Supported versions
------------------

3.0.1

Protocols
---------

* HEMNMA: Hybrid Electron-Microscopy Normal-Mode-Analysis
* StructMap: Structural Mapping

References
----------
1. Harastani M, Sorzano CO, Jonić S. Hybrid Electron Microscopy Normal Mode Analysis with Scipion. Protein Science. 2020 Jan;29(1):223-36.
2. Jin Q, Sorzano CO, de la Rosa-Trevin JM, Bilbao-Castro JR, Nunez-Ramirez R, Llorca O, Tama F,Jonic S: Iterative elastic 3D-to-2D alignment method using normal modes for studying structural dynamics of large macromolecular complexes. Structure 2014, 22:496-506.
3. Sorzano CO, de la Rosa-Trevin JM, Tama F, Jonic S: Hybrid Electron Microscopy Normal Mode Analysis graphical interface and protocol. J Struct Biol 2014, 188:134-141.
4. Sanchez Sorzano CO, Alvarez-Cabrera AL, Kazemi M, Carazo JM, Jonic S: StructMap: Elastic Distance Analysis of Electron Microscopy Maps for Studying Conformational Changes. Biophys J 2016, 110:1753-1765.



# scipion-em-continuousflex
