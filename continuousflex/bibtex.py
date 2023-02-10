# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Mohamad Harastani (mohamad.harastani@igbmc.fr)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

"""

@article{Jin2014,
title = "Iterative Elastic 3D-to-2D Alignment Method Using Normal Modes for Studying Structural Dynamics of Large Macromolecular Complexes",
journal = "Structure",
volume = "22",
pages = "1 - 11",
year = "2014",
doi = "http://dx.doi.org/10.1016/j.str.2014.01.004",
url = "http://www.ncbi.nlm.nih.gov/pubmed/24508340",
author = "Jin, Q. and Sorzano, C. O. S. and de la Rosa-Trevín, J. M. and Bilbao-Castro, J. R. and Núñez-Ramirez, R. and Llorca, O. and Tama, F. and Jonic, S.",
keywords = "Normal mode analysis, NMA "
}

@article{Jonic2005,
title = "Spline-based image-to-volume registration for three-dimensional electron microscopy ",
journal = "Ultramicroscopy ",
volume = "103",
number = "4",
pages = "303 - 317",
year = "2005",
issn = "0304-3991",
doi = "http://dx.doi.org/10.1016/j.ultramic.2005.02.002",
url = "http://www.sciencedirect.com/science/article/pii/S0304399105000173",
author = "Jonic, S. and C.O.S. Sorzano and P. Thevenaz and C. El-Bez and S. De Carlo and M. Unser",
keywords = "2D/3D registration, Splines, 3DEM, Angular assignment "
}

@article{Nogales2013,
title={3DEM Loupe: analysis of macromolecular dynamics using structures from electron microscopy},
author={Nogales-Cadenas, R. and Jonic, S. and Tama, F. and Arteni, A. A. and Tabas-Madrid, D. and V{\'a}zquez, M. and Pascual-Montano, A. and Sorzano, C. O. S.},
journal={Nucleic acids research},
year={2013},
publisher={Oxford Univ Press},
doi={http://dx.doi.org/10.1093/nar/gkt385}
}


@article{Sorzano2004b,
    volume = "146", 
    doi = "http://dx.doi.org/10.1016/j.jsb.2004.01.006", 
    author = "Sorzano, C.O.S. and S. Jonic and C. El-Bez and J.M. Carazo and S. De Carlo and P. Thevenaz and M. Unser", 
    title = "A multiresolution approach to orientation assignment in 3D electron microscopy of single particles ", 
    journal = "JSB ", 
    issn = "1047-8477", 
    number = "3", 
    note = "", 
    link = "http://www.sciencedirect.com/science/article/pii/S1047847704000073", 
    year = "2004", 
    pages = "381 - 392"
}

@article{Sorzano2016,
title = "StructMap: Elastic distance analysis of electron microscopy maps for studying conformational changes",
journal = "Biophysical J.",
volume = "110",
number = "",
pages = "1753-1765",
year = "2016",
note = "",
issn = "",
doi = "http://doi.org/10.1016/j.bpj.2016.03.019",
url = "http://doi.org/10.1016/j.bpj.2016.03.019",
author = "C.O.S. Sorzano, A.L. Álvarez-Cabrera, M. Kazemi, J.M. Carazo, S. Jonic",
keywords = ""
}


@article{harastani2020hybrid,
  title={Hybrid Electron Microscopy Normal Mode Analysis with Scipion},
  author={Harastani, Mohamad and Sorzano, Carlos Oscar S and Joni{\'c}, Slavica},
  journal={Protein Science},
  volume={29},
  number={1},
  pages={223--236},
  year={2020},
  publisher={Wiley Online Library},
  doi= {https://doi.org/10.1002/pro.3772}
}

@article{moebel2020monte,
  title={A Monte Carlo framework for missing wedge restoration and noise removal in cryo-electron tomography},
  author={Moebel, Emmanuel and Kervrann, Charles},
  journal={Journal of Structural Biology: X},
  volume={4},
  pages={100013},
  year={2020},
  publisher={Elsevier}
}

@article{vuillemot2022NMMD,
title = {NMMD: Efficient Cryo-EM Flexible Fitting Based on Simultaneous Normal Mode and Molecular Dynamics atomic displacements},
journal = {Journal of Molecular Biology},
volume = {434},
number = {7},
pages = {167483},
year = {2022},
issn = {0022-2836},
doi = {https://doi.org/10.1016/j.jmb.2022.167483},
url = {https://www.sciencedirect.com/science/article/pii/S0022283622000523},
author = {Rémi Vuillemot and Osamu Miyashita and Florence Tama and Isabelle Rouiller and Slavica Jonic}
}

@article{vuillemot2023mdspace,
  title={MDSPACE: Extracting continuous conformational landscapes from cryo-EM single particle datasets using 3D-to-2D flexible fitting based on Molecular Dynamics simulation},
  author={Vuillemot, R{\'e}mi and Mirzaei, Alex and Harastani, Mohamad and Hamitouche, Ilyes and Fr{\'e}chin, L{\'e}o and Klaholz, Bruno P and Miyashita, Osamu and Tama, Florence and Rouiller, Isabelle and Jonic, Slavica},
  journal={Journal of Molecular Biology},
  pages={167951},
  year={2023},
  publisher={Elsevier}
}

@article{kobayashi2017genesis,
author = {Kobayashi, Chigusa and Jung, Jaewoon and Matsunaga, Yasuhiro and Mori, Takaharu and Ando, Tadashi and Tamura, Koichi and Kamiya, Motoshi and Sugita, Yuji},
title = {GENESIS 1.1: A hybrid-parallel molecular dynamics simulator with enhanced sampling algorithms on multiple computational platforms},
journal = {Journal of Computational Chemistry},
volume = {38},
number = {25},
pages = {2193-2206},
keywords = {molecular dynamics, string method, replica exchange molecular dynamics, graphics processing unit, multiple time step integration},
doi = {https://doi.org/10.1002/jcc.24874},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.24874},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.24874},
abstract = {GENeralized-Ensemble SImulation System (GENESIS) is a software package for molecular dynamics (MD) simulation of biological systems. It is designed to extend limitations in system size and accessible time scale by adopting highly parallelized schemes and enhanced conformational sampling algorithms. In this new version, GENESIS 1.1, new functions and advanced algorithms have been added. The all-atom and coarse-grained potential energy functions used in AMBER and GROMACS packages now become available in addition to CHARMM energy functions. The performance of MD simulations has been greatly improved by further optimization, multiple time-step integration, and hybrid (CPU + GPU) computing. The string method and replica-exchange umbrella sampling with flexible collective variable choice are used for finding the minimum free-energy pathway and obtaining free-energy profiles for conformational changes of a macromolecule. These new features increase the usefulness and power of GENESIS for modeling and simulation in biological research. © 2017 Wiley Periodicals, Inc.},
year = {2017}
}

@article{CHEN2013235,
title = {Fast and accurate reference-free alignment of subtomograms},
journal = {Journal of Structural Biology},
volume = {182},
number = {3},
pages = {235-245},
year = {2013},
issn = {1047-8477},
doi = {https://doi.org/10.1016/j.jsb.2013.03.002},
url = {https://www.sciencedirect.com/science/article/pii/S1047847713000737},
author = {Yuxiang Chen and Stefan Pfeffer and Thomas Hrabe and Jan Michael Schuller and Friedrich Förster},
keywords = {Cryo-electron tomography, Subtomogram averaging, Spherical harmonics},
abstract = {In cryoelectron tomography alignment and averaging of subtomograms, each dnepicting the same macromolecule, improves the resolution compared to the individual subtomogram. Major challenges of subtomogram alignment are noise enhancement due to overfitting, the bias of an initial reference in the iterative alignment process, and the computational cost of processing increasingly large amounts of data. Here, we propose an efficient and accurate alignment algorithm via a generalized convolution theorem, which allows computation of a constrained correlation function using spherical harmonics. This formulation increases computational speed of rotational matching dramatically compared to rotation search in Cartesian space without sacrificing accuracy in contrast to other spherical harmonic based approaches. Using this sampling method, a reference-free alignment procedure is proposed to tackle reference bias and overfitting, which also includes contrast transfer function correction by Wiener filtering. Application of the method to simulated data allowed us to obtain resolutions near the ground truth. For two experimental datasets, ribosomes from yeast lysate and purified 20S proteasomes, we achieved reconstructions of approximately 20Å and 16Å, respectively. The software is ready-to-use and made public to the community.}
}

@article{harastani2022continuousflex,
  title={ContinuousFlex: Software package for analyzing continuous conformational variability of macromolecules in cryo electron microscopy and tomography data},
  author={Harastani, Mohamad and Vuillemot, R{\'e}mi and Hamitouche, Ilyes and Moghadam, Nima Barati and Jonic, Slavica},
  journal={Journal of Structural Biology},
  volume={214},
  number={4},
  pages={107906},
  year={2022},
  publisher={Elsevier}
}

@article{harastani2022tomoflow,
  title={TomoFlow: Analysis of continuous conformational variability of macromolecules in cryogenic subtomograms based on 3D dense optical flow},
  author={Harastani, Mohamad and Eltsov, Mikhail and Leforestier, Am{\'e}lie and Jonic, Slavica},
  journal={Journal of molecular biology},
  volume={434},
  number={2},
  pages={167381},
  year={2022},
  publisher={Elsevier}
}

@article{harastani2021hemnma,
  title={Hemnma-3d: Cryo electron tomography method based on normal mode analysis to study continuous conformational variability of macromolecular complexes},
  author={Harastani, Mohamad and Eltsov, Mikhail and Leforestier, Am{\'e}lie and Jonic, Slavica},
  journal={Frontiers in molecular biosciences},
  pages={317},
  year={2021},
  publisher={Frontiers}
}

@article{hamitouche2022deephemnma,
  title={DeepHEMNMA: ResNet-based hybrid analysis of continuous conformational heterogeneity in cryo-EM single particle images},
  author={Hamitouche, Ilyes and Jonic, Slavica},
  journal={Frontiers in Molecular Biosciences},
  year={2022}
}

"""

