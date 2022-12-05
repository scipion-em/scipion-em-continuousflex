# **************************************************************************
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * Slavica Jonic (slavica.jonic@upmc.fr)
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
# **************************************************************************

from setuptools import setup, find_packages
from codecs import open
from os import path
from continuousflex import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='scipion-em-continuousflex',
    version=__version__,
    description='Plugin to use continuousflex protocols within the Scipion framework',
    long_description=long_description,
    url='https://github.com/scipion-em/scipion-em-continuousflex',
    author='Mohamad Harastani, Remi Vuillemot, Ilyes Hamitouche and Slavica Jonic',
    author_email='slavica.jonic@upmc.fr',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'
    ],

    keywords='cryo-em cryo-et image-processing continuous-conformational-variability',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,
    package_data={
       'continuousflex': ['protocols.conf'],
    },

    entry_points={
        'pyworkflow.plugin': 'continuousflex = continuousflex'
    },

    project_urls={
        'Bug Reports': 'https://github.com/scipion-em/scipion-em-continuousflex/issues',
        'Source': 'https://github.com/scipion-em/scipion-em-continuousflex/',
    },
)
