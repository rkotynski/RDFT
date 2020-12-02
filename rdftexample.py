# RDFT - restricted domain Fourier Transform for Python

# A typical situation when RDFT is usefull is when the 2D Discrete Fourier 
# Transform is evaluated many times at a limited number of fixed 
# frequencies on large sparse matrices (with same dimensions but with
# changing positions and number of non-zero elements).

# This program shows how to use RDFT
#
# If you find this program useful, please cite the work:
# M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects,"
# Opt. Express 28(24), 36206-36218 (2020) 
# https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-28-24-36206&id=442641 (RDFT is described in the Appix)
#
# Copyright (C) 2020 R. Kotynski, K. Czajkowski, M. Bancerek
# rafalk@fuw.edu.pl, Krzysztof.Czajkowski@fuw.edu.pl, Maria.Bancerek@fuw.edu.pl
#
###########################################################
#
#  GPL LICENSE INFORMATION
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np 
import scipy.sparse as sp
import time
from rdftprep import RDFTprep

Nmask=1000
Nft=300
Nobj=50
Py=8192
Px=8192

print('\nRestricted domain Fourier transform (RDFT) example:\n')
print('In this example, a large 2D array is generated with a small number of non-zero elements\n and its 2D discrete Fourier Transform is evaluated in 4 ways.The accuracies and calculation times\nare compared.\n\n')

# Select some frequencies to be evaluated
MaskFT=dict()
fkx=np.random.randint(0,Px-1,size=Nft)
fky=np.random.randint(0,Py-1,size=Nft)
MaskFT['fk']=(fky,fkx)
MaskFT['shape']=(Py,Px)

# Create a large sparse matrix with Nobj elements
xix=np.random.randint(0,Px-1,size=Nobj)
xiy=np.random.randint(0,Py-1,size=Nobj)
M=sp.csc_matrix((np.random.randn(Nobj),(xix,xiy)),shape=(Py,Px),dtype=complex)

# select the filtering matrix that marks potentially nonzero elements of the 2d signal (for use with sdft2() and pdft2())
MaskImage=M.nonzero() #MaskImage=logical(M) 
#if len(MaskImage[0])<Nmask:
#  MaskImage(randi(Nmask-nnz(MaskImage),Py*Px,1))=true 


#print(f'Matrix size:[{Py} x {Px}]\nNumber of nonzero matrix elements:{len(MaskImage[0])}\nNumber of frequencies:{}\n')
#print(f'Mask size for the maximal number of nonzero matrix elements (for pdft2() and sdft2() only):#d\n',nnz(MaskImage))

print('\nComparison of calculation methods:\n')
# CALCULATE THE DFT2 OF MATRIX M AT FREQUENCIES SELECTED WITH MaskFT

# Direct calculation: standard fft2 is calculated on a full marrix and part of the result is thrown away
Mfull=M.todense()        # full form
t_start = time.time()
FM_direct=np.fft.fft2(Mfull)
t0=time.time() - t_start
FM_direct=FM_direct[MaskFT['fk']]
print(f'Direct,  \t  fft2():\t time={t0*1e6} us\n')

# Variant 1: the DFT matrix is dynamically created,   which is memory efficient
#  this is the most flexible and memory efficient use of rdft but may be slower than Variants 2 and 3
RDFT=RDFTprep(MaskFT)# Precalculation stage (required for the all variants)

t_start = time.time()
FM_variant1=RDFT.rdft2(M)
t1=time.time() - t_start
print(f'Variant 1: \t rdft2():\t time={t1*1e6} us, speedup factor={t0/t1}, error={np.linalg.norm(FM_variant1.flatten()-FM_direct.flatten())}\n')
# Variant 2: part of the dense DFT2 matrix is precalculated,   which may take more memory
# a function  pdft2() that calculates dft2 on a vector taken from the original matrix is first created 
# DFT2 is then evaluated as a matrix-vector product (but the DFT2 matrix must be recalculated whenever we modify the location of non-zero elements in the transformed array)

pdft2=RDFT.pdft2(MaskImage)  # precalculate part of the  dft2 matrix with rows and columns selected by MaskImage and MaskFT
t_start = time.time()
FM_variant2=pdft2(M[MaskImage]) # calculate dft2 as a matrix-vector product
t2=time.time() - t_start
print(f'Variant 2: \t pdft2():\t time={t2*1e6} us, speedup factor={t0/t2}, error={np.linalg.norm(FM_variant2.flatten()-FM_direct.flatten())}\n')

# Variant 3: part of the dense DFT2 matrix is precalculated,   which may take more memory
# but this time it has a form of a sparse matrix and is used to create
# a function  sdft2() that calculates dft2 on sparse matrices 
# Variant 3 may be faster than Variant 2 when the non-zero elements of M make a small part of MaskImage 
sdft2=RDFT.sdft2(MaskImage) # precalculate part of the  dft2 matrix as a sparse matrix  with rows and columns selected by MaskImage and MaskFT
t_start = time.time()
FM_variant3=sdft2(M) # calculate dft2 as a sparse matrix-vector product
t3=time.time() - t_start
#assert(all(M.flatten()*MaskImage.flatten()==M.flatten())) # non-zero elements of M can not fall out of the mask
print(f'Variant 3: \t sdft2():\t time={t3*1e6} us, speedup factor={t0/t3}, error={np.linalg.norm(FM_variant3.reshape(1,FM_variant3.shape[0]*FM_variant3.shape[1])-FM_direct.flatten())}\n')

