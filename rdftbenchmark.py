# RDFT - restricted domain Fourier Transform for Python

# A typical situation when RDFT is usefull is when the 2D Discrete Fourier 
# Transform is evaluated many times at a limited number of fixed 
# frequencies on large sparse matrices (with same dimensions but with
# changing positions and number of non-zero elements).

# This program shows how to use RDFT
#
# RDFT is described in Appendix of:
# M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects,"
# Opt. Express 28(24), 36206-36218 (2020) 
# https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-28-24-36206&id=442641
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


import time
import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse as sp
from rdftprep import RDFTprep

Nobj=50 # number of non-zero elements in the image domain
Nft=300 # number of FT coefficients to be calculated
Nrep=5 # number of repetitions


Px=np.round(2**np.arange(10,14.5,0.5)).astype('int32')# matrix sizes (10:.5:14)
Px[1:len(Px)-1:2]=Px[1:len(Px)-1:2]+1 # to test also the odd sizes
Px=Px[:4]

Py=Px+0
tfft=np.zeros((len(Px),Nrep))
trdft=np.zeros((len(Px),Nrep))
Err=np.zeros((len(Px),Nrep))
for i in range(len(Px)):
    # optimize the fftw algorithm
    #fftw('planner','measure')
    #fft2(np.random.randn(Py[i],Px[i]))
    for j in range(Nrep): 
        print((i,j,len(Px)))
        # Select the frequencies that will be needed and store their positions
        # in a spase matrix MaskFT
        MaskFT=dict()
        fkx=np.random.randint(0,Px[i]-1,size=Nft)
        fky=np.random.randint(0,Py[i]-1,size=Nft)         
        MaskFT['fk']=(fky,fkx)
        MaskFT['shape']=(Py[i],Px[i])
        #MaskFT=sp.csc_matrix((np.ones(Nft),(fkx,fky)),shape=(Py[i],Px[i])).astype('bool')
        
        # prepare for fft evaluation
        rdft2=RDFTprep(MaskFT)
        
        # Create Py[i] x Px[i] matrix with Nobj non-zero elements
        xix=np.random.randint(0,Px[i]-1,size=Nobj)
        xiy=np.random.randint(0,Py[i]-1,size=Nobj)
        Xsparse=sp.csc_matrix((np.random.randn(Nobj),(xix,xiy)),shape=(Py[i],Px[i]),dtype=complex)
        Xfull=Xsparse.todense()        # full form
        
        
        # benchmark the sparse FT
        t_start=time.time()
        Ysparse=rdft2.rdft2(Xsparse)
        trdft[i,j]=time.time()-t_start
        
        
        # benchmark dense FFTW
        t_start=time.time()
        Y=np.fft.fft2(Xfull)
        tfft[i,j]=time.time()-t_start
        
        # check the difference
        Err[i,j]=np.linalg.norm(Ysparse-Y[MaskFT['fk']] )/ np.linalg.norm(Y[MaskFT['fk']])

# Plot a comparison
plt.figure(figsize=(5,9))
plt.subplot(3,1,1)
plt.loglog(np.sqrt(Px*Py),1e3*np.mean(tfft,axis=1),'*r')
plt.loglog(np.sqrt(Px*Py),1e3*np.mean(trdft,axis=1),'*g')
#plt.legend('FFTW (full matrices)','RDFT (sparse matrices)')
plt.ylabel('Time (ms)')
#plt.xlabel('Matrix size np.sqrt(Nx\cdot Ny)')
#title(sprintf('Restricted Domain FT benchmark\n (#d Non-zero elements, #d Fourier coefficients)',Nobj,Nft))

plt.subplot(3,1,2)
plt.loglog(np.sqrt(Px*Py),np.mean(tfft,axis=1)/np.mean(trdft,axis=1),'*k')
plt.ylabel('Time ratio')
#plt.xlabel('Matrix size np.sqrt(Nx\cdot Ny)')

plt.subplot(3,1,3)
plt.loglog(np.sqrt(Px*Py),np.mean(Err,axis=1),'ok')
plt.ylabel('mean error (%)')
plt.xlabel('Matrix size $\sqrt{(Nx\cdot Ny)}$')

plt.tight_layout()
