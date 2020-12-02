# RDFT - restricted domain Fourier Transform for Python

# A typical situation when RDFT is usefull is when the 2D Discrete Fourier 
# Transform is evaluated many times at a limited number of fixed 
# frequencies on large sparse matrices (with same dimensions but with
# changing positions and number of non-zero elements).

# This program shows how to use RDFT
#
# RDFT is described in the Appendix of:
# M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects," Opt. Express 28(24), 36206-36218 (2020)
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
###########################################################

import numpy as np 
import scipy.sparse as sp


def FTax(N):
    if N%2:
        x=np.concatenate((np.arange((N-1/2)+1),np.arange(-(N-1)/2,0)))  # [0:(N-1)/2,-(N-1)/2:-1] 
    else:
         x=np.concatenate((np.arange(N/2),np.arange(-N/2,0))) # 0:N/2-1,-N/2:-1]
    return x

class RDFTprep:    
    def __init__(self,MaskFT):
        """
        rdft - restricted domain Fourier transform
        """        
         
        self.MaskFT=MaskFT['fk']        
        self.Ny,self.Nx=MaskFT['shape']
        self.px=FTax(self.Nx)
        self.py=FTax(self.Ny)
        self.Px,self.Py=np.meshgrid(self.px,self.py)

        self.NN=np.lcm(self.Nx,self.Ny)
        self.selfx=self.NN/self.Nx
        self.selfy=self.NN/self.Ny
        self.mx=self.Px[self.MaskFT]*self.selfx
        self.my=self.Py[self.MaskFT]*self.selfy
        self.PHI=np.exp(-2j*np.pi/self.NN*np.arange(self.NN))        
        #self.M=lambda SpVals: self.PHI[(self.mx*self.Px[np.where(SpVals)].T+self.my*self.Py[np.where(SpVals)].T)%self.NN]
        self.sdft2 = lambda SpVals : self.SparseDFT2(SpVals)
        self.pdft2 = lambda SpVals : self.PartDFT2(SpVals)

    def M(self, SpVals):
        ir,ic=SpVals
        return self.PHI[np.mod((np.kron(self.mx[:,np.newaxis],self.Px[ir,ic].T)+np.kron(self.my[:,np.newaxis],self.Py[ir,ic].T)),self.NN).astype('int32')]        
        
    def SparseDFT2(self,SpVals):
        M=self.M(SpVals)
        FT=sp.csc_matrix((M.shape[0],self.Nx*self.Ny),dtype=complex)
        row,col = SpVals
        FT[:,row*self.Nx+col]=M
        return lambda X: FT*X.reshape(X.shape[1]*X.shape[0],1)
         
    def PartDFT2(self,SpVals):
        M=self.M(SpVals)
        DFT2=lambda X: M*X.reshape(X.shape[1]*X.shape[0],1)
        return DFT2   
    
    def rdft2(self,SpVals):
        ir,ic,nonzeros=sp.find(SpVals)
        return self.PHI[np.mod((np.kron(self.mx[:,np.newaxis],self.Px[ir,ic].T)+np.kron(self.my[:,np.newaxis],self.Py[ir,ic].T)),self.NN).astype('int32')]@nonzeros
