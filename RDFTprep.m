% Restricted Domain Fourier Transform
% This is a Matlab/Octave program which calculates a 2D Discrete Fourier Transform
% of sparse matrices at fixed frequencies
%
% If you find this program useful, please cite the work:
% M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects,"
% https://arxiv.org/abs/2009.06324 (RDFT is described in the Appendix)
%
%
%
% Copyright (C) 2020 R. Kotyński, rafalk@fuw.edu.pl
% Funding acknowledgement: National Science Center, Poland, UMO:2017/27/B/ST7/00885
%
%
%
%  GPL LICENSE INFORMATION
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


function [rdft2,RDFT]=RDFTprep(MaskFT)
% 2D DFT evaluation on sparse matrices at a limited number of fixed frequencies
% MaskFT - sparse logical array pointing to the frequencies that will be calculated
%
% RDFT.rdft2(SpVals) calculates the 2D fft on a sparse array SpVals and
% returns a vector of DFT coefficients
%
% RDFT.M(SpVals) returnes the dense 2D FFT matrix that operates on the non-zero
% elements of SpVals
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    function [x]=FTax(N)
        if mod(N,2)
            x=int32([0:(N-1)/2,-(N-1)/2:-1]);
        else
            x=int32([0:N/2-1,-N/2:-1]);
        end
    end

    function [DFT2]=SparseDFT2(RDFT,SpVals)
      M=RDFT.M(SpVals);
      %F=spalloc(RDFT.Nx*RDFT.Ny,RDFT.Nx*RDFT.Ny,numel(M));
      %F(find(RDFT.MaskFT),find(SpVals)) =M;
      FT=spalloc(nnz(RDFT.MaskFT),RDFT.Nx*RDFT.Ny,numel(M));
      FT(:,find(SpVals)) =M;
      DFT2=@(X)FT*X(:);
     end
     
    function [DFT2]=PartDFT2(RDFT,SpVals)
      M=RDFT.M(SpVals);
      DFT2=@(X)M*X(:);
     end
     
RDFT=[];
RDFT.MaskFT= MaskFT;
[RDFT.Ny,RDFT.Nx]=size(MaskFT);
RDFT.px=FTax(RDFT.Nx);
RDFT.py=FTax(RDFT.Ny);
[RDFT.Px,RDFT.Py]=meshgrid(RDFT.px,RDFT.py);

RDFT.NN=lcm(RDFT.Nx,RDFT.Ny);
RDFT.RDFTx=int32(RDFT.NN/RDFT.Nx);
RDFT.RDFTy=int32(RDFT.NN/RDFT.Ny);
RDFT.mx=int32(RDFT.Px(MaskFT).*RDFT.RDFTx);
RDFT.my=int32(RDFT.Py(MaskFT).*RDFT.RDFTy);
RDFT.PHI=exp(-2i*pi/RDFT.NN.*(0:RDFT.NN-1));
RDFT.rdft2=@(SpVals)RDFT.PHI((1+mod((RDFT.mx.*RDFT.Px(find(SpVals))'+RDFT.my.*RDFT.Py(find(SpVals))'),RDFT.NN) ))*nonzeros(SpVals);
%RDFT.M=@(SpVals)RDFT.PHI(uint32(1+mod(uint32(RDFT.mx.*RDFT.Px(find(SpVals))'+RDFT.my.*RDFT.Py(find(SpVals))'),RDFT.NN) ));
RDFT.M=@(SpVals)RDFT.PHI((1+mod((RDFT.mx.*RDFT.Px(find(SpVals))'+RDFT.my.*RDFT.Py(find(SpVals))'),RDFT.NN) ));
RDFT.sdft2=@(SpVals)SparseDFT2(RDFT,SpVals); % shielded sparse dft2
RDFT.pdft2=@(SpVals)PartDFT2(RDFT,SpVals); % dft2 on a fragment of the image using a dense dft matrix
rdft2=RDFT.rdft2;
end
