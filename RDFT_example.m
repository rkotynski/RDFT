% RDFT - restricted domain Fourier Transform for Matlab/Octave

% A typical situation when RDFT is usefull is when the 2D Discrete Fourier 
% Transform is evaluated many times at a limited number of fixed 
% frequencies on large sparse matrices (with same dimensions but with
% changing positions and number of non-zero elements).

% This program shows how to use RDFT
%
% If you find this program useful, please cite the work: M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects," Arxiv, 2020
% (RDFT is described in the Appendix)
% Copyright (C) 2020 R. Kotyński, rafalk@fuw.edu.pl
% Funding acknowledgement: National Science Center, Poland, UMO:2017/27/B/ST7/00885
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

function RDFT_example(Px,Py,Nobj,Nft,Nmask)
  
  if nargin<5, Nmask=1000; end % area size where nonzero elements may be located
  if nargin<4, Nft=300;end % number of frequencies to calculate
  if nargin<3, Nobj=50;end % nonzero matrix elements
  if nargin<2,Py=8192;end % full matrix size [Py,Px]
  if nargin<1,Px=8192;end   

fprintf('\nRestricted domain Fourier transform (RDFT) example:\n')
fprintf('In this example, a large 2D array is generated with a small number of non-zero elements\n and its 2D discrete Fourier Transform is evaluated in 4 ways.The accuracies and calculation times\nare compared.\n\n')

% Select some frequencies to be evaluated
MaskFT=logical(sparse(Py,Px));
fk=randi(Px*Py,Nft,1);
MaskFT(fk)=true;

% Create a large sparse matrix with Nobj elements
M=sparse(Py,Px);
M(randi(Nobj,Py*Px,1))=randn(Py*Px,1);

% select the filtering matrix that marks potentially nonzero elements of the 2d signal (for use with sdft2() and pdft2())
MaskImage=logical(M); 
if nnz(MaskImage)<Nmask
  MaskImage(randi(Nmask-nnz(MaskImage),Py*Px,1))=true; 
end

fprintf('Matrix size:[%d x %d]\nNumber of nonzero matrix elements:%d\nNumber of frequencies:%d\n',Py,Px,nnz(M),nnz(MaskFT));
fprintf('Mask size for the maximal number of nonzero matrix elements (for pdft2() and sdft2() only):%d\n',nnz(MaskImage));

fprintf('\nComparison of calculation methods:\n')
% CALCULATE THE DFT2 OF MATRIX M AT FREQUENCIES SELECTED WITH MaskFT

% Direct calculation: standard fft2 is calculated on a full marrix and part of the result is thrown away
Mfull=full(M);
 fftw('planner','measure');fft2(Mfull);
tic
FM_direct=fft2(Mfull);
t0=toc;
FM_direct=FM_direct(MaskFT);
fprintf('Direct,  \t  fft2():\t time=%d us\n',t0*1e6)

% Variant 1: the DFT matrix is dynamically created,   which is memory efficient
%  this is the most flexible and memory efficient use of rdft but may be slower than Variants 2 and 3
[rdft2,RDFT]=RDFTprep(MaskFT);% Precalculation stage (required for the all variants)

tic
FM_variant1=rdft2(M);
t1=toc;
fprintf('Variant 1: \t rdft2():\t time=%d us, speedup factor=%d, error=%.2g\n',t1*1e6,t0/t1,norm(FM_variant1(:)-FM_direct(:)))

% Variant 2: part of the dense DFT2 matrix is precalculated,   which may take more memory
% a function  pdft2() that calculates dft2 on a vector taken from the original matrix is first created 
% DFT2 is then evaluated as a matrix-vector product (but the DFT2 matrix must be recalculated whenever we modify the location of non-zero elements in the transformed array)

pdft2=RDFT.pdft2(MaskImage);  % precalculate part of the  dft2 matrix with rows and columns selected by MaskImage and MaskFT
tic
FM_variant2=pdft2(M(MaskImage)); % calculate dft2 as a matrix-vector product
t2=toc;
fprintf('Variant 2: \t pdft2():\t time=%d us, speedup factor=%d, error=%.2g\n',t2*1e6,t0/t2,norm(FM_variant2(:)-FM_direct(:)))

% Variant 3: part of the dense DFT2 matrix is precalculated,   which may take more memory
% but this time it has a form of a sparse matrix and is used to create
% a function  sdft2() that calculates dft2 on sparse matrices 
% Variant 3 may be faster than Variant 2 when the non-zero elements of M make a small part of MaskImage 
sdft2=RDFT.sdft2(MaskImage); % precalculate part of the  dft2 matrix as a sparse matrix  with rows and columns selected by MaskImage and MaskFT
tic
FM_variant3=sdft2(M); % calculate dft2 as a sparse matrix-vector product
t3=toc;
assert(all(M(:).*MaskImage(:)==M(:))); % non-zero elements of M can not fall out of the mask
fprintf('Variant 3: \t sdft2():\t time=%d us, speedup factor=%d, error=%.2g\n',t3*1e6,t0/t3,norm(FM_variant3(:)-FM_direct(:)))
end