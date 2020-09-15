% Restricted Domain Fourier Transform
% This is a Matlab/Octave example program which calculates a 2D Discrete Fourier Transform
% of sparse matrices at fixed frequencies
%
% If you find this program useful, please cite the work:
% M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects,"
% https://arxiv.org/abs/2009.06324 (RDFT is described in the Appendix)
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


% A typical situation when RDFT is usefull is when the 2D discrete Fourier 
% Transform has to be evaluated many times at a limited number of fixed 
% frequencies on large sparse matrices (with same dimensions but with
% changing positions and number of non-zero elements).
% This program benchmarks RDFT against Matlab fft2() function
% 
% If you find this program useful, please cite the work: M. Bancerek, K. Czajkowski, R. Kotynski "Far-field intensity signature of sub-wavelength microscopic objects," Arxiv, 2020
% The RDFT method is described in the Appendix
%
% Copyright (C) 2020 R. Kotyński, rafalk@fuw.edu.pl
% Funding acknowledgement: National Science Center, Poland, UMO:2017/27/B/ST7/00885

function RDFT_benchmark()
Nobj=50; % number of non-zero elements in the image domain
Nft=300; % number of FT coefficients to be calculated
Nrep=5; % number of repetitions


Px=round(2.^(10:.5:14));% matrix sizes
Px(2:2:end)=Px(2:2:end)+1;% to test also odd sizes

Py=Px;
tfft=zeros(numel(Px),Nrep);
trdft=zeros(numel(Px),Nrep);
Err=zeros(numel(Px),Nrep);
for i=1:numel(Px)
    % optimize the fftw algorithm
    fftw('planner','measure');fft2(randn(Py(i),Px(i)));
    for j=1:Nrep        
        % Select the frequencies that will be needed and store their positions
        % in a spase matrix MaskFT
        MaskFT=logical(sparse(Py(i),Px(i)));
        fk=randi(Px(i)*Py(i),Nft,1);
        MaskFT(fk)=true;
        
        % prepare for fft evaluation
        rdft2=RDFTprep(MaskFT);


        
        % Create Py(i) x Px(i) matrix with Nobj non-zero elements
        xi=randi(Px(i)*Py(i),Nobj,1);
        Xsparse=sparse(Py(i),Px(i));
        Xsparse(xi)=randn(Nobj,1);  % Sparse form
        Xfull=full(Xsparse);        % full form
        
        
        % benchmark the sparse FT
        tic;
        Ysparse=rdft2(Xsparse);
        trdft(i,j)=toc;
        
        
        % benchmark dense FFTW
        tic;
        Y=fft2(Xfull);
        tfft(i,j)=toc;
        
        % check the difference
        Err(i,j)=norm(Ysparse-Y(MaskFT) )/ norm(Y(MaskFT));
        
    end
end

% Plot a comparison
subplot(3,1,1)
loglog(sqrt(Px.*Py),1e3*mean(tfft,2),'*r')
hold on
loglog(sqrt(Px.*Py),1e3*mean(trdft,2),'*g')
hold off
legend('FFTW (full matrices)','RDFT (sparse matrices)')
ylabel('Time (ms)')
grid on
axis tight
%xlabel('Matrix size sqrt(Nx\cdot Ny)')
title(sprintf('Restricted Domain FT benchmark\n (%d Non-zero elements, %d Fourier coefficients)',Nobj,Nft))

subplot(3,1,2)
loglog(sqrt(Px.*Py),mean(tfft,2)./mean(trdft,2),'*k')
ylabel('Time ratio')
axis tight
%xlabel('Matrix size sqrt(Nx\cdot Ny)')
grid on

subplot(3,1,3)
loglog(sqrt(Px.*Py),mean(Err,2),'ok')
ylabel('Mean error (%)')
xlabel('Matrix size sqrt(Nx\cdot Ny)')
grid on
axis tight

end