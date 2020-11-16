# RDFT  restricted domain Fourier Transform for Matlab/Octave
(a Python version will be also includesd soon)

A typical situation when RDFT is usefull is when the 2D Discrete Fourier 
Transform is evaluated many times at a limited number of fixed 
frequencies on large sparse matrices (with same dimensions but with
changing positions and number of non-zero elements).

The main function is included in the RDFTPrep.m file, and an example showing its use is in RDFT_exampl.m


If you find this program useful, please cite the work:
Maria Bancerek, Krzysztof M. Czajkowski, and Rafał Kotyński, "Far-field signature of sub-wavelength microscopic objects," Opt. Express 28, 36206-36218 (2020), https://doi.org/10.1364/OE.410240
(RDFT is described in the Appendix)

Contact info: rafalk@fuw.edu.pl

----------------------------------------------------------------------------------------------------------------------------
Below you may find a typical speedup possible to obtain with respect to fft2() function (which uses non-sparse FFTW package)
The results have been obtained by running RDFT_example.m on a Debian system with AMD Ryzen 2700X CPU
-------------------------------------------------------------------------------------------
Matlab '9.6.0.1150989 (R2019a) Update 4':

Restricted domain Fourier transform (RDFT) example:
In this example, a large 2D array is generated with a small number of non-zero elements
 and its 2D discrete Fourier Transform is evaluated in 4 ways.The accuracies and calculation times
are compared.

Matrix size:[8192 x 8192]

Number of nonzero matrix elements:50

Number of frequencies:300

Mask size for the maximal number of nonzero matrix elements (for pdft2() and sdft2() only):950

Comparison of calculation methods:

Direct,  	  fft2():	 time=627660 us

Variant 1: 	 rdft2():	 time=293 us, speedup factor=2.142184e+03, error=5e-14

Variant 2: 	 pdft2():	 time=133 us, speedup factor=4.719248e+03, error=5e-14

Variant 3: 	 sdft2():	 time=114 us, speedup factor=5.505789e+03, error=5e-14

-------------------------------------------------------------------------------------------
Octave 4.4.1 with OpenBLAS 0.3.5:

Restricted domain Fourier transform (RDFT) example:
In this example, a large 2D array is generated with a small number of non-zero elements
 and its 2D discrete Fourier Transform is evaluated in 4 ways.The accuracies and calculation times
are compared.

Matrix size:[8192 x 8192]

Number of nonzero matrix elements:50

Number of frequencies:300

Mask size for the maximal number of nonzero matrix elements (for pdft2() and sdft2() only):950

Comparison of calculation methods:

Direct,           fft2():        time=424437 us

Variant 1:       rfft2():        time=629.902 us, speedup factor=673.815, error=4e-14

Variant 2:       pfft2():        time=123.978 us, speedup factor=3423.5, error=4.3e-14

Variant 3:       sfft2():        time=129.223 us, speedup factor=3284.54, error=4.3e-14


