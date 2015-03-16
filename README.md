nnet
====

High performance Artificial Neural Network library.

Dependencies:
* FFTW3

Building
--------

We use the CMake build system. Here is an example of how you might build the library:

	mkdir build
	cd build
	cmake ..
	make
	sudo make install

If your CPU has access to AVX instructions you should use the `-DENABLE_AVX=ON` flag when running cmake.

Similarly, if your CPU has access to FMA instructions the `-DENABLE_FMA=ON` flag should be used.
