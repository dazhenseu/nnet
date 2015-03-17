#ifndef _CIFAR10_HPP_
#define _CIFAR10_HPP_

#include <vector>

#include "types.hpp"

size_t loadCifar10(const std::vector<const char *> &filenames, std::vector<nnet_float> &features, std::vector<nnet_float> &labels);

#endif
