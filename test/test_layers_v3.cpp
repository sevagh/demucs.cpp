// use gtest to test the load_audio_for_kissfft function

#include "conv.hpp"
#include "dsp.hpp"
#include "encdec.hpp"
#include "layers.hpp"
#include "model.hpp"
#include "lstm.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>

namespace demucscppdebug_test_v3
{

inline void assert_(bool condition)
{
    if (!condition)
    {
        std::cout << "Assertion failed!" << std::endl;
        std::exit(1);
    }
}

inline void debug_tensor_4dxf(const Eigen::Tensor4dXf &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ", " << x.dimension(2) << ", " << x.dimension(3) << ")"
              << std::endl;

    float x_min = 100000000.0f;
    float x_max = -100000000.0f;
    float x_sum = 0.0f;
    float x_mean = 0.0f;
    float x_stddev = 0.0f;

    // store dimension index to save index of min/max
    int x_min_idx_0 = -1;
    int x_min_idx_1 = -1;
    int x_min_idx_2 = -1;
    int x_min_idx_3 = -1;
    int x_max_idx_0 = -1;
    int x_max_idx_1 = -1;
    int x_max_idx_2 = -1;
    int x_max_idx_3 = -1;

    // loop over tensor and find min/max/sum
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                for (int l = 0; l < x.dimension(3); ++l)
                {
                    float val = x(i, j, k, l);
                    x_sum += val;
                    if (val < x_min)
                    {
                        x_min = val;
                        x_min_idx_0 = i;
                        x_min_idx_1 = j;
                        x_min_idx_2 = k;
                        x_min_idx_3 = l;
                    }
                    if (val > x_max)
                    {
                        x_max = val;
                        x_max_idx_0 = i;
                        x_max_idx_1 = j;
                        x_max_idx_2 = k;
                        x_max_idx_3 = l;
                    }
                }
            }
        }
    }

    // compute mean and standard deviation
    x_mean = x_sum / (x.dimension(0) * x.dimension(1) * x.dimension(2) *
                      x.dimension(3));
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                for (int l = 0; l < x.dimension(3); ++l)
                {
                    float val = x(i, j, k, l);
                    x_stddev += (val - x_mean) * (val - x_mean);
                }
            }
        }
    }
    x_stddev = std::sqrt(x_stddev / (x.dimension(0) * x.dimension(1) *
                                     x.dimension(2) * x.dimension(3)));

    // now print min, max, mean, stddev, and indices
    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ", "
              << x_min_idx_2 << ", " << x_min_idx_3 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ", "
              << x_max_idx_2 << ", " << x_max_idx_3 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
    ;
}

// write function to debug a tensor and pause execution
inline void debug_tensor_3dxcf(const Eigen::Tensor3dXcf &x,
                               const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ", " << x.dimension(2) << ")" << std::endl;

    float x_min_real = 100000000.0f;
    float x_max_real = -100000000.0f;
    float x_min_imag = 100000000.0f;
    float x_max_imag = -100000000.0f;
    float x_sum_real = 0.0f;
    float x_sum_imag = 0.0f;
    float x_mean_real = 0.0f;
    float x_mean_imag = 0.0f;
    float x_stddev_real = 0.0f;
    float x_stddev_imag = 0.0f;

    // store dimension index to save index of min/max
    int x_min_real_idx_0 = -1;
    int x_min_real_idx_1 = -1;
    int x_min_real_idx_2 = -1;
    int x_max_real_idx_0 = -1;
    int x_max_real_idx_1 = -1;
    int x_max_real_idx_2 = -1;

    int x_min_imag_idx_0 = -1;
    int x_min_imag_idx_1 = -1;
    int x_min_imag_idx_2 = -1;
    int x_max_imag_idx_0 = -1;
    int x_max_imag_idx_1 = -1;
    int x_max_imag_idx_2 = -1;

    // loop over tensor and find min/max/sum
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                float real = std::real(x(i, j, k));
                float imag = std::imag(x(i, j, k));
                x_sum_real += real;
                x_sum_imag += imag;
                if (real < x_min_real)
                {
                    x_min_real = real;
                    x_min_real_idx_0 = i;
                    x_min_real_idx_1 = j;
                    x_min_real_idx_2 = k;
                }
                if (real > x_max_real)
                {
                    x_max_real = real;
                    x_max_real_idx_0 = i;
                    x_max_real_idx_1 = j;
                    x_max_real_idx_2 = k;
                }
                if (imag < x_min_imag)
                {
                    x_min_imag = imag;
                    x_min_imag_idx_0 = i;
                    x_min_imag_idx_1 = j;
                    x_min_imag_idx_2 = k;
                }
                if (imag > x_max_imag)
                {
                    x_max_imag = imag;
                    x_max_imag_idx_0 = i;
                    x_max_imag_idx_1 = j;
                    x_max_imag_idx_2 = k;
                }
            }
        }
    }

    // compute mean and standard deviation
    x_mean_real =
        x_sum_real / (x.dimension(0) * x.dimension(1) * x.dimension(2));
    x_mean_imag =
        x_sum_imag / (x.dimension(0) * x.dimension(1) * x.dimension(2));
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                float real = std::real(x(i, j, k));
                float imag = std::imag(x(i, j, k));
                x_stddev_real += (real - x_mean_real) * (real - x_mean_real);
                x_stddev_imag += (imag - x_mean_imag) * (imag - x_mean_imag);
            }
        }
    }
    x_stddev_real = std::sqrt(
        x_stddev_real / (x.dimension(0) * x.dimension(1) * x.dimension(2)));
    x_stddev_imag = std::sqrt(
        x_stddev_imag / (x.dimension(0) * x.dimension(1) * x.dimension(2)));

    // now print min, max, mean, stddev, and indices for real and imaginary
    // parts
    std::cout << "\tmin real: " << x_min_real << std::endl;
    std::cout << "\tmax real: " << x_max_real << std::endl;
    std::cout << "\tmean real: " << x_mean_real << std::endl;
    std::cout << "\tstddev real: " << x_stddev_real << std::endl;
    std::cout << "\tmin real idx: (" << x_min_real_idx_0 << ", "
              << x_min_real_idx_1 << ", " << x_min_real_idx_2 << ")"
              << std::endl;
    std::cout << "\tmax real idx: (" << x_max_real_idx_0 << ", "
              << x_max_real_idx_1 << ", " << x_max_real_idx_2 << ")"
              << std::endl;
    std::cout << "\tsum real: " << x_sum_real << std::endl;

    std::cout << "\tmin imag: " << x_min_imag << std::endl;
    std::cout << "\tmax imag: " << x_max_imag << std::endl;
    std::cout << "\tmean imag: " << x_mean_imag << std::endl;
    std::cout << "\tstddev imag: " << x_stddev_imag << std::endl;
    std::cout << "\tmin imag idx: (" << x_min_imag_idx_0 << ", "
              << x_min_imag_idx_1 << ", " << x_min_imag_idx_2 << ")"
              << std::endl;
    std::cout << "\tmax imag idx: (" << x_max_imag_idx_0 << ", "
              << x_max_imag_idx_1 << ", " << x_max_imag_idx_2 << ")"
              << std::endl;
    std::cout << "\tsum imag: " << x_sum_imag << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For Tensor3dXf
inline void debug_tensor_3dxf(const Eigen::Tensor3dXf &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ", " << x.dimension(2) << ")" << std::endl;

    auto x_min = x.minimum();
    auto x_max = x.maximum();
    Eigen::Tensor<float, 0> x_sum_tensor = x.sum();
    float x_sum = x_sum_tensor(0);
    Eigen::Tensor<float, 0> x_mean_tensor = x.mean();
    float x_mean = x_mean_tensor(0);
    Eigen::Tensor<float, 0> x_stddev_tensor =
        ((x - x_mean).square()).mean().sqrt();
    float x_stddev = x_stddev_tensor(0);

    // You might need to keep the existing loop for this purpose, or use other
    // methods Re-inserting the loop for finding indices of min and max
    int x_min_idx_0 = -1, x_min_idx_1 = -1, x_min_idx_2 = -1;
    int x_max_idx_0 = -1, x_max_idx_1 = -1, x_max_idx_2 = -1;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                float val = x(i, j, k);
                if (val < min_val)
                {
                    min_val = val;
                    x_min_idx_0 = i;
                    x_min_idx_1 = j;
                    x_min_idx_2 = k;
                }
                if (val > max_val)
                {
                    max_val = val;
                    x_max_idx_0 = i;
                    x_max_idx_1 = j;
                    x_max_idx_2 = k;
                }
            }
        }
    }

    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ", "
              << x_min_idx_2 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ", "
              << x_max_idx_2 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For Tensor3dXf
inline void debug_tensor_2dxf(const Eigen::Tensor<float, 2> &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1) << ")"
              << std::endl;

    auto x_min = x.minimum();
    auto x_max = x.maximum();
    Eigen::Tensor<float, 0> x_sum_tensor = x.sum();
    float x_sum = x_sum_tensor(0);
    Eigen::Tensor<float, 0> x_mean_tensor = x.mean();
    float x_mean = x_mean_tensor(0);
    Eigen::Tensor<float, 0> x_stddev_tensor =
        ((x - x_mean).square()).mean().sqrt();
    float x_stddev = x_stddev_tensor(0);

    // You might need to keep the existing loop for this purpose, or use other
    // methods Re-inserting the loop for finding indices of min and max
    int x_min_idx_0 = -1, x_min_idx_1 = -1;
    int x_max_idx_0 = -1, x_max_idx_1 = -1;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            float val = x(i, j);
            if (val < min_val)
            {
                min_val = val;
                x_min_idx_0 = i;
                x_min_idx_1 = j;
            }
            if (val > max_val)
            {
                max_val = val;
                x_max_idx_0 = i;
                x_max_idx_1 = j;
            }
        }
    }

    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ")"
              << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ")"
              << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For Tensor1dXf
inline void debug_tensor_1dxf(const Eigen::Tensor1dXf &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ")" << std::endl;

    float x_min = 100000000.0f;
    float x_max = -100000000.0f;
    float x_sum = 0.0f;
    float x_mean = 0.0f;
    float x_stddev = 0.0f;

    // store dimension index to save index of min/max
    int x_min_idx_0 = -1;
    int x_max_idx_0 = -1;

    // loop over tensor and find min/max/sum
    for (int i = 0; i < x.dimension(0); ++i)
    {
        float val = x(i);
        x_sum += val;
        if (val < x_min)
        {
            x_min = val;
            x_min_idx_0 = i;
        }
        if (val > x_max)
        {
            x_max = val;
            x_max_idx_0 = i;
        }
    }

    // compute mean and standard deviation
    x_mean = x_sum / x.dimension(0);
    for (int i = 0; i < x.dimension(0); ++i)
    {
        float val = x(i);
        x_stddev += (val - x_mean) * (val - x_mean);
    }
    x_stddev = std::sqrt(x_stddev / x.dimension(0));

    // now print min, max, mean, stddev, and indices
    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For Tensor4dXh
inline void debug_tensor_4dxh(const Eigen::Tensor4dXh &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ", " << x.dimension(2) << ", " << x.dimension(3) << ")"
              << std::endl;

    Eigen::half x_min = Eigen::half(100000000.0f);
    Eigen::half x_max = Eigen::half(-100000000.0f);
    float x_sum = 0.0f;
    float x_mean = 0.0f;
    float x_stddev = 0.0f;

    // store dimension index to save index of min/max
    int x_min_idx_0 = -1;
    int x_min_idx_1 = -1;
    int x_min_idx_2 = -1;
    int x_min_idx_3 = -1;
    int x_max_idx_0 = -1;
    int x_max_idx_1 = -1;
    int x_max_idx_2 = -1;
    int x_max_idx_3 = -1;

    // loop over tensor and find min/max/sum
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                for (int l = 0; l < x.dimension(3); ++l)
                {
                    Eigen::half val = x(i, j, k, l);
                    x_sum += val;
                    if (val < x_min)
                    {
                        x_min = val;
                        x_min_idx_0 = i;
                        x_min_idx_1 = j;
                        x_min_idx_2 = k;
                        x_min_idx_3 = l;
                    }
                    if (val > x_max)
                    {
                        x_max = val;
                        x_max_idx_0 = i;
                        x_max_idx_1 = j;
                        x_max_idx_2 = k;
                        x_max_idx_3 = l;
                    }
                }
            }
        }
    }

    // compute mean and standard deviation
    x_mean = x_sum / (x.dimension(0) * x.dimension(1) * x.dimension(2) *
                      x.dimension(3));
    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                for (int l = 0; l < x.dimension(3); ++l)
                {
                    Eigen::half val = x(i, j, k, l);
                    x_stddev += (val - x_mean) * (val - x_mean);
                }
            }
        }
    }
    x_stddev = std::sqrt(x_stddev / (x.dimension(0) * x.dimension(1) *
                                     x.dimension(2) * x.dimension(3)));

    // now print min, max, mean, stddev, and indices
    std::cout << "\tmin: " << Eigen::half(x_min) << std::endl;
    std::cout << "\tmax: " << Eigen::half(x_max) << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ", "
              << x_min_idx_2 << ", " << x_min_idx_3 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ", "
              << x_max_idx_2 << ", " << x_max_idx_3 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For MatrixXf
inline void debug_matrix_xf(const Eigen::MatrixXf &x, const std::string &name)
{
    // return;
    std::cout << "Debugging matrix!: " << name << std::endl;
    std::cout << "\tshape: (" << x.rows() << ", " << x.cols() << ")"
              << std::endl;

    float x_min = 100000000.0f;
    float x_max = -100000000.0f;
    float x_sum = 0.0f;
    float x_mean = 0.0f;
    float x_stddev = 0.0f;

    // store dimension index to save index of min/max
    int x_min_idx_0 = -1;
    int x_min_idx_1 = -1;
    int x_max_idx_0 = -1;
    int x_max_idx_1 = -1;

    // loop over matrix and find min/max/sum
    for (int i = 0; i < x.rows(); ++i)
    {
        for (int j = 0; j < x.cols(); ++j)
        {
            float value = x(i, j);
            x_sum += value;
            if (value < x_min)
            {
                x_min = value;
                x_min_idx_0 = i;
                x_min_idx_1 = j;
            }
            if (value > x_max)
            {
                x_max = value;
                x_max_idx_0 = i;
                x_max_idx_1 = j;
            }
        }
    }

    // compute mean and standard deviation
    x_mean = x_sum / (x.rows() * x.cols());
    for (int i = 0; i < x.rows(); ++i)
    {
        for (int j = 0; j < x.cols(); ++j)
        {
            float value = x(i, j);
            x_stddev += (value - x_mean) * (value - x_mean);
        }
    }
    x_stddev = std::sqrt(x_stddev / (x.rows() * x.cols()));

    // now print min, max, mean, stddev, median, and indices
    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ")"
              << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ")"
              << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// debug VectorXf
inline void debug_vector_xf(const Eigen::VectorXf &x, const std::string &name)
{
    // return;
    std::cout << "Debugging vector!: " << name << std::endl;
    std::cout << "\tshape: (" << x.size() << ")" << std::endl;

    float x_min = 100000000.0f;
    float x_max = -100000000.0f;
    float x_sum = 0.0f;
    float x_mean = 0.0f;
    float x_stddev = 0.0f;

    // store dimension index to save index of min/max
    int x_min_idx_0 = -1;
    int x_max_idx_0 = -1;

    // loop over vector and find min/max/sum
    for (int i = 0; i < x.size(); ++i)
    {
        float value = static_cast<float>(x(i));
        x_sum += value;
        if (value < x_min)
        {
            x_min = value;
            x_min_idx_0 = i;
        }
        if (value > x_max)
        {
            x_max = value;
            x_max_idx_0 = i;
        }
    }

    // compute mean and standard deviation
    x_mean = x_sum / x.size();
    for (int i = 0; i < x.size(); ++i)
    {
        float value = static_cast<float>(x(i));
        x_stddev += (value - x_mean) * (value - x_mean);
    }
    x_stddev = std::sqrt(x_stddev / x.size());

    // now print min, max, mean, stddev, median, and indices
    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

inline void debug_tensor_3dxd(const Eigen::Tensor<double, 3> &x,
                              const std::string &name)
{
    // return;
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ", " << x.dimension(2) << ")" << std::endl;

    auto x_min = x.minimum();
    auto x_max = x.maximum();
    Eigen::Tensor<double, 0> x_sum_tensor = x.sum();
    double x_sum = x_sum_tensor(0);
    Eigen::Tensor<double, 0> x_mean_tensor = x.mean();
    double x_mean = x_mean_tensor(0);
    Eigen::Tensor<double, 0> x_stddev_tensor =
        ((x - x_mean).square()).mean().sqrt();
    double x_stddev = x_stddev_tensor(0);

    // You might need to keep the existing loop for this purpose, or use other
    // methods Re-inserting the loop for finding indices of min and max
    int x_min_idx_0 = -1, x_min_idx_1 = -1, x_min_idx_2 = -1;
    int x_max_idx_0 = -1, x_max_idx_1 = -1, x_max_idx_2 = -1;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    for (int i = 0; i < x.dimension(0); ++i)
    {
        for (int j = 0; j < x.dimension(1); ++j)
        {
            for (int k = 0; k < x.dimension(2); ++k)
            {
                double val = x(i, j, k);
                if (val < min_val)
                {
                    min_val = val;
                    x_min_idx_0 = i;
                    x_min_idx_1 = j;
                    x_min_idx_2 = k;
                }
                if (val > max_val)
                {
                    max_val = val;
                    x_max_idx_0 = i;
                    x_max_idx_1 = j;
                    x_max_idx_2 = k;
                }
            }
        }
    }

    std::cout << "\tmin: " << x_min << std::endl;
    std::cout << "\tmax: " << x_max << std::endl;
    std::cout << "\tmean: " << x_mean << std::endl;
    std::cout << "\tstddev: " << x_stddev << std::endl;
    std::cout << "\tsum: " << x_sum << std::endl;
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ", "
              << x_min_idx_2 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ", "
              << x_max_idx_2 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}
} // namespace demucscppdebug

#define NEAR_TOLERANCE 1e-4

// initialize a struct demucs_model
static struct demucscpp_v3::demucs_v3_model model
{
};
static bool loaded = false;

// google test global setup for model before all tests
static void setUpTestSuite()
{
    if (loaded)
    {
        return;
    }

    // load model from "../ggml-demucs/ggml-model-htdemucs-f16.bin"
    std::string model_file = "../ggml-demucs/ggml-model-hdemucs_mmi-v3-f16.bin";

    auto ret = demucscpp_v3::load_demucs_v3_model(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    loaded = true;
}

// write a basic test case for a stereo file
TEST(DemucsCPP_V3_Layers, FreqEncoders03)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf x_fake(4, 2048, 336);

    // fill with -1, 1 alternating
    for (long i = 0; i < 4; ++i)
    {
        for (long j = 0; j < 2048; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    Eigen::Tensor3dXf x_fake_enc_0(48, 512, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 0, x_fake, x_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_0, "x_fake_enc_0");

    Eigen::Tensor3dXf x_fake_enc_1(96, 128, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 1, x_fake_enc_0, x_fake_enc_1);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_1, "x_fake_enc_1");

    Eigen::Tensor3dXf x_fake_enc_2(192, 32, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 2, x_fake_enc_1, x_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_2, "x_fake_enc_2");

    Eigen::Tensor3dXf x_fake_enc_3(384, 8, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 3, x_fake_enc_2, x_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_3, "x_fake_enc_3");
}

// write a basic test case for a stereo file
TEST(DemucsCPP_V3_Layers, TimeEncoders03)
{
    setUpTestSuite();
    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf xt_fake(1, 2, 343980);

    // fill with -1, 1 alternating
    for (long i = 0; i < 1; ++i)
    {
        for (long j = 0; j < 2; ++j)
        {
            for (long k = 0; k < 343980; ++k)
            {
                if (k % 2 == 0)
                {
                    xt_fake(i, j, k) = -1.0;
                }
                else
                {
                    xt_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake, "xt_fake");

    Eigen::Tensor3dXf xt_fake_enc_0(1, 48, 85995);
    demucscpp_v3::apply_time_encoder_v3(model, 0, xt_fake, xt_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_0, "xt_fake_enc_0");

    Eigen::Tensor3dXf xt_fake_enc_1(1, 96, 21499);
    demucscpp_v3::apply_time_encoder_v3(model, 1, xt_fake_enc_0, xt_fake_enc_1);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_1, "xt_fake_enc_1");

    Eigen::Tensor3dXf xt_fake_enc_2(1, 192, 5375);

    demucscpp_v3::apply_time_encoder_v3(model, 2, xt_fake_enc_1, xt_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_2, "xt_fake_enc_2");

    Eigen::Tensor3dXf xt_fake_enc_3(1, 384, 1344);

    demucscpp_v3::apply_time_encoder_v3(model, 3, xt_fake_enc_2, xt_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_3, "xt_fake_enc_3");
}

TEST(DemucsCPP_V3_Layers, Encoders45)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf x_fake(4, 2048, 336);

    // fill with -1, 1 alternating
    for (long i = 0; i < 4; ++i)
    {
        for (long j = 0; j < 2048; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    Eigen::Tensor3dXf xt_fake(1, 2, 343980);

    // fill with -1, 1 alternating
    for (long i = 0; i < 1; ++i)
    {
        for (long j = 0; j < 2; ++j)
        {
            for (long k = 0; k < 343980; ++k)
            {
                if (k % 2 == 0)
                {
                    xt_fake(i, j, k) = -1.0;
                }
                else
                {
                    xt_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake, "xt_fake");

    // first 4 freq encoders
    Eigen::Tensor3dXf x_fake_enc_0(48, 512, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 0, x_fake, x_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_0, "x_fake_enc_0");

    Eigen::Tensor3dXf x_fake_enc_1(96, 128, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 1, x_fake_enc_0, x_fake_enc_1);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_1, "x_fake_enc_1");

    Eigen::Tensor3dXf x_fake_enc_2(192, 32, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 2, x_fake_enc_1, x_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_2, "x_fake_enc_2");

    Eigen::Tensor3dXf x_fake_enc_3(384, 8, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 3, x_fake_enc_2, x_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_3, "x_fake_enc_3");

    // calculate segment in samples
    int segment_samples =
        (int)(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers with padded sizes
    struct demucscpp_v3::demucs_v3_segment_buffers buffers(2, segment_samples,
                                                     4);

    // then 4 time encoders
    Eigen::Tensor3dXf xt_fake_enc_0(1, 48, 85995);
    demucscpp_v3::apply_time_encoder_v3(model, 0, xt_fake, xt_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_0, "xt_fake_enc_0");

    Eigen::Tensor3dXf xt_fake_enc_1(1, 96, 21499);
    demucscpp_v3::apply_time_encoder_v3(model, 1, xt_fake_enc_0, xt_fake_enc_1);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_1, "xt_fake_enc_1");

    Eigen::Tensor3dXf xt_fake_enc_2(1, 192, 5375);

    demucscpp_v3::apply_time_encoder_v3(model, 2, xt_fake_enc_1, xt_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_2, "xt_fake_enc_2");

    Eigen::Tensor3dXf xt_fake_enc_3(1, 384, 1344);

    demucscpp_v3::apply_time_encoder_v3(model, 3, xt_fake_enc_2, xt_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_3, "xt_fake_enc_3");

    Eigen::Tensor3dXf xt_fake_enc_4(1, 768, 336);
    demucscpp_v3::apply_time_encoder_4(model, xt_fake_enc_3, xt_fake_enc_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_4, "xt_fake_enc_4");

    // now apply the shared encoders with time inject

    Eigen::Tensor3dXf x_fake_enc_4(768, 1, 336);
    demucscpp_v3::apply_freq_encoder_4(model, x_fake_enc_3, xt_fake_enc_4,
                                       x_fake_enc_4, buffers);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_4, "x_fake_enc_4");

    Eigen::Tensor3dXf x_fake_shared_enc_5(1536, 1, 168);
    demucscpp_v3::apply_shared_encoder_5(model, x_fake_enc_4, x_fake_shared_enc_5, buffers);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_shared_enc_5, "x_fake_shared_enc_5");
}

TEST(DemucsCPP_V3_Layers, Decoders01)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    // calculate segment in samples
    int segment_samples =
        (int)(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers with padded sizes
    struct demucscpp_v3::demucs_v3_segment_buffers buffers(2, segment_samples,
                                                     4);

    Eigen::Tensor3dXf x_fake_shared_enc_5(1, 1536, 168);

    // fill with -1, 1 alternating
    for (long j = 0; j < 1536; ++j)
    {
        for (long k = 0; k < 168; ++k)
        {
            if (k % 2 == 0)
            {
                x_fake_shared_enc_5(0, j, k) = -1.0;
            }
            else
            {
                x_fake_shared_enc_5(0, j, k) = 1.0;
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_4(768, 1, 336);
    // fill with alternating -0.5, 0.5

    for (long j = 0; j < 768; ++j)
    {
        for (long k = 0; k < 336; ++k)
        {
            if (k % 2 == 0)
            {
                skip_fake_dec_4(j, 0, k) = 0.5;
            }
            else
            {
                skip_fake_dec_4(j, 0, k) = -0.5;
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_shared_enc_5, "x_fake_shared_enc_5");
    demucscppdebug_test_v3::debug_tensor_3dxf(skip_fake_dec_4, "skip_fake_dec_4");

    Eigen::Tensor3dXf x_fake_dec_4(768, 1, 336);
    Eigen::Tensor3dXf pre_t_unused = demucscpp_v3::apply_shared_decoder_0(model, x_fake_dec_4, x_fake_shared_enc_5);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_4, "x_fake_dec_4");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t_unused, "pre_t_unused");

    Eigen::Tensor3dXf x_fake_dec_3(384, 8, 336);

    Eigen::Tensor3dXf pre_t = demucscpp_v3::apply_freq_decoder_1(
        model, x_fake_dec_4, x_fake_dec_3, skip_fake_dec_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t, "pre_t");

    // remember we leapfrog xt_fake_dec_4
    Eigen::Tensor3dXf xt_fake_dec_3(1, 768, 336);

    demucscpp_v3::apply_time_decoder_0(model, pre_t, xt_fake_dec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_3, "xt_fake_dec_3");
}

TEST(DemucsCPP_V3_Layers, Decoder1Isolated)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    // calculate segment in samples
    int segment_samples =
        (int)(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers with padded sizes
    struct demucscpp_v3::demucs_v3_segment_buffers buffers(2, segment_samples,
                                                     4);

    Eigen::Tensor3dXf x_fake_dec_4(1, 768, 336);

    // fill with -1, 1 alternating
    for (long j = 0; j < 768; ++j)
    {
        for (long k = 0; k < 336; ++k)
        {
            if (k % 2 == 0)
            {
                x_fake_dec_4(0, j, k) = -1.0;
            }
            else
            {
                x_fake_dec_4(0, j, k) = 1.0;
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_4(768, 1, 336);
    // fill with alternating -0.5, 0.5

    for (long j = 0; j < 768; ++j)
    {
        for (long k = 0; k < 336; ++k)
        {
            if (k % 2 == 0)
            {
                skip_fake_dec_4(j, 0, k) = 0.5;
            }
            else
            {
                skip_fake_dec_4(j, 0, k) = -0.5;
            }
        }
    }

    Eigen::Tensor3dXf x_fake_dec_3(384, 8, 336);

    Eigen::Tensor3dXf pre_t = demucscpp_v3::apply_freq_decoder_1(
        model, x_fake_dec_4, x_fake_dec_3, skip_fake_dec_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t, "pre_t");

    // remember we leapfrog xt_fake_dec_4
    Eigen::Tensor3dXf xt_fake_dec_3(1, 768, 336);

    demucscpp_v3::apply_time_decoder_0(model, pre_t, xt_fake_dec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_3, "xt_fake_dec_3");
}

TEST(DemucsCPP_V3_Layers, AllDecoders)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    // calculate segment in samples
    int segment_samples =
        (int)(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers with padded sizes
    struct demucscpp_v3::demucs_v3_segment_buffers buffers(2, segment_samples,
                                                     4);

    Eigen::Tensor3dXf x_fake_shared_enc_5(1, 1536, 168);

    // fill with -1, 1 alternating
    for (long j = 0; j < 1536; ++j)
    {
        for (long k = 0; k < 168; ++k)
        {
            if (k % 2 == 0)
            {
                x_fake_shared_enc_5(0, j, k) = -1.0;
            }
            else
            {
                x_fake_shared_enc_5(0, j, k) = 1.0;
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_4(768, 1, 336);
    // fill with alternating -0.5, 0.5

    for (long j = 0; j < 768; ++j)
    {
        for (long k = 0; k < 336; ++k)
        {
            if (k % 2 == 0)
            {
                skip_fake_dec_4(j, 0, k) = 0.5;
            }
            else
            {
                skip_fake_dec_4(j, 0, k) = -0.5;
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_3(384, 8, 336);
    // fill with alternating -0.5, 0.5

    for (long i = 0; i < 8; ++i)
    {
        for (long j = 0; j < 384; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_3(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_3(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_2(192, 32, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 32; ++i)
    {
        for (long j = 0; j < 192; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_2(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_2(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_1(96, 128, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 128; ++i)
    {
        for (long j = 0; j < 96; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_1(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_1(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_0(48, 512, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 512; ++i)
    {
        for (long j = 0; j < 48; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_0(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_0(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_3(1, 384, 1344);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 384; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 1344; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_3(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_3(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_2(1, 192, 5375);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 192; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 5375; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_2(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_2(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_1(1, 96, 21499);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 96; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 21499; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_1(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_1(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_0(1, 48, 85995);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 48; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 85995; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_0(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_0(j, i, k) = -0.5;
                }
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_shared_enc_5, "x_fake_shared_enc_5");
    demucscppdebug_test_v3::debug_tensor_3dxf(skip_fake_dec_4, "skip_fake_dec_4");

    Eigen::Tensor3dXf x_fake_dec_4(768, 1, 336);
    Eigen::Tensor3dXf pre_t_unused = demucscpp_v3::apply_shared_decoder_0(model, x_fake_dec_4, x_fake_shared_enc_5);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_4, "x_fake_dec_4");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t_unused, "pre_t_unused");

    Eigen::Tensor3dXf x_fake_dec_3(384, 8, 336);
    Eigen::Tensor3dXf pre_t = demucscpp_v3::apply_freq_decoder_1(
        model, x_fake_dec_4, x_fake_dec_3, skip_fake_dec_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t, "pre_t");

    // remember we leapfrog xt_fake_dec_4
    Eigen::Tensor3dXf xt_fake_dec_3(1, 768, 336);
    demucscpp_v3::apply_time_decoder_0(model, pre_t, xt_fake_dec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_3, "xt_fake_dec_3");

    Eigen::Tensor3dXf x_fake_dec_2(192, 32, 336);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(skip_fake_dec_3, "skip_fake_dec_3");

    demucscpp_v3::apply_common_decoder(model, 0, 0, x_fake_dec_3, x_fake_dec_2, skip_fake_dec_3);

    Eigen::Tensor3dXf xt_fake_dec_2(1, 384, 1344);
    demucscpp_v3::apply_common_decoder(model, 1, 0, xt_fake_dec_3, xt_fake_dec_2, skip_fake_tdec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_2, "x_fake_dec_2");
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_2, "xt_fake_dec_2");
}

// write a basic test case for a stereo file
TEST(DemucsCPP_V3_Layers, End2End)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf x_fake(4, 2048, 336);

    // fill with -1, 1 alternating
    for (long i = 0; i < 4; ++i)
    {
        for (long j = 0; j < 2048; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    Eigen::Tensor3dXf x_fake_enc_0(48, 512, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 0, x_fake, x_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_0, "x_fake_enc_0");

    std::cout << "DEBUG HERE!" << std::endl;
    std::cin.ignore();

    Eigen::Tensor3dXf x_fake_enc_1(96, 128, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 1, x_fake_enc_0, x_fake_enc_1);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_1, "x_fake_enc_1");

    Eigen::Tensor3dXf x_fake_enc_2(192, 32, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 2, x_fake_enc_1, x_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_2, "x_fake_enc_2");

    Eigen::Tensor3dXf x_fake_enc_3(384, 8, 336);
    demucscpp_v3::apply_freq_encoder_v3(model, 3, x_fake_enc_2, x_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_3, "x_fake_enc_3");

    Eigen::Tensor3dXf xt_fake(1, 2, 343980);

    // fill with -1, 1 alternating
    for (long i = 0; i < 1; ++i)
    {
        for (long j = 0; j < 2; ++j)
        {
            for (long k = 0; k < 343980; ++k)
            {
                if (k % 2 == 0)
                {
                    xt_fake(i, j, k) = -1.0;
                }
                else
                {
                    xt_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake, "xt_fake");

    Eigen::Tensor3dXf xt_fake_enc_0(1, 48, 85995);
    demucscpp_v3::apply_time_encoder_v3(model, 0, xt_fake, xt_fake_enc_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_0, "xt_fake_enc_0");

    Eigen::Tensor3dXf xt_fake_enc_1(1, 96, 21499);
    demucscpp_v3::apply_time_encoder_v3(model, 1, xt_fake_enc_0, xt_fake_enc_1);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_1, "xt_fake_enc_1");

    Eigen::Tensor3dXf xt_fake_enc_2(1, 192, 5375);

    demucscpp_v3::apply_time_encoder_v3(model, 2, xt_fake_enc_1, xt_fake_enc_2);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_2, "xt_fake_enc_2");

    Eigen::Tensor3dXf xt_fake_enc_3(1, 384, 1344);

    demucscpp_v3::apply_time_encoder_v3(model, 3, xt_fake_enc_2, xt_fake_enc_3);
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_3, "xt_fake_enc_3");

    // calculate segment in samples
    int segment_samples =
        (int)(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers with padded sizes
    struct demucscpp_v3::demucs_v3_segment_buffers buffers(2, segment_samples,
                                                     4);

    Eigen::Tensor3dXf xt_fake_enc_4(1, 768, 336);
    demucscpp_v3::apply_time_encoder_4(model, xt_fake_enc_3, xt_fake_enc_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_enc_4, "xt_fake_enc_4");

    // now apply the shared encoders with time inject

    Eigen::Tensor3dXf x_fake_enc_4(768, 1, 336);
    demucscpp_v3::apply_freq_encoder_4(model, x_fake_enc_3, xt_fake_enc_4,
                                       x_fake_enc_4, buffers);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_enc_4, "x_fake_enc_4");

    Eigen::Tensor3dXf x_fake_shared_enc_5(1536, 1, 168);
    demucscpp_v3::apply_shared_encoder_5(model, x_fake_enc_4, x_fake_shared_enc_5, buffers);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_shared_enc_5, "x_fake_shared_enc_5");

    std::cout << "DEBUG HERE!" << std::endl;
    std::cin.ignore();

    Eigen::Tensor3dXf skip_fake_dec_4(768, 1, 336);
    // fill with alternating -0.5, 0.5

    for (long j = 0; j < 768; ++j)
    {
        for (long k = 0; k < 336; ++k)
        {
            if (k % 2 == 0)
            {
                skip_fake_dec_4(j, 0, k) = 0.5;
            }
            else
            {
                skip_fake_dec_4(j, 0, k) = -0.5;
            }
        }
    }

    demucscppdebug_test_v3::debug_tensor_3dxf(skip_fake_dec_4, "skip_fake_dec_4");

    Eigen::Tensor3dXf x_fake_dec_4(768, 1, 336);
    Eigen::Tensor3dXf pre_t_unused = demucscpp_v3::apply_shared_decoder_0(
        model, x_fake_dec_4, x_fake_shared_enc_5);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_4, "x_fake_dec_4");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t_unused, "pre_t_unused");

    Eigen::Tensor3dXf x_fake_dec_3(384, 8, 336);
    Eigen::Tensor3dXf pre_t = demucscpp_v3::apply_freq_decoder_1(
        model, x_fake_dec_4, x_fake_dec_3, skip_fake_dec_4);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(pre_t, "pre_t");

    // remember we leapfrog xt_fake_dec_4
    Eigen::Tensor3dXf xt_fake_dec_3(1, 768, 336);

    demucscpp_v3::apply_time_decoder_0(model, pre_t, xt_fake_dec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_3, "xt_fake_dec_3");

    std::cout << "DEBUG HERE!" << std::endl;
    std::cin.ignore();

    Eigen::Tensor3dXf skip_fake_dec_3(384, 8, 336);
    // fill with alternating -0.5, 0.5

    for (long i = 0; i < 8; ++i)
    {
        for (long j = 0; j < 384; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_3(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_3(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_2(192, 32, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 32; ++i)
    {
        for (long j = 0; j < 192; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_2(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_2(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_1(96, 128, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 128; ++i)
    {
        for (long j = 0; j < 96; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_1(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_1(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_dec_0(48, 512, 336);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 512; ++i)
    {
        for (long j = 0; j < 48; ++j)
        {
            for (long k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_0(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_0(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_3(1, 384, 1344);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 384; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 1344; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_3(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_3(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_2(1, 192, 5375);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 192; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 5375; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_2(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_2(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_1(1, 96, 21499);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 96; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 21499; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_1(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_1(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf skip_fake_tdec_0(1, 48, 85995);

    // fill with alternating -0.5, 0.5
    for (long i = 0; i < 48; ++i)
    {
        for (long j = 0; j < 1; ++j)
        {
            for (long k = 0; k < 85995; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_tdec_0(j, i, k) = 0.5;
                }
                else
                {
                    skip_fake_tdec_0(j, i, k) = -0.5;
                }
            }
        }
    }

    Eigen::Tensor3dXf x_fake_dec_2(192, 32, 336);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");
    demucscppdebug_test_v3::debug_tensor_3dxf(skip_fake_dec_3, "skip_fake_dec_3");

    demucscpp_v3::apply_common_decoder(model, 0, 0, x_fake_dec_3, x_fake_dec_2, skip_fake_dec_3);

    Eigen::Tensor3dXf xt_fake_dec_2(1, 384, 1344);
    demucscpp_v3::apply_common_decoder(model, 1, 0, xt_fake_dec_3, xt_fake_dec_2, skip_fake_tdec_3);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_2, "x_fake_dec_2");
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_2, "xt_fake_dec_2");

    Eigen::Tensor3dXf x_fake_dec_1(96, 128, 336);
    demucscpp_v3::apply_common_decoder(model, 0, 1, x_fake_dec_2, x_fake_dec_1, skip_fake_dec_2);

    Eigen::Tensor3dXf xt_fake_dec_1(1, 192, 5375);
    demucscpp_v3::apply_common_decoder(model, 1, 1, xt_fake_dec_2, xt_fake_dec_1, skip_fake_tdec_2);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_1, "x_fake_dec_1");
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_1, "xt_fake_dec_1");

    Eigen::Tensor3dXf x_fake_dec_0(48, 512, 336);
    demucscpp_v3::apply_common_decoder(model, 0, 2, x_fake_dec_1, x_fake_dec_0, skip_fake_dec_1);

    Eigen::Tensor3dXf xt_fake_dec_0(1, 96, 21499);
    demucscpp_v3::apply_common_decoder(model, 1, 2, xt_fake_dec_1, xt_fake_dec_0, skip_fake_tdec_1);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_fake_dec_0, "x_fake_dec_0");
    demucscppdebug_test_v3::debug_tensor_3dxf(xt_fake_dec_0, "xt_fake_dec_0");

    // now apply the final decoder
    Eigen::Tensor3dXf x_out(16, 2048, 336);
    demucscpp_v3::apply_common_decoder(model, 0, 3, x_fake_dec_0, x_out, skip_fake_dec_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(x_out, "x_out");

    // now apply the final decoder
    Eigen::Tensor3dXf xt_out(1, 8, 343980);
    demucscpp_v3::apply_common_decoder(model, 1, 3, xt_fake_dec_0, xt_out, skip_fake_tdec_0);

    demucscppdebug_test_v3::debug_tensor_3dxf(xt_out, "xt_out");
}
