#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace Eigen
{
// half/float16 typedefs for weights
typedef Tensor<Eigen::half, 3, Eigen::RowMajor> Tensor3dXh;
typedef Tensor<std::complex<Eigen::half>, 3, Eigen::RowMajor> Tensor3dXch;
typedef Tensor<Eigen::half, 1, Eigen::RowMajor> Tensor1dXh;
typedef Tensor<Eigen::half, 4, Eigen::RowMajor> Tensor4dXh;
typedef Vector<Eigen::half, Dynamic> VectorXh;

// define MatrixXh for some layers in demucs
typedef Matrix<Eigen::half, Dynamic, Dynamic, Eigen::RowMajor> MatrixXh;

// define Tensor3dXf, Tensor3dXcf for spectrograms etc.
typedef Tensor<float, 4> Tensor4dXf;
typedef Tensor<float, 3> Tensor3dXf;
typedef Tensor<float, 2> Tensor2dXf;
typedef Tensor<float, 1> Tensor1dXf;
typedef Tensor<std::complex<float>, 3> Tensor3dXcf;
} // namespace Eigen

namespace demucscppdebug
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
    std::cout << "Debugging tensor!: " << name << std::endl;
    std::cout << "\tshape: (" << x.dimension(0) << ", " << x.dimension(1)
              << ")" << std::endl;

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
    std::cout << "\tmin idx: (" << x_min_idx_0 << ", " << x_min_idx_1 << ")" << std::endl;
    std::cout << "\tmax idx: (" << x_max_idx_0 << ", " << x_max_idx_1 << ")" << std::endl;

    std::cout << "FINISHED DEBUG for tensor: " << name << std::endl;
}

// For Tensor1dXf
inline void debug_tensor_1dxf(const Eigen::Tensor1dXf &x,
                              const std::string &name)
{
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
    // std::cin.ignore();
}

inline void debug_tensor_3dxd(const Eigen::Tensor<double, 3> &x,
                              const std::string &name)
{
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

#endif // TENSOR_HPP
