#ifndef LSTM_HPP
#define LSTM_HPP

#include "model.hpp"
#include <Eigen/Dense>

namespace demucscpp_v3
{

void lstm_forward(const struct demucscpp_v3::demucs_v3_model& model,
                  const int encoder_idx,
                  const int dconv_idx,
                  const Eigen::MatrixXf &input,
                  struct demucscpp_v3::demucs_v3_segment_buffers &data,
                  int hidden_size);


void lstm_reset_zero(const int encoder_idx,
                     const int dconv_idx,
                     struct demucscpp_v3::demucs_v3_segment_buffers &buffers);

}; // namespace demucscpp_v3

#endif // LSTM_HPP
