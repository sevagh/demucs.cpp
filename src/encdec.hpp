#ifndef ENCDEC_HPP
#define ENCDEC_HPP

#include "model.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace demucscpp
{
void apply_freq_encoder(const struct demucscpp::demucs_model &model,
                        int encoder_idx, const Eigen::Tensor3dXf &x_in,
                        Eigen::Tensor3dXf &x_out);

// forward declaration to apply a frequency decoder
void apply_freq_decoder(const struct demucscpp::demucs_model &model,
                        int decoder_idx, const Eigen::Tensor3dXf &x_in,
                        Eigen::Tensor3dXf &x_out,
                        const Eigen::Tensor3dXf &skip);

// forward declaration to apply a time encoder
void apply_time_encoder(const struct demucscpp::demucs_model &model,
                        int encoder_idx, const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out);

// forward declaration to apply a time decoder
void apply_time_decoder(const struct demucscpp::demucs_model &model,
                        int decoder_idx, const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out,
                        const Eigen::Tensor3dXf &skip);
} // namespace demucscpp

namespace demucscpp_v3
{
void apply_freq_encoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                        int encoder_idx, const Eigen::Tensor3dXf &x_in,
                        Eigen::Tensor3dXf &x_out);

// forward declaration to apply a time encoder
void apply_time_encoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                        int encoder_idx, const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out);

// unique time encoder 4
void apply_time_encoder_4(const struct demucscpp_v3::demucs_v3_model &model,
                        const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out);

// freq encoder 4, shared encoder 5
// uniquely contain bilstm, localattn
void apply_freq_shared_encoder_4_5(const struct demucscpp_v3::demucs_v3_model &model,
                                   const Eigen::Tensor3dXf &x_in,
                                   const Eigen::Tensor3dXf &x_inject,
                                   const int encoder_idx,
                                   Eigen::Tensor3dXf &x_out,
                                   struct demucscpp_v3::demucs_v3_segment_buffers &buffers);

Eigen::Tensor3dXf apply_freq_shared_decoder_0_1(
    const struct demucscpp_v3::demucs_v3_model &model,
    const int decoder_idx,
    const Eigen::Tensor3dXf &x_in,
    Eigen::Tensor3dXf &x_out,
    const Eigen::Tensor3dXf &skip);

void apply_time_decoder_0(
    const struct demucscpp_v3::demucs_v3_model &model,
    const Eigen::Tensor3dXf &x_in,
    Eigen::Tensor3dXf &x_out,
    const Eigen::Tensor3dXf &skip);

// forward declaration to apply a common freq or time decoder
void apply_common_decoder(const struct demucscpp_v3::demucs_v3_model &model,
                          const int freq_or_time_idx,
                          const int decoder_idx, const Eigen::Tensor3dXf &x_in,
                          Eigen::Tensor3dXf &x_out,
                          const Eigen::Tensor3dXf &skip);

} // namespace demucscpp_v3

#endif // ENCDEC_HPP
