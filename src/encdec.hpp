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

// forward declaration to apply a frequency decoder
void apply_freq_decoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                        int decoder_idx, const Eigen::Tensor3dXf &x_in,
                        Eigen::Tensor3dXf &x_out,
                        const Eigen::Tensor3dXf &skip);

// forward declaration to apply a time encoder
void apply_time_encoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                        int encoder_idx, const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out);

// forward declaration to apply a time decoder
void apply_time_decoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                        int decoder_idx, const Eigen::Tensor3dXf &xt_in,
                        Eigen::Tensor3dXf &xt_out,
                        const Eigen::Tensor3dXf &skip);
} // namespace demucscpp_v3

#endif // ENCDEC_HPP
