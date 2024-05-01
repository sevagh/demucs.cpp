#ifndef MODEL_HPP
#define MODEL_HPP

#include "dsp.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace demucscpp
{

// Define a type for your callback function
using ProgressCallback = std::function<void(float, const std::string &)>;

const int FREQ_BRANCH_LEN = 336;
const int TIME_BRANCH_LEN_IN = 343980;
const int TIME_BRANCH_LEN_0 = 85995;
const int TIME_BRANCH_LEN_1 = 21499;
const int TIME_BRANCH_LEN_2 = 5375;
const int TIME_BRANCH_LEN_3 = 1344;

struct crosstransformer_base
{
    // crosstransformer.norm_in
    Eigen::Tensor1dXf crosstransformer_norm_in_weight;
    Eigen::Tensor1dXf crosstransformer_norm_in_bias;

    // crosstransformer.norm_in_t
    Eigen::Tensor1dXf crosstransformer_norm_in_t_weight;
    Eigen::Tensor1dXf crosstransformer_norm_in_t_bias;

    // MyTransformerEncoderLayer: index 0, 2, 4
    // second index [2] represents the frequency and time weights (same shapes)
    Eigen::MatrixXf crosstransformer_my_layers_self_attn_in_proj_weight[2][3];
    Eigen::VectorXf crosstransformer_my_layers_self_attn_in_proj_bias[2][3];
    Eigen::MatrixXf crosstransformer_my_layers_self_attn_out_proj_weight[2][3];
    Eigen::VectorXf crosstransformer_my_layers_self_attn_out_proj_bias[2][3];
    Eigen::MatrixXf crosstransformer_my_layers_linear1_weight[2][3];
    Eigen::VectorXf crosstransformer_my_layers_linear1_bias[2][3];
    Eigen::MatrixXf crosstransformer_my_layers_linear2_weight[2][3];
    Eigen::VectorXf crosstransformer_my_layers_linear2_bias[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm1_weight[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm1_bias[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm2_weight[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm2_bias[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm_out_weight[2][3];
    Eigen::Tensor1dXf crosstransformer_my_layers_norm_out_bias[2][3];
    Eigen::VectorXf crosstransformer_my_layers_gamma_1_scale[2][3];
    Eigen::VectorXf crosstransformer_my_layers_gamma_2_scale[2][3];

    // CrossTransformerEncoderLayer: index 1, 3
    // second index [2] represents the frequency and time weights (same shapes)
    Eigen::MatrixXf crosstransformer_cross_layers_cross_attn_in_proj_weight[2]
                                                                           [2];
    Eigen::VectorXf crosstransformer_cross_layers_cross_attn_in_proj_bias[2][2];
    Eigen::MatrixXf crosstransformer_cross_layers_cross_attn_out_proj_weight[2]
                                                                            [2];
    Eigen::VectorXf crosstransformer_cross_layers_cross_attn_out_proj_bias[2]
                                                                          [2];
    Eigen::MatrixXf crosstransformer_cross_layers_linear1_weight[2][2];
    Eigen::VectorXf crosstransformer_cross_layers_linear1_bias[2][2];
    Eigen::MatrixXf crosstransformer_cross_layers_linear2_weight[2][2];
    Eigen::VectorXf crosstransformer_cross_layers_linear2_bias[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm1_weight[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm1_bias[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm2_weight[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm2_bias[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm3_weight[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm3_bias[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm_out_weight[2][2];
    Eigen::Tensor1dXf crosstransformer_cross_layers_norm_out_bias[2][2];
    Eigen::VectorXf crosstransformer_cross_layers_gamma_1_scale[2][2];
    Eigen::VectorXf crosstransformer_cross_layers_gamma_2_scale[2][2];

    crosstransformer_base(int size1, int size2, int size3)
        : crosstransformer_norm_in_weight(Eigen::Tensor1dXf(size1)),
          crosstransformer_norm_in_bias(Eigen::Tensor1dXf(size1)),
          crosstransformer_norm_in_t_weight(Eigen::Tensor1dXf(size1)),
          crosstransformer_norm_in_t_bias(Eigen::Tensor1dXf(size1)),
          // second index [2] represents the frequency and time weights (same
          // shapes)
          crosstransformer_my_layers_self_attn_in_proj_weight{
              {{Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)}},
              {{Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)}}},
          crosstransformer_my_layers_self_attn_in_proj_bias{
              {{Eigen::VectorXf(size2)},
               {Eigen::VectorXf(size2)},
               {Eigen::VectorXf(size2)}},
              {{Eigen::VectorXf(size2)},
               {Eigen::VectorXf(size2)},
               {Eigen::VectorXf(size2)}}},
          crosstransformer_my_layers_self_attn_out_proj_weight{
              {{Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)}},
              {{Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)}}},
          crosstransformer_my_layers_self_attn_out_proj_bias{
              {{Eigen::VectorXf(size1)},
               {Eigen::VectorXf(size1)},
               {Eigen::VectorXf(size1)}},
              {{Eigen::VectorXf(size1)},
               {Eigen::VectorXf(size1)},
               {Eigen::VectorXf(size1)}}},
          crosstransformer_my_layers_linear1_weight{
              {{Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)}},
              {{Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)}}},
          crosstransformer_my_layers_linear1_bias{{{Eigen::VectorXf(size3)},
                                                   {Eigen::VectorXf(size3)},
                                                   {Eigen::VectorXf(size3)}},
                                                  {{Eigen::VectorXf(size3)},
                                                   {Eigen::VectorXf(size3)},
                                                   {Eigen::VectorXf(size3)}}},
          crosstransformer_my_layers_linear2_weight{
              {{Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)}},
              {{Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)}}},
          crosstransformer_my_layers_linear2_bias{{{Eigen::VectorXf(size1)},
                                                   {Eigen::VectorXf(size1)},
                                                   {Eigen::VectorXf(size1)}},
                                                  {{Eigen::VectorXf(size1)},
                                                   {Eigen::VectorXf(size1)},
                                                   {Eigen::VectorXf(size1)}}},
          crosstransformer_my_layers_norm1_weight{{{Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)}},
                                                  {{Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_norm1_bias{{{Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)}},
                                                {{Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_norm2_weight{{{Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)}},
                                                  {{Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)},
                                                   {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_norm2_bias{{{Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)}},
                                                {{Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)},
                                                 {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_norm_out_weight{
              {{Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_norm_out_bias{
              {{Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)},
               {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_my_layers_gamma_1_scale{{{Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)}},
                                                   {{Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)}}},
          crosstransformer_my_layers_gamma_2_scale{{{Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)}},
                                                   {{Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)},
                                                    {Eigen::VectorXf(size1)}}},
          crosstransformer_cross_layers_cross_attn_in_proj_weight{
              {{Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)}},
              {{Eigen::MatrixXf(size2, size1)},
               {Eigen::MatrixXf(size2, size1)}}},
          crosstransformer_cross_layers_cross_attn_in_proj_bias{
              {{Eigen::VectorXf(size2)}, {Eigen::VectorXf(size2)}},
              {{Eigen::VectorXf(size2)}, {Eigen::VectorXf(size2)}}},
          crosstransformer_cross_layers_cross_attn_out_proj_weight{
              {{Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)}},
              {{Eigen::MatrixXf(size1, size1)},
               {Eigen::MatrixXf(size1, size1)}}},
          crosstransformer_cross_layers_cross_attn_out_proj_bias{
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}},
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}}},
          crosstransformer_cross_layers_linear1_weight{
              {{Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)}},
              {{Eigen::MatrixXf(size3, size1)},
               {Eigen::MatrixXf(size3, size1)}}},
          crosstransformer_cross_layers_linear1_bias{
              {{Eigen::VectorXf(size3)}, {Eigen::VectorXf(size3)}},
              {{Eigen::VectorXf(size3)}, {Eigen::VectorXf(size3)}}},
          crosstransformer_cross_layers_linear2_weight{
              {{Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)}},
              {{Eigen::MatrixXf(size1, size3)},
               {Eigen::MatrixXf(size1, size3)}}},
          crosstransformer_cross_layers_linear2_bias{
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}},
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}}},
          crosstransformer_cross_layers_norm1_weight{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm1_bias{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm2_weight{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm2_bias{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm3_weight{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm3_bias{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm_out_weight{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_norm_out_bias{
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}},
              {{Eigen::Tensor1dXf(size1)}, {Eigen::Tensor1dXf(size1)}}},
          crosstransformer_cross_layers_gamma_1_scale{
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}},
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}}},
          crosstransformer_cross_layers_gamma_2_scale{
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}},
              {{Eigen::VectorXf(size1)}, {Eigen::VectorXf(size1)}}}
    {
    }

    // Common members and methods...
    virtual ~crosstransformer_base() = default;
};

struct demucs_crosstransformer_4s : crosstransformer_base
{
    demucs_crosstransformer_4s() : crosstransformer_base(512, 1536, 2048){};

    // channel_upsampler
    Eigen::Tensor3dXf channel_upsampler_weight{Eigen::Tensor3dXf(512, 384, 1)};
    Eigen::Tensor1dXf channel_upsampler_bias{Eigen::Tensor1dXf(512)};
    // channel_downsampler
    Eigen::Tensor3dXf channel_downsampler_weight{
        Eigen::Tensor3dXf(384, 512, 1)};
    Eigen::Tensor1dXf channel_downsampler_bias{Eigen::Tensor1dXf(384)};
    // channel_upsampler_t
    Eigen::Tensor3dXf channel_upsampler_t_weight{
        Eigen::Tensor3dXf(512, 384, 1)};
    Eigen::Tensor1dXf channel_upsampler_t_bias{Eigen::Tensor1dXf(512)};
    // channel_downsampler_t
    Eigen::Tensor3dXf channel_downsampler_t_weight{
        Eigen::Tensor3dXf(384, 512, 1)};
    Eigen::Tensor1dXf channel_downsampler_t_bias{Eigen::Tensor1dXf(384)};
};

struct demucs_crosstransformer_6s : crosstransformer_base
{
    demucs_crosstransformer_6s() : crosstransformer_base(384, 1152, 1536){};
};

struct demucs_model
{
    bool is_4sources;

    // Encoders 0-3
    Eigen::Tensor3dXf encoder_conv_weight[4]{
        Eigen::Tensor3dXf(48, 4, 8),
        Eigen::Tensor3dXf(96, 48, 8),
        Eigen::Tensor3dXf(192, 96, 8),
        Eigen::Tensor3dXf(384, 192, 8),
    };

    Eigen::Tensor1dXf encoder_conv_bias[4]{
        Eigen::Tensor1dXf(48),
        Eigen::Tensor1dXf(96),
        Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384),
    };

    Eigen::Tensor3dXf encoder_rewrite_weight[4]{
        Eigen::Tensor3dXf(96, 48, 1),
        Eigen::Tensor3dXf(192, 96, 1),
        Eigen::Tensor3dXf(384, 192, 1),
        Eigen::Tensor3dXf(768, 384, 1),
    };

    Eigen::Tensor1dXf encoder_rewrite_bias[4]{
        Eigen::Tensor1dXf(96),
        Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384),
        Eigen::Tensor1dXf(768),
    };

    // TEncoder 0-3
    Eigen::Tensor3dXf tencoder_conv_weight[4] = {
        Eigen::Tensor3dXf(48, 2, 8), Eigen::Tensor3dXf(96, 48, 8),
        Eigen::Tensor3dXf(192, 96, 8), Eigen::Tensor3dXf(384, 192, 8)};

    Eigen::Tensor1dXf tencoder_conv_bias[4] = {
        Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384)};

    Eigen::Tensor3dXf tencoder_rewrite_weight[4] = {
        Eigen::Tensor3dXf(96, 48, 1), Eigen::Tensor3dXf(192, 96, 1),
        Eigen::Tensor3dXf(384, 192, 1), Eigen::Tensor3dXf(768, 384, 1)};

    Eigen::Tensor1dXf tencoder_rewrite_bias[4] = {
        Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(384),
        Eigen::Tensor1dXf(768)};

    // Decoders 0-3
    Eigen::Tensor4dXf decoder_conv_tr_weight[4] = {
        Eigen::Tensor4dXf(384, 192, 8, 1), Eigen::Tensor4dXf(192, 96, 8, 1),
        Eigen::Tensor4dXf(96, 48, 8, 1), Eigen::Tensor4dXf(48, 16, 8, 1)};

    Eigen::Tensor1dXf decoder_conv_tr_bias[4] = {
        Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(48),
        Eigen::Tensor1dXf(16)};

    Eigen::Tensor4dXf decoder_rewrite_weight[4] = {
        Eigen::Tensor4dXf(768, 384, 3, 3), Eigen::Tensor4dXf(384, 192, 3, 3),
        Eigen::Tensor4dXf(192, 96, 3, 3), Eigen::Tensor4dXf(96, 48, 3, 3)};

    Eigen::Tensor1dXf decoder_rewrite_bias[4] = {
        Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(96)};

    // TDecoder 0-3
    Eigen::Tensor3dXf tdecoder_conv_tr_weight[4] = {
        Eigen::Tensor3dXf(384, 192, 8), Eigen::Tensor3dXf(192, 96, 8),
        Eigen::Tensor3dXf(96, 48, 8), Eigen::Tensor3dXf(48, 8, 8)};

    Eigen::Tensor1dXf tdecoder_conv_tr_bias[4] = {
        Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(48),
        Eigen::Tensor1dXf(8)};

    Eigen::Tensor3dXf tdecoder_rewrite_weight[4] = {
        Eigen::Tensor3dXf(768, 384, 3), Eigen::Tensor3dXf(384, 192, 3),
        Eigen::Tensor3dXf(192, 96, 3), Eigen::Tensor3dXf(96, 48, 3)};

    Eigen::Tensor1dXf tdecoder_rewrite_bias[4] = {
        Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(96)};

    // DConv layers
    // first index: time or frequency
    // second index: encoder or decoder
    // third index: enc/dec layer number
    // fourth index: dconv 0 or 1
    Eigen::Tensor3dXf dconv_layers_0_conv1d_weight[2][2][4][2]{
        {
            {{Eigen::Tensor3dXf(6, 48, 3), Eigen::Tensor3dXf(6, 48, 3)},
             {Eigen::Tensor3dXf(12, 96, 3), Eigen::Tensor3dXf(12, 96, 3)},
             {Eigen::Tensor3dXf(24, 192, 3), Eigen::Tensor3dXf(24, 192, 3)},
             {Eigen::Tensor3dXf(48, 384, 3), Eigen::Tensor3dXf(48, 384, 3)}},
            {{Eigen::Tensor3dXf(6, 48, 3), Eigen::Tensor3dXf(6, 48, 3)},
             {Eigen::Tensor3dXf(12, 96, 3), Eigen::Tensor3dXf(12, 96, 3)},
             {Eigen::Tensor3dXf(24, 192, 3), Eigen::Tensor3dXf(24, 192, 3)},
             {Eigen::Tensor3dXf(48, 384, 3), Eigen::Tensor3dXf(48, 384, 3)}},
        },
        {
            {{Eigen::Tensor3dXf(6, 48, 3), Eigen::Tensor3dXf(6, 48, 3)},
             {Eigen::Tensor3dXf(12, 96, 3), Eigen::Tensor3dXf(12, 96, 3)},
             {Eigen::Tensor3dXf(24, 192, 3), Eigen::Tensor3dXf(24, 192, 3)},
             {Eigen::Tensor3dXf(48, 384, 3), Eigen::Tensor3dXf(48, 384, 3)}},
            {{Eigen::Tensor3dXf(6, 48, 3), Eigen::Tensor3dXf(6, 48, 3)},
             {Eigen::Tensor3dXf(12, 96, 3), Eigen::Tensor3dXf(12, 96, 3)},
             {Eigen::Tensor3dXf(24, 192, 3), Eigen::Tensor3dXf(24, 192, 3)},
             {Eigen::Tensor3dXf(48, 384, 3), Eigen::Tensor3dXf(48, 384, 3)}},
        }};

    Eigen::Tensor1dXf dconv_layers_0_conv1d_bias[2][2][4][2]{
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}},
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}}};

    Eigen::Tensor1dXf dconv_layers_1_groupnorm_weight[2][2][4][2]{
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}},
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}}};

    Eigen::Tensor1dXf dconv_layers_1_groupnorm_bias[2][2][4][2]{
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}},
        {{{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}},
         {{Eigen::Tensor1dXf(6), Eigen::Tensor1dXf(6)},
          {Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)}}}};

    Eigen::Tensor3dXf dconv_layers_3_conv1d_weight[2][2][4][2]{
        {{{Eigen::Tensor3dXf(96, 6, 1), Eigen::Tensor3dXf(96, 6, 1)},
          {Eigen::Tensor3dXf(192, 12, 1), Eigen::Tensor3dXf(192, 12, 1)},
          {Eigen::Tensor3dXf(384, 24, 1), Eigen::Tensor3dXf(384, 24, 1)},
          {Eigen::Tensor3dXf(768, 48, 1), Eigen::Tensor3dXf(768, 48, 1)}},
         {{Eigen::Tensor3dXf(96, 6, 1), Eigen::Tensor3dXf(96, 6, 1)},
          {Eigen::Tensor3dXf(192, 12, 1), Eigen::Tensor3dXf(192, 12, 1)},
          {Eigen::Tensor3dXf(384, 24, 1), Eigen::Tensor3dXf(384, 24, 1)},
          {Eigen::Tensor3dXf(768, 48, 1), Eigen::Tensor3dXf(768, 48, 1)}}},
        {{{Eigen::Tensor3dXf(96, 6, 1), Eigen::Tensor3dXf(96, 6, 1)},
          {Eigen::Tensor3dXf(192, 12, 1), Eigen::Tensor3dXf(192, 12, 1)},
          {Eigen::Tensor3dXf(384, 24, 1), Eigen::Tensor3dXf(384, 24, 1)},
          {Eigen::Tensor3dXf(768, 48, 1), Eigen::Tensor3dXf(768, 48, 1)}},
         {{Eigen::Tensor3dXf(96, 6, 1), Eigen::Tensor3dXf(96, 6, 1)},
          {Eigen::Tensor3dXf(192, 12, 1), Eigen::Tensor3dXf(192, 12, 1)},
          {Eigen::Tensor3dXf(384, 24, 1), Eigen::Tensor3dXf(384, 24, 1)},
          {Eigen::Tensor3dXf(768, 48, 1), Eigen::Tensor3dXf(768, 48, 1)}}}};

    Eigen::Tensor1dXf dconv_layers_3_conv1d_bias[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_4_groupnorm_weight[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_4_groupnorm_bias[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_6_scale[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
        },
        {
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
        }};

    // freq_emb
    Eigen::MatrixXf freq_emb_embedding_weight{Eigen::MatrixXf(512, 48)};

    std::unique_ptr<crosstransformer_base> crosstransformer;
};

inline std::unique_ptr<crosstransformer_base>
initialize_crosstransformer(bool is_4sources)
{
    if (is_4sources)
    {
        return std::make_unique<struct demucs_crosstransformer_4s>();
    }
    else
    {
        return std::make_unique<struct demucs_crosstransformer_6s>();
    }
}

struct demucs_segment_buffers
{
    int segment_samples;
    int le;
    int pad;
    int pad_end;
    int padded_segment_samples;
    int nb_stft_frames;
    int nb_stft_bins;

    Eigen::MatrixXf mix;
    Eigen::Tensor3dXf targets_out;
    Eigen::MatrixXf padded_mix;
    Eigen::Tensor3dXcf z;

    // freq branch, one for each encoded representation
    Eigen::Tensor3dXf x;     // input
    Eigen::Tensor3dXf x_out; // input
    Eigen::Tensor3dXf x_0;
    Eigen::Tensor3dXf x_1;
    Eigen::Tensor3dXf x_2;
    Eigen::Tensor3dXf x_3;
    Eigen::Tensor3dXf x_3_channel_upsampled;

    // time branch
    Eigen::Tensor3dXf xt;             // input
    Eigen::Tensor3dXf xt_out;         // output
    Eigen::Tensor3dXf xt_decoded_out; // hold time decoder output
    Eigen::Tensor3dXf xt_0;
    Eigen::Tensor3dXf xt_1;
    Eigen::Tensor3dXf xt_2;
    Eigen::Tensor3dXf xt_3;
    Eigen::Tensor3dXf xt_3_channel_upsampled;

    // skip conns for frequency and time
    // easier as hardcoded matrix sizes
    Eigen::Tensor3dXf saved_0;
    Eigen::Tensor3dXf saved_1;
    Eigen::Tensor3dXf saved_2;
    Eigen::Tensor3dXf saved_3;

    Eigen::Tensor3dXf savedt_0;
    Eigen::Tensor3dXf savedt_1;
    Eigen::Tensor3dXf savedt_2;
    Eigen::Tensor3dXf savedt_3;

    // constructor for demucs_segment_buffers that takes int parameters

    // let's do pesky precomputing of the signal repadding to 1/4 hop
    // for time and frequency alignment
    demucs_segment_buffers(int nb_channels, int segment_samples, int nb_sources)
        : segment_samples(segment_samples),
          le(int(std::ceil((float)segment_samples / (float)FFT_HOP_SIZE))),
          pad(std::floor((float)FFT_HOP_SIZE / 2.0f) * 3),
          pad_end(pad + le * FFT_HOP_SIZE - segment_samples),
          padded_segment_samples(segment_samples + pad + pad_end),
          nb_stft_frames(segment_samples / demucscpp::FFT_HOP_SIZE + 1),
          nb_stft_bins(demucscpp::FFT_WINDOW_SIZE / 2 + 1),
          mix(nb_channels, segment_samples),
          targets_out(nb_sources, nb_channels, segment_samples),
          padded_mix(nb_channels, padded_segment_samples),
          z(nb_channels, nb_stft_bins, nb_stft_frames),
          // complex-as-channels implies 2*nb_channels for real+imag
          x(2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          x_out(nb_sources * 2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          x_0(48, 512, FREQ_BRANCH_LEN), x_1(96, 128, FREQ_BRANCH_LEN),
          x_2(192, 32, FREQ_BRANCH_LEN), x_3(384, 8, FREQ_BRANCH_LEN),
          x_3_channel_upsampled(512, 8, FREQ_BRANCH_LEN),
          xt(1, nb_channels, segment_samples),
          xt_out(1, nb_sources * nb_channels, segment_samples),
          xt_decoded_out(1, 8, segment_samples), xt_0(1, 48, TIME_BRANCH_LEN_0),
          xt_1(1, 96, TIME_BRANCH_LEN_1), xt_2(1, 192, TIME_BRANCH_LEN_2),
          xt_3(1, 384, TIME_BRANCH_LEN_3),
          xt_3_channel_upsampled(1, 512, TIME_BRANCH_LEN_3),
          saved_0(48, 512, FREQ_BRANCH_LEN), saved_1(96, 128, FREQ_BRANCH_LEN),
          saved_2(192, 32, FREQ_BRANCH_LEN), saved_3(384, 8, FREQ_BRANCH_LEN),
          savedt_0(1, 48, TIME_BRANCH_LEN_0),
          savedt_1(1, 96, TIME_BRANCH_LEN_1),
          savedt_2(1, 192, TIME_BRANCH_LEN_2),
          savedt_3(1, 384, TIME_BRANCH_LEN_3){};
};

bool load_demucs_model(const std::string &model_dir,
                       struct demucs_model *model);

const float SEGMENT_LEN_SECS = 7.8;      // 8 seconds, the demucs chunk size
const float SEGMENT_OVERLAP_SECS = 0.25; // 0.25 overlap
const float MAX_SHIFT_SECS = 0.5;        // max shift
const float OVERLAP = 0.25;              // overlap between segments
const float TRANSITION_POWER = 1.0;      // transition between segments

Eigen::Tensor3dXf demucs_inference(const struct demucs_model &model,
                                   const Eigen::MatrixXf &full_audio,
                                   ProgressCallback cb);

void model_inference(const struct demucs_model &model,
                     struct demucscpp::demucs_segment_buffers &buffers,
                     struct demucscpp::stft_buffers &stft_buf,
                     ProgressCallback cb, float current_progress,
                     float segment_progress);
} // namespace demucscpp

// V3 Hybrid time-frequency model (no transformer)
namespace demucscpp_v3
{

const int FREQ_BRANCH_LEN = 336;
const int TIME_BRANCH_LEN_IN = 343980;
const int TIME_BRANCH_LEN_0 = 85995;
const int TIME_BRANCH_LEN_1 = 21499;
const int TIME_BRANCH_LEN_2 = 5375;
const int TIME_BRANCH_LEN_3 = 1344;
const int TIME_BRANCH_LEN_4 = 336;

const int SHARED_BRANCH_LEN = 168;

// dconv lstm constants
// the seq len is 336, the final encoded time branch length
// (for both time and frequency)
const int LSTM_HIDDEN_SIZE_0 = 192;
const int LSTM_HIDDEN_SIZE_1 = 384;

// dconv localstate
const int LOCAL_ATTN_N_HEADS = 4;
const int LOCAL_ATTN_N_FREQS = 0;
const int LOCAL_ATTN_N_DECAY = 4;
const int LOCAL_ATTN_CHANNELS = 192;

struct demucs_v3_model
{
    // Encoder convolution layers
    Eigen::Tensor3dXf encoder_conv_weight[4] = {
        Eigen::Tensor3dXf(48, 4, 8),
        Eigen::Tensor3dXf(96, 48, 8),
        Eigen::Tensor3dXf(192, 96, 8),
        Eigen::Tensor3dXf(384, 192, 8)
    };

    Eigen::Tensor1dXf encoder_conv_bias[4] = {
        Eigen::Tensor1dXf(48),
        Eigen::Tensor1dXf(96),
        Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384)
    };

    // Encoder rewrite layers
    Eigen::Tensor3dXf encoder_rewrite_weight[4] = {
        Eigen::Tensor3dXf(96, 48, 1),
        Eigen::Tensor3dXf(192, 96, 1),
        Eigen::Tensor3dXf(384, 192, 1),
        Eigen::Tensor3dXf(768, 384, 1)
    };

    Eigen::Tensor1dXf encoder_rewrite_bias[4] = {
        Eigen::Tensor1dXf(96),
        Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384),
        Eigen::Tensor1dXf(768)
    };

    // TEncoder 0-3
    Eigen::Tensor3dXf tencoder_conv_weight[4] = {
        Eigen::Tensor3dXf(48, 2, 8), Eigen::Tensor3dXf(96, 48, 8),
        Eigen::Tensor3dXf(192, 96, 8), Eigen::Tensor3dXf(384, 192, 8)};

    Eigen::Tensor1dXf tencoder_conv_bias[4] = {
        Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(192),
        Eigen::Tensor1dXf(384)};

    Eigen::Tensor3dXf tencoder_rewrite_weight[4] = {
        Eigen::Tensor3dXf(96, 48, 1), Eigen::Tensor3dXf(192, 96, 1),
        Eigen::Tensor3dXf(384, 192, 1), Eigen::Tensor3dXf(768, 384, 1)};

    Eigen::Tensor1dXf tencoder_rewrite_bias[4] = {
        Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(384),
        Eigen::Tensor1dXf(768)};

    // DConv layers
    // first index: time or frequency
    // second index: encoder or decoder
    // third index: enc/dec layer number
    // fourth index: dconv 0 or 1
    // this takes care of 4 freq encoders and 4 time encoders
    // each with 2 dconv layers
    Eigen::Tensor3dXf dconv_layers_0_conv1d_weight[2][2][4][2]{
        {
            {{Eigen::Tensor3dXf(12, 48, 3), Eigen::Tensor3dXf(12, 48, 3)},
             {Eigen::Tensor3dXf(24, 96, 3), Eigen::Tensor3dXf(24, 96, 3)},
             {Eigen::Tensor3dXf(48, 192, 3), Eigen::Tensor3dXf(48, 192, 3)},
             {Eigen::Tensor3dXf(96, 384, 3), Eigen::Tensor3dXf(96, 384, 3)}},
            {{Eigen::Tensor3dXf(12, 48, 3), Eigen::Tensor3dXf(12, 48, 3)},
             {Eigen::Tensor3dXf(24, 96, 3), Eigen::Tensor3dXf(24, 96, 3)},
             {Eigen::Tensor3dXf(48, 192, 3), Eigen::Tensor3dXf(48, 192, 3)},
             {Eigen::Tensor3dXf(96, 384, 3), Eigen::Tensor3dXf(96, 384, 3)}},
        },
        {
            {{Eigen::Tensor3dXf(12, 48, 3), Eigen::Tensor3dXf(12, 48, 3)},
             {Eigen::Tensor3dXf(24, 96, 3), Eigen::Tensor3dXf(24, 96, 3)},
             {Eigen::Tensor3dXf(48, 192, 3), Eigen::Tensor3dXf(48, 192, 3)},
             {Eigen::Tensor3dXf(96, 384, 3), Eigen::Tensor3dXf(96, 384, 3)}},
            {{Eigen::Tensor3dXf(12, 48, 3), Eigen::Tensor3dXf(12, 48, 3)},
             {Eigen::Tensor3dXf(24, 96, 3), Eigen::Tensor3dXf(24, 96, 3)},
             {Eigen::Tensor3dXf(48, 192, 3), Eigen::Tensor3dXf(48, 192, 3)},
             {Eigen::Tensor3dXf(96, 384, 3), Eigen::Tensor3dXf(96, 384, 3)}},
        }};

    Eigen::Tensor1dXf dconv_layers_0_conv1d_bias[2][2][4][2]{
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}},
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}
        }};

    Eigen::Tensor1dXf dconv_layers_1_groupnorm_weight[2][2][4][2]{
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}},
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}}};

    Eigen::Tensor1dXf dconv_layers_1_groupnorm_bias[2][2][4][2]{
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}},
        {{{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}},
         {{Eigen::Tensor1dXf(12), Eigen::Tensor1dXf(12)},
          {Eigen::Tensor1dXf(24), Eigen::Tensor1dXf(24)},
          {Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
          {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)}}}};

    Eigen::Tensor3dXf dconv_layers_3_conv1d_weight[2][2][4][2]{
        {{{Eigen::Tensor3dXf(96, 12, 1), Eigen::Tensor3dXf(96, 12, 1)},
          {Eigen::Tensor3dXf(192, 24, 1), Eigen::Tensor3dXf(192, 24, 1)},
          {Eigen::Tensor3dXf(384, 48, 1), Eigen::Tensor3dXf(384, 48, 1)},
          {Eigen::Tensor3dXf(768, 96, 1), Eigen::Tensor3dXf(768, 96, 1)}},
         {{Eigen::Tensor3dXf(96, 12, 1), Eigen::Tensor3dXf(96, 12, 1)},
          {Eigen::Tensor3dXf(192, 24, 1), Eigen::Tensor3dXf(192, 24, 1)},
          {Eigen::Tensor3dXf(384, 48, 1), Eigen::Tensor3dXf(384, 48, 1)},
          {Eigen::Tensor3dXf(768, 96, 1), Eigen::Tensor3dXf(768, 96, 1)}}},
        {{{Eigen::Tensor3dXf(96, 12, 1), Eigen::Tensor3dXf(96, 12, 1)},
          {Eigen::Tensor3dXf(192, 24, 1), Eigen::Tensor3dXf(192, 24, 1)},
          {Eigen::Tensor3dXf(384, 48, 1), Eigen::Tensor3dXf(384, 48, 1)},
          {Eigen::Tensor3dXf(768, 96, 1), Eigen::Tensor3dXf(768, 96, 1)}},
         {{Eigen::Tensor3dXf(96, 12, 1), Eigen::Tensor3dXf(96, 12, 1)},
          {Eigen::Tensor3dXf(192, 24, 1), Eigen::Tensor3dXf(192, 24, 1)},
          {Eigen::Tensor3dXf(384, 48, 1), Eigen::Tensor3dXf(384, 48, 1)},
          {Eigen::Tensor3dXf(768, 96, 1), Eigen::Tensor3dXf(768, 96, 1)}}}};

    Eigen::Tensor1dXf dconv_layers_3_conv1d_bias[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_4_groupnorm_weight[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_4_groupnorm_bias[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
            {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
             {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
        },
        {{{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}},
         {{Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
          {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
          {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)},
          {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)}}}};

    Eigen::Tensor1dXf dconv_layers_6_scale[2][2][4][2]{
        {
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
        },
        {
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
            {{Eigen::Tensor1dXf(48), Eigen::Tensor1dXf(48)},
             {Eigen::Tensor1dXf(96), Eigen::Tensor1dXf(96)},
             {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
             {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}},
        }};

    // time encoder 4 is super simple, just 1 conv
    Eigen::Tensor3dXf tencoder_4_conv_weight{
        Eigen::Tensor3dXf(768, 384, 8)};

    Eigen::Tensor1dXf tencoder_4_conv_bias{
        Eigen::Tensor1dXf(768)};

    // freq encoder 4 and shared encoder 5
    // have the bilistm and localattention layers, similar to each other
    // index of two to hold both
    Eigen::Tensor4dXf encoder_4_conv_weight{Eigen::Tensor4dXf(768, 384, 8, 1)};
    Eigen::Tensor3dXf encoder_5_conv_weight{Eigen::Tensor3dXf(1536, 768, 4)};

    Eigen::Tensor1dXf encoder_4_5_conv_bias[2]{
        Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(1536)};

    Eigen::Tensor1dXf encoder_4_5_norm1_weight[2]{
        Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(1536)};

    Eigen::Tensor1dXf encoder_4_5_norm1_bias[2]{
        Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(1536)};

    Eigen::Tensor3dXf encoder_4_5_rewrite_weight[2]{
        Eigen::Tensor3dXf(1536, 768, 1), Eigen::Tensor3dXf(3072, 1536, 1)};

    Eigen::Tensor1dXf encoder_4_5_rewrite_bias[2]{
        Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(3072)};

    Eigen::Tensor1dXf encoder_4_5_norm2_weight[2]{
        Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(3072)};

    Eigen::Tensor1dXf encoder_4_5_norm2_bias[2]{
        Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(3072)};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_0_conv1d_weight[2][2]{
        {Eigen::Tensor3dXf(192, 768, 3), Eigen::Tensor3dXf(192, 768, 3)},
        {Eigen::Tensor3dXf(384, 1536, 3), Eigen::Tensor3dXf(384, 1536, 3)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_0_conv1d_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_1_groupnorm_weight[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_1_groupnorm_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    // 2 encoders, 2 dconv layers, 2 layer bi-lstm (2 layers 2 directions) => [2][2][2][2]
    // first index = encoder, second index = dconv layer, third index = layer, fourth index = direction
    Eigen::MatrixXf encoder_4_5_dconv_layers_3_lstm_ih_w[2][2][2][2]{
        // encoder 4
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 384), Eigen::MatrixXf(768, 384)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 384), Eigen::MatrixXf(768, 384)},
            },
        },
        // encoder 5
        {
            //  dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 768), Eigen::MatrixXf(1536, 768)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 768), Eigen::MatrixXf(1536, 768)},
            },
        }};

    Eigen::MatrixXf encoder_4_5_dconv_layers_3_lstm_ih_b[2][2][2][2]{
        // encoder 4
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
            },
        },
        // encoder 5
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
            },
        },
    };

    Eigen::MatrixXf encoder_4_5_dconv_layers_3_lstm_hh_w[2][2][2][2]{
        // encoder 4
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 192), Eigen::MatrixXf(768, 192)},
            },
        },
        // encoder 5
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 384), Eigen::MatrixXf(1536, 384)},
            },
        },
    };

    Eigen::MatrixXf encoder_4_5_dconv_layers_3_lstm_hh_b[2][2][2][2]{
        // encoder 4
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(768, 1), Eigen::MatrixXf(768, 1)},
            },
        },
        // encoder 5
        {
            // dconv layer 0
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
            },
            // dconv layer 1
            {
                // ih_l0, ih_l0_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
                // ih_l1, ih_l1_reverse
                {Eigen::MatrixXf(1536, 1), Eigen::MatrixXf(1536, 1)},
            },
        },
    };

    Eigen::MatrixXf encoder_4_5_dconv_layers_3_linear_weight[2][2]{
        {Eigen::MatrixXf(192, 384), Eigen::MatrixXf(192, 384)},
        {Eigen::MatrixXf(384, 768), Eigen::MatrixXf(384, 768)}};

    Eigen::VectorXf encoder_4_5_dconv_layers_3_linear_bias[2][2]{
        {Eigen::VectorXf(192), Eigen::VectorXf(192)},
        {Eigen::VectorXf(384), Eigen::VectorXf(384)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_4_content_weight[2][2]{
        {Eigen::Tensor3dXf(192, 192, 1), Eigen::Tensor3dXf(192, 192, 1)},
        {Eigen::Tensor3dXf(384, 384, 1), Eigen::Tensor3dXf(384, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_4_content_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_4_query_weight[2][2]{
        {Eigen::Tensor3dXf(192, 192, 1), Eigen::Tensor3dXf(192, 192, 1)},
        {Eigen::Tensor3dXf(384, 384, 1), Eigen::Tensor3dXf(384, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_4_query_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_4_key_weight[2][2]{
        {Eigen::Tensor3dXf(192, 192, 1), Eigen::Tensor3dXf(192, 192, 1)},
        {Eigen::Tensor3dXf(384, 384, 1), Eigen::Tensor3dXf(384, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_4_key_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_4_query_decay_weight[2][2]{
        {Eigen::Tensor3dXf(16, 192, 1), Eigen::Tensor3dXf(16, 192, 1)},
        {Eigen::Tensor3dXf(16, 384, 1), Eigen::Tensor3dXf(16, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_4_query_decay_bias[2][2]{
        {Eigen::Tensor1dXf(16), Eigen::Tensor1dXf(16)},
        {Eigen::Tensor1dXf(16), Eigen::Tensor1dXf(16)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_4_proj_weight[2][2]{
        {Eigen::Tensor3dXf(192, 192, 1), Eigen::Tensor3dXf(192, 192, 1)},
        {Eigen::Tensor3dXf(384, 384, 1), Eigen::Tensor3dXf(384, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_4_proj_bias[2][2]{
        {Eigen::Tensor1dXf(192), Eigen::Tensor1dXf(192)},
        {Eigen::Tensor1dXf(384), Eigen::Tensor1dXf(384)}};

    Eigen::Tensor3dXf encoder_4_5_dconv_layers_5_conv1d_weight[2][2]{
        {Eigen::Tensor3dXf(1536, 192, 1), Eigen::Tensor3dXf(1536, 192, 1)},
        {Eigen::Tensor3dXf(3072, 384, 1), Eigen::Tensor3dXf(3072, 384, 1)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_5_conv1d_bias[2][2]{
        {Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(1536)},
        {Eigen::Tensor1dXf(3072), Eigen::Tensor1dXf(3072)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_6_groupnorm_weight[2][2]{
        {Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(1536)},
        {Eigen::Tensor1dXf(3072), Eigen::Tensor1dXf(3072)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_6_groupnorm_bias[2][2]{
        {Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(1536)},
        {Eigen::Tensor1dXf(3072), Eigen::Tensor1dXf(3072)}};

    Eigen::Tensor1dXf encoder_4_5_dconv_layers_8_scale[2][2]{
        {Eigen::Tensor1dXf(768), Eigen::Tensor1dXf(768)},
        {Eigen::Tensor1dXf(1536), Eigen::Tensor1dXf(1536)}};

    // now we need 8 decoders that have a simple similar structure
    // conv_tr (weight + bias), rewrite (weight + bias)
    // first array dim is [2] for freq, time
    // next is layer for (2,3,4,5) for freq, (1,2,3,4) for time
    // Reshaped struct arrays to [2][4]
    Eigen::Tensor4dXf freq_decoders_conv_tr_weight[4]{
        Eigen::Tensor4dXf(384, 192, 8, 1), // decoder.2.conv_tr.weight
        Eigen::Tensor4dXf(192, 96, 8, 1),  // decoder.3.conv_tr.weight
        Eigen::Tensor4dXf(96, 48, 8, 1),   // decoder.4.conv_tr.weight
        Eigen::Tensor4dXf(48, 16, 8, 1)    // decoder.5.conv_tr.weight
    };

    Eigen::Tensor3dXf time_decoders_conv_tr_weight[4]{
        Eigen::Tensor3dXf(384, 192, 8), // tdecoder.1.conv_tr.weight
        Eigen::Tensor3dXf(192, 96, 8),  // tdecoder.2.conv_tr.weight
        Eigen::Tensor3dXf(96, 48, 8),   // tdecoder.3.conv_tr.weight
        Eigen::Tensor3dXf(48, 8, 8)     // tdecoder.4.conv_tr.weight
    };

    Eigen::Tensor1dXf decoders_conv_tr_bias[2][4]{
        {
            Eigen::Tensor1dXf(192), // decoder.2.conv_tr.bias
            Eigen::Tensor1dXf(96),  // decoder.3.conv_tr.bias
            Eigen::Tensor1dXf(48),  // decoder.4.conv_tr.bias
            Eigen::Tensor1dXf(16)   // decoder.5.conv_tr.bias
        },
        {
            Eigen::Tensor1dXf(192), // tdecoder.1.conv_tr.bias
            Eigen::Tensor1dXf(96),  // tdecoder.2.conv_tr.bias
            Eigen::Tensor1dXf(48),  // tdecoder.3.conv_tr.bias
            Eigen::Tensor1dXf(8)    // tdecoder.4.conv_tr.bias
        }
    };

    Eigen::Tensor4dXf freq_decoders_rewrite_weight[4]{
        Eigen::Tensor4dXf(768, 384, 3, 3), // decoder.2.rewrite.weight
        Eigen::Tensor4dXf(384, 192, 3, 3), // decoder.3.rewrite.weight
        Eigen::Tensor4dXf(192, 96, 3, 3),  // decoder.4.rewrite.weight
        Eigen::Tensor4dXf(96, 48, 3, 3)    // decoder.5.rewrite.weight
    };

    Eigen::Tensor3dXf time_decoders_rewrite_weight[4]{
        Eigen::Tensor3dXf(768, 384, 3), // tdecoder.1.rewrite.weight
        Eigen::Tensor3dXf(384, 192, 3), // tdecoder.2.rewrite.weight
        Eigen::Tensor3dXf(192, 96, 3),  // tdecoder.3.rewrite.weight
        Eigen::Tensor3dXf(96, 48, 3)    // tdecoder.4.rewrite.weight
    };

    Eigen::Tensor1dXf decoders_rewrite_bias[2][4]{
        {
            Eigen::Tensor1dXf(768), // decoder.2.rewrite.bias
            Eigen::Tensor1dXf(384), // decoder.3.rewrite.bias
            Eigen::Tensor1dXf(192), // decoder.4.rewrite.bias
            Eigen::Tensor1dXf(96)   // decoder.5.rewrite.bias
        },
        {
            Eigen::Tensor1dXf(768), // tdecoder.1.rewrite.bias
            Eigen::Tensor1dXf(384), // tdecoder.2.rewrite.bias
            Eigen::Tensor1dXf(192), // tdecoder.3.rewrite.bias
            Eigen::Tensor1dXf(96)   // tdecoder.4.rewrite.bias
        }
    };

    // Frequency Decoders
    Eigen::Tensor3dXf decoder_0_conv_tr_weight{
        Eigen::Tensor3dXf(1536, 768, 4)}; // decoder.0.conv_tr.weight


    Eigen::Tensor4dXf decoder_1_conv_tr_weight{
        Eigen::Tensor4dXf(768, 384, 8, 1)   // decoder.1.conv_tr.weight
    };

    Eigen::Tensor1dXf decoder_0_1_conv_tr_bias[2]{
        Eigen::Tensor1dXf(768), // decoder.0.conv_tr.bias
        Eigen::Tensor1dXf(384)  // decoder.1.conv_tr.bias
    };

    Eigen::Tensor1dXf decoder_0_1_norm2_weight[2]{
        Eigen::Tensor1dXf(768), // decoder.0.norm2.weight
        Eigen::Tensor1dXf(384)  // decoder.1.norm2.weight
    };

    Eigen::Tensor1dXf decoder_0_1_norm2_bias[2]{
        Eigen::Tensor1dXf(768), // decoder.0.norm2.bias
        Eigen::Tensor1dXf(384)  // decoder.1.norm2.bias
    };

    Eigen::Tensor3dXf decoder_0_rewrite_weight{
        Eigen::Tensor3dXf(3072, 1536, 3)};

    Eigen::Tensor4dXf decoder_1_rewrite_weight{
        Eigen::Tensor4dXf(1536, 768, 3, 3)};

    Eigen::Tensor1dXf decoder_0_1_rewrite_bias[2]{
        Eigen::Tensor1dXf(3072), // decoder.0.rewrite.bias
        Eigen::Tensor1dXf(1536)  // decoder.1.rewrite.bias
    };

    Eigen::Tensor1dXf decoder_0_1_norm1_weight[2]{
        Eigen::Tensor1dXf(3072), // decoder.0.norm1.weight
        Eigen::Tensor1dXf(1536)  // decoder.1.norm1.weight
    };

    Eigen::Tensor1dXf decoder_0_1_norm1_bias[2]{
        Eigen::Tensor1dXf(3072), // decoder.0.norm1.bias
        Eigen::Tensor1dXf(1536)  // decoder.1.norm1.bias
    };

    // Unique tdecoder 0
    Eigen::Tensor3dXf tdecoder_0_conv_tr_weight{Eigen::Tensor3dXf(768, 384, 8)}; // tdecoder.0.conv_tr.weight
    Eigen::Tensor1dXf tdecoder_0_conv_tr_bias{Eigen::Tensor1dXf(384)};           // tdecoder.0.conv_tr.bias
    Eigen::Tensor1dXf tdecoder_0_norm2_weight{Eigen::Tensor1dXf(384)};           // tdecoder.0.norm2.weight
    Eigen::Tensor1dXf tdecoder_0_norm2_bias{Eigen::Tensor1dXf(384)};             // tdecoder.0.norm2.bias

    // freq_emb
    Eigen::MatrixXf freq_emb_embedding_weight{Eigen::MatrixXf(512, 48)};
};

struct demucs_v3_segment_buffers
{
    int segment_samples;
    int le;
    int pad;
    int pad_end;
    int padded_segment_samples;
    int nb_stft_frames;
    int nb_stft_bins;

    Eigen::MatrixXf mix;
    Eigen::Tensor3dXf targets_out;
    Eigen::MatrixXf padded_mix;
    Eigen::Tensor3dXcf z;

    // freq branch, one for each encoded representation
    Eigen::Tensor3dXf x;     // input
    Eigen::Tensor3dXf x_out; // input
    Eigen::Tensor3dXf x_0;
    Eigen::Tensor3dXf x_1;
    Eigen::Tensor3dXf x_2;
    Eigen::Tensor3dXf x_3;
    Eigen::Tensor3dXf x_4;

    // shared after encoder 5
    Eigen::Tensor3dXf x_shared_5;

    // time branch
    Eigen::Tensor3dXf xt;             // input
    Eigen::Tensor3dXf xt_out;         // output
    Eigen::Tensor3dXf xt_decoded_out; // hold time decoder output
    Eigen::Tensor3dXf xt_0;
    Eigen::Tensor3dXf xt_1;
    Eigen::Tensor3dXf xt_2;
    Eigen::Tensor3dXf xt_3;
    Eigen::Tensor3dXf xt_4;

    // empty tensors to hold decoded output
    // in conjunction with skip connections
    Eigen::Tensor3dXf x_decode;
    Eigen::Tensor3dXf xt_decode;

    // empty skip conn for encoder 5
    Eigen::Tensor3dXf x_shared_5_empty_skip;

    // skip conns for frequency and time
    // easier as hardcoded matrix sizes
    Eigen::Tensor3dXf saved_0;
    Eigen::Tensor3dXf saved_1;
    Eigen::Tensor3dXf saved_2;
    Eigen::Tensor3dXf saved_3;
    Eigen::Tensor3dXf saved_4;

    Eigen::Tensor3dXf savedt_0;
    Eigen::Tensor3dXf savedt_1;
    Eigen::Tensor3dXf savedt_2;
    Eigen::Tensor3dXf savedt_3;
    Eigen::Tensor3dXf savedt_4;

    // LSTM data
    // 2 encoders, 2 dconv layers, 2 layers, 2 directions
    // per-direction buffers
    Eigen::MatrixXf lstm_output_per_direction[2][2][2][2];
    Eigen::MatrixXf lstm_hidden[2][2][2][2];
    Eigen::MatrixXf lstm_cell[2][2][2][2];
    // out-of-direction buffers
    Eigen::MatrixXf lstm_output[2][2][2];

    // LocalAttention structs
    Eigen::VectorXi local_attn_index;
    Eigen::MatrixXi local_attn_delta;
    Eigen::Tensor1dXf local_attn_decays;

    Eigen::Tensor2dXf local_attn_decay_kernel;

    // constructor for demucs_segment_buffers that takes int parameters

    // let's do pesky precomputing of the signal repadding to 1/4 hop
    // for time and frequency alignment
    demucs_v3_segment_buffers(int nb_channels, int segment_samples,
                              int nb_sources)
        : segment_samples(segment_samples),
          le(int(std::ceil((float)segment_samples /
                           (float)demucscpp::FFT_HOP_SIZE))),
          pad(std::floor((float)demucscpp::FFT_HOP_SIZE / 2.0f) * 3),
          pad_end(pad + le * demucscpp::FFT_HOP_SIZE - segment_samples),
          padded_segment_samples(segment_samples + pad + pad_end),
          nb_stft_frames(segment_samples / demucscpp::FFT_HOP_SIZE + 1),
          nb_stft_bins(demucscpp::FFT_WINDOW_SIZE / 2 + 1),
          mix(nb_channels, segment_samples),
          targets_out(nb_sources, nb_channels, segment_samples),
          padded_mix(nb_channels, padded_segment_samples),
          z(nb_channels, nb_stft_bins, nb_stft_frames),
          // complex-as-channels implies 2*nb_channels for real+imag
          x(2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          x_out(nb_sources * 2 * nb_channels, nb_stft_bins - 1, nb_stft_frames),
          x_0(48, 512, FREQ_BRANCH_LEN), x_1(96, 128, FREQ_BRANCH_LEN),
          x_2(192, 32, FREQ_BRANCH_LEN), x_3(384, 8, FREQ_BRANCH_LEN),
          x_4(768, 1, FREQ_BRANCH_LEN),
          x_shared_5(1, 1536, SHARED_BRANCH_LEN), // merged freq and time
          xt(1, nb_channels, segment_samples),
          xt_out(1, nb_sources * nb_channels, segment_samples),
          xt_decoded_out(1, 8, segment_samples), xt_0(1, 48, TIME_BRANCH_LEN_0),
          xt_1(1, 96, TIME_BRANCH_LEN_1), xt_2(1, 192, TIME_BRANCH_LEN_2),
          xt_3(1, 384, TIME_BRANCH_LEN_3), xt_4(1, 768, TIME_BRANCH_LEN_4),
          x_decode(1536, 1, SHARED_BRANCH_LEN),
          xt_decode(768, 1, FREQ_BRANCH_LEN),
          saved_0(48, 512, FREQ_BRANCH_LEN), saved_1(96, 128, FREQ_BRANCH_LEN),
          saved_2(192, 32, FREQ_BRANCH_LEN), saved_3(384, 8, FREQ_BRANCH_LEN),
          saved_4(768, 1, FREQ_BRANCH_LEN),
          savedt_0(1, 48, TIME_BRANCH_LEN_0),
          savedt_1(1, 96, TIME_BRANCH_LEN_1),
          savedt_2(1, 192, TIME_BRANCH_LEN_2),
          savedt_3(1, 384, TIME_BRANCH_LEN_3),
          savedt_4(1, 768, TIME_BRANCH_LEN_4),
          local_attn_index(FREQ_BRANCH_LEN),
          local_attn_delta(FREQ_BRANCH_LEN, FREQ_BRANCH_LEN),
          local_attn_decays(LOCAL_ATTN_N_DECAY),
          local_attn_decay_kernel(LOCAL_ATTN_N_DECAY, FREQ_BRANCH_LEN) {
            // initialize lstm buffers
            int hidden_size = -1;
            int cell_size = -1;
            int lstm_seq_len = -1;

            // encoder layer
            for (int i = 0; i < 2; i++) {
                if (i == 0) {
                    hidden_size = LSTM_HIDDEN_SIZE_0;
                    cell_size = LSTM_HIDDEN_SIZE_0;
                    lstm_seq_len = FREQ_BRANCH_LEN;
                } else {
                    hidden_size = LSTM_HIDDEN_SIZE_1;
                    cell_size = LSTM_HIDDEN_SIZE_1;
                    lstm_seq_len = SHARED_BRANCH_LEN;
                }

                // dconv layer
                for (int j = 0; j < 2; j++) {
                    // lstm layer
                    for (int k = 0; k < 2; k++) {
                        // lstm direction
                        for (int l = 0; l < 2; l++) {
                            lstm_output_per_direction[i][j][k][l] = Eigen::MatrixXf::Zero(lstm_seq_len, hidden_size);
                            lstm_hidden[i][j][k][l] = Eigen::MatrixXf::Zero(hidden_size, 1);
                            lstm_cell[i][j][k][l] = Eigen::MatrixXf::Zero(cell_size, 1);
                        }

                        lstm_output[i][j][k] = Eigen::MatrixXf::Zero(lstm_seq_len, 2 * hidden_size);
                    }
                }
            }
            // initialize local attn stuff
            for (int i = 0; i < FREQ_BRANCH_LEN; ++i) {
                local_attn_index(i) = i;
            }

            // delta = indexes[:, None] - indexes[None, :]
            for (int i = 0; i < FREQ_BRANCH_LEN; ++i) {
                for (int j = 0; j < FREQ_BRANCH_LEN; ++j) {
                    local_attn_delta(i, j) = local_attn_index(i) - local_attn_index(j);
                }
            }

            // Decay levels from 1 to ndecay
            for (int i = 0; i < LOCAL_ATTN_N_DECAY; ++i) {
                local_attn_decays(i) = i + 1;
            }

            for (int d = 0; d < LOCAL_ATTN_N_DECAY; ++d) {
                for (int t = 0; t < FREQ_BRANCH_LEN; ++t) {
                    local_attn_decay_kernel(d, t) = -local_attn_decays(d) * std::abs(local_attn_delta(0, t)) / std::sqrt(LOCAL_ATTN_N_DECAY);
                }
            }
        };
};

bool load_demucs_v3_model(const std::string &model_dir,
                          struct demucs_v3_model *model);

const float SEGMENT_LEN_SECS = 7.8;      // 8 seconds, the demucs chunk size
const float SEGMENT_OVERLAP_SECS = 0.25; // 0.25 overlap
const float MAX_SHIFT_SECS = 0.5;        // max shift
const float OVERLAP = 0.25;              // overlap between segments
const float TRANSITION_POWER = 1.0;      // transition between segments

Eigen::Tensor3dXf
demucs_v3_inference(const struct demucscpp_v3::demucs_v3_model &model,
                    const Eigen::MatrixXf &full_audio,
                    demucscpp::ProgressCallback cb);

void model_v3_inference(const struct demucs_v3_model &model,
                        struct demucscpp_v3::demucs_v3_segment_buffers &buffers,
                        struct demucscpp::stft_buffers &stft_buf,
                        demucscpp::ProgressCallback cb, float current_progress,
                        float segment_progress);
} // namespace demucscpp_v3

#endif // MODEL_HPP
