#include "model.hpp"
#include <Eigen/Dense>
#include <cstdarg>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void my_fprintf(const std::FILE *stream, const char *format, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    if (stream == stderr)
    {
        std::cerr << buffer;
    }
    else
    {
        std::cout << buffer;
    }
}

// forward declaration
static size_t load_single_tensor1d(FILE *f, std::string &name,
                                   Eigen::Tensor1dXf &matrix, int *ne,
                                   int32_t nelements);

static size_t load_single_vector(FILE *f, std::string &name,
                                 Eigen::VectorXf &matrix, int *ne,
                                 int32_t nelements);

static size_t load_single_matrix(FILE *f, std::string &name,
                                 Eigen::MatrixXf &matrix, int *ne,
                                 int32_t nelements);

static size_t load_single_tensor2d(FILE *f, std::string &name,
                                   Eigen::Tensor2dXf &tensor, int *ne,
                                   int32_t nelements);

static size_t load_single_tensor3d(FILE *f, std::string &name,
                                   Eigen::Tensor3dXf &tensor, int *ne,
                                   int32_t nelements);

static size_t load_single_tensor4d(FILE *f, std::string &name,
                                   Eigen::Tensor4dXf &tensor, int *ne,
                                   int32_t nelements);

// from scripts/convert-pth-to-ggml.py
bool demucscpp::load_demucs_model(const std::string &model_file,
                                  struct demucs_model *model)
{
    my_fprintf(stderr, "%s: loading model\n", __func__);

    // compute t_start_us using C++ std::chrono
    const auto t_start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    std::cout << "Loading model_file... " << std::endl;

    FILE *f = fopen(model_file.c_str(), "rb");
    if (!f)
    {
        my_fprintf(stderr, "%s: failed to open %s\n", __func__,
                   model_file.c_str());
        return false;
    }

    // verify magic
    uint32_t magic;

    std::cout << "Checking the magic of model_file" << std::endl;

    // read the size of uint32_t bytes from f into magic
    fread(&magic, sizeof(uint32_t), 1, f);

    if (magic == 0x646d6336) // dmc6
    {
        model->is_4sources = false;
        std::cout << "Model magic is Demucs 6-source" << std::endl;

        // modify a few tensor shapes in the model corresponding to the
        // number of sources
        model->decoder_conv_tr_weight[3] = Eigen::Tensor4dXf(48, 24, 8, 1);
        model->decoder_conv_tr_bias[3] = Eigen::Tensor1dXf(24);

        model->tdecoder_conv_tr_weight[3] = Eigen::Tensor3dXf(48, 12, 8);
        model->tdecoder_conv_tr_bias[3] = Eigen::Tensor1dXf(12);
    }
    else if (magic == 0x646d6334) // dmc4
    {
        model->is_4sources = true;
        std::cout << "Model magic is Demucs 4-source" << std::endl;
    }
    else
    {
        fprintf(stderr, "%s: invalid model data (bad magic)\n", __func__);
        fclose(f);
        return false;
    }

    model->crosstransformer =
        demucscpp::initialize_crosstransformer(model->is_4sources);

    std::cout << "Loading demucs model... " << std::endl;

    // we dont need to prepare memory for the weights
    // they come preallocated in the hardcoded model

    size_t total_size = 0;
    uint32_t n_loaded = 0;

    // equivalent of with open(...) as f on each model_file
    std::cout << "Loading weights from model_file" << std::endl;

    // load weights from the file one tensor at a time

    for (;;)
    {
        int32_t n_dims;
        int32_t length;

        fread(&n_dims, sizeof(int32_t), 1, f);
        fread(&length, sizeof(int32_t), 1, f);

        int32_t nelements = 1;

        // we are loading up to 4d tensors, so allocate 4 dims
        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i)
        {
            fread(&ne[i], sizeof(int32_t), 1, f);
            nelements *= ne[i];
        }

        std::string name;
        std::vector<char> tmp(length);               // create a buffer
        fread(&tmp[0], sizeof(char), tmp.size(), f); // read to buffer
        name.assign(&tmp[0], tmp.size());

        // check if we reached eof of the open file f
        if (feof(f))
        {
            break;
        }

        // std::cout << "Loading tensor " << name << " with shape [" << ne[0]
        //             << ", " << ne[1] << ", " << ne[2] << ", " << ne[3] << "]"
        //             << std::endl;

        // match the tensor name to the correct tensor in the model
        size_t loaded_size = 0;

        // 4 Encoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "encoder." + std::to_string(i) + ".conv.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->encoder_conv_weight[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".conv.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_conv_bias[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->encoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                if (name == "encoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_0_conv1d_bias[0][0][i][j],
                        ne, nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_bias[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_3_conv1d_bias[0][0][i][j],
                        ne, nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_bias[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_6_scale[0][0][i][j], ne,
                        nelements);
                }
            }
        }

        // 4 Decoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "decoder." + std::to_string(i) + ".conv_tr.weight")
            {
                loaded_size = load_single_tensor4d(
                    f, name, model->decoder_conv_tr_weight[i], ne, nelements);
            }
            else if (name == "decoder." + std::to_string(i) + ".conv_tr.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_conv_tr_bias[i], ne, nelements);
            }
            else if (name == "decoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor4d(
                    f, name, model->decoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "decoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                int reverse_i = 4 - i - 1;
                if (name == "decoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_0_conv1d_bias[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[0][1][reverse_i]
                                                              [j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model
                            ->dconv_layers_1_groupnorm_bias[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_3_conv1d_bias[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[0][1][reverse_i]
                                                              [j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model
                            ->dconv_layers_4_groupnorm_bias[0][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "decoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_6_scale[0][1][reverse_i][j], ne,
                        nelements);
                }
            }
        }

        // 4 TEncoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "tencoder." + std::to_string(i) + ".conv.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tencoder_conv_weight[i], ne, nelements);
            }
            else if (name == "tencoder." + std::to_string(i) + ".conv.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tencoder_conv_bias[i], ne, nelements);
            }
            else if (name ==
                     "tencoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tencoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "tencoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tencoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                if (name == "tencoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_0_conv1d_bias[1][0][i][j],
                        ne, nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_bias[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_3_conv1d_bias[1][0][i][j],
                        ne, nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_bias[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_6_scale[1][0][i][j], ne,
                        nelements);
                }
            }
        }

        // 4 TDecoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "tdecoder." + std::to_string(i) + ".conv_tr.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tdecoder_conv_tr_weight[i], ne, nelements);
            }
            else if (name == "tdecoder." + std::to_string(i) + ".conv_tr.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tdecoder_conv_tr_bias[i], ne, nelements);
            }
            else if (name ==
                     "tdecoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tdecoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "tdecoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tdecoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                int reverse_i = 4 - i - 1;
                if (name == "tdecoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_0_conv1d_bias[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[1][1][reverse_i]
                                                              [j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model
                            ->dconv_layers_1_groupnorm_bias[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_3_conv1d_bias[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[1][1][reverse_i]
                                                              [j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model
                            ->dconv_layers_4_groupnorm_bias[1][1][reverse_i][j],
                        ne, nelements);
                }
                else if (name == "tdecoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_6_scale[1][1][reverse_i][j], ne,
                        nelements);
                }
            }
        }

        if (name == "freq_emb.embedding.weight")
        {
            loaded_size = load_single_matrix(
                f, name, model->freq_emb_embedding_weight, ne, nelements);
        }
        else if ((name == "channel_upsampler.weight") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor3d(
                f, name, ct_4s->channel_upsampler_weight, ne, nelements);
        }
        else if ((name == "channel_upsampler.bias") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor1d(
                f, name, ct_4s->channel_upsampler_bias, ne, nelements);
        }
        else if ((name == "channel_downsampler.weight") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor3d(
                f, name, ct_4s->channel_downsampler_weight, ne, nelements);
        }
        else if ((name == "channel_downsampler.bias") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor1d(
                f, name, ct_4s->channel_downsampler_bias, ne, nelements);
        }
        else if ((name == "channel_upsampler_t.weight") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor3d(
                f, name, ct_4s->channel_upsampler_t_weight, ne, nelements);
        }
        else if ((name == "channel_upsampler_t.bias") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor1d(
                f, name, ct_4s->channel_upsampler_t_bias, ne, nelements);
        }
        else if ((name == "channel_downsampler_t.weight") &&
                 (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor3d(
                f, name, ct_4s->channel_downsampler_t_weight, ne, nelements);
        }
        else if ((name == "channel_downsampler_t.bias") && (model->is_4sources))
        {
            auto *ct_4s = static_cast<demucs_crosstransformer_4s *>(
                model->crosstransformer.get());
            loaded_size = load_single_tensor1d(
                f, name, ct_4s->channel_downsampler_t_bias, ne, nelements);
        }
        else if (name == "crosstransformer.norm_in.weight")
        {
            loaded_size = load_single_tensor1d(
                f, name,
                model->crosstransformer->crosstransformer_norm_in_weight, ne,
                nelements);
        }
        else if (name == "crosstransformer.norm_in.bias")
        {
            loaded_size = load_single_tensor1d(
                f, name, model->crosstransformer->crosstransformer_norm_in_bias,
                ne, nelements);
        }
        else if (name == "crosstransformer.norm_in_t.weight")
        {
            loaded_size = load_single_tensor1d(
                f, name,
                model->crosstransformer->crosstransformer_norm_in_t_weight, ne,
                nelements);
        }
        else if (name == "crosstransformer.norm_in_t.bias")
        {
            loaded_size = load_single_tensor1d(
                f, name,
                model->crosstransformer->crosstransformer_norm_in_t_bias, ne,
                nelements);
        }

        // 5 crosstransformer layers, * 2 for time and frequency
        for (int transformer_layer = 0; transformer_layer < 5;
             ++transformer_layer)
        {
            for (int freq_or_time = 0; freq_or_time < 2; ++freq_or_time)
            {
                std::string suffix = "";
                if (freq_or_time == 1)
                {
                    suffix = "_t";
                }
                suffix += "." + std::to_string(transformer_layer);

                // even indexes are self_attn, odd are cross_attn
                if (transformer_layer % 2 == 0)
                {
                    // even case, 0,2,4 divided by 2 will lead to indexes
                    // 0,1,2 in the Eigen C++ struct member
                    int layer_index = transformer_layer / 2;

                    if (name == "crosstransformer.layers" + suffix +
                                    ".self_attn.in_proj_weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_self_attn_in_proj_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".self_attn.in_proj_bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_self_attn_in_proj_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".self_attn.out_proj.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_self_attn_out_proj_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".self_attn.out_proj.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_self_attn_out_proj_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }

                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear1.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_linear1_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear1.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_linear1_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear2.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_linear2_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear2.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_linear2_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm1.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm1_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name ==
                             "crosstransformer.layers" + suffix + ".norm1.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm1_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm2.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm2_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name ==
                             "crosstransformer.layers" + suffix + ".norm2.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm2_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm_out.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm_out_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm_out.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_norm_out_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".gamma_1.scale")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_gamma_1_scale
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".gamma_2.scale")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_my_layers_gamma_2_scale
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                }

                // even indexes are self_attn, odd are cross_attn
                else if (transformer_layer % 2 == 1)
                {
                    // odd case, ({1,3}-1)/2 maps to 0,1 in the Eigen struct
                    int layer_index = (transformer_layer - 1) / 2;

                    if (name == "crosstransformer.layers" + suffix +
                                    ".cross_attn.in_proj_weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_cross_attn_in_proj_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".cross_attn.in_proj_bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_cross_attn_in_proj_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".cross_attn.out_proj.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_cross_attn_out_proj_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".cross_attn.out_proj.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_cross_attn_out_proj_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear1.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_linear1_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear1.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_linear1_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear2.weight")
                    {
                        loaded_size = load_single_matrix(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_linear2_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".linear2.bias")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_linear2_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm1.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm1_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name ==
                             "crosstransformer.layers" + suffix + ".norm1.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm1_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm2.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm2_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name ==
                             "crosstransformer.layers" + suffix + ".norm2.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm2_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm3.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm3_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name ==
                             "crosstransformer.layers" + suffix + ".norm3.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm3_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm_out.weight")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm_out_weight
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".norm_out.bias")
                    {
                        loaded_size = load_single_tensor1d(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_norm_out_bias
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".gamma_1.scale")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_gamma_1_scale
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                    else if (name == "crosstransformer.layers" + suffix +
                                         ".gamma_2.scale")
                    {
                        loaded_size = load_single_vector(
                            f, name,
                            model->crosstransformer
                                ->crosstransformer_cross_layers_gamma_2_scale
                                    [freq_or_time][layer_index],
                            ne, nelements);
                    }
                }
            }
        }

        if (loaded_size == 0)
        {
            my_fprintf(stderr, "%s: failed to load %s\n", __func__,
                       name.c_str());
            return false;
        }
        total_size += loaded_size;
        n_loaded++;
    }

    fclose(f);

    // compute finish time in microseconds using std::chrono

    const auto t_end_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    // print load time in seconds
    my_fprintf(stdout, "Loaded model (%u tensors, %6.2f MB) in %f s\n",
               n_loaded, total_size / 1024.0 / 1024.0,
               (float)(t_end_us - t_start_us) / 1000000.0f);

    return true;
}

static size_t load_single_matrix(FILE *f, std::string &name,
                                 Eigen::MatrixXf &matrix, int *ne,
                                 int32_t nelements)
{
    if (matrix.size() != nelements)
    {
        my_fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(
            stderr,
            "%s: model file shape: [%d, %d], demucs.cpp shape: [%d, %d]\n",
            __func__, ne[0], ne[1], (int)matrix.rows(), (int)matrix.cols());
        return 0;
    }

    // loading quantized weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_tensor = matrix.size() * bpe_half;

    // create a Eigen::half Eigen::Matrix to hold the quantized weights
    // of the same shape as the float matrix
    Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        matrix_half =
            Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>::Zero(matrix.rows(), matrix.cols());

    fread(matrix_half.data(), bpe_half, nelements, f);

    // my_fprintf(stdout, "%16s: [%5d, %5d], type = float, %6.2f MB\n",
    // name.data(), ne[0],
    //        ne[1], nbytes_tensor / 1024.0 / 1024.0);

    // and copy them into the float matrix
    for (int i = 0; i < ne[0]; i++)
    {
        for (int j = 0; j < ne[1]; j++)
        {
            matrix(i, j) = static_cast<float>(matrix_half(i, j));
        }
    }

    return nbytes_tensor;
}

static size_t load_single_tensor2d(FILE *f, std::string &name,
                                   Eigen::Tensor2dXf &tensor, int *ne,
                                   int32_t nelements)
{
    if (tensor.size() != nelements)
    {
        my_fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(stderr,
                   "%s: model file shape: [%d, %d], demucs.cpp shape: [%d, "
                   "%d]\n",
                   __func__, ne[0], ne[1], (int)tensor.dimension(0),
                   (int)tensor.dimension(1));
        return 0;
    }

    // loading weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_tensor = tensor.size() * bpe_half;

    // create a Eigen::half Eigen::Matrix to hold the quantized weights
    // of the same shape as the float matrix
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor> tensor_half(ne[0], ne[1]);
    fread(tensor_half.data(), bpe_half, nelements, f);

    // Uncomment to print tensor info
    // my_fprintf(stdout, "%16s: [%5d, %5d], type = float, %6.2f MB\n",
    //            name.data(), ne[0], ne[1], nbytes_tensor / 1024.0 / 1024.0);

    // Manually copy the data from tensor_half to tensor
    for (int i = 0; i < ne[0]; ++i)
    {
        for (int j = 0; j < ne[1]; ++j)
        {
            tensor(i, j) = static_cast<float>(tensor_half(i, j));
        }
    }

    return nbytes_tensor;
}

static size_t load_single_tensor3d(FILE *f, std::string &name,
                                   Eigen::Tensor3dXf &tensor, int *ne,
                                   int32_t nelements)
{
    if (tensor.size() != nelements)
    {
        my_fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(stderr,
                   "%s: model file shape: [%d, %d, %d], demucs.cpp shape: [%d, "
                   "%d, %d]\n",
                   __func__, ne[0], ne[1], ne[2], (int)tensor.dimension(0),
                   (int)tensor.dimension(1), (int)tensor.dimension(2));
        return 0;
    }

    // loading weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_tensor = tensor.size() * bpe_half;

    // create a Eigen::half Eigen::Matrix to hold the quantized weights
    // of the same shape as the float matrix
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor> tensor_half(ne[0], ne[1],
                                                               ne[2]);
    fread(tensor_half.data(), bpe_half, nelements, f);

    // my_fprintf(stdout, "%16s: [%5d, %5d, %5d], type = float, %6.2f MB\n",
    // name.data(),
    //        ne[0], ne[1], ne[2], nbytes_tensor / 1024.0 / 1024.0);

    // Manually copy the data from tensor_half to tensor
    for (int i = 0; i < ne[0]; ++i)
    {
        for (int j = 0; j < ne[1]; ++j)
        {
            for (int k = 0; k < ne[2]; ++k)
            {
                tensor(i, j, k) = static_cast<float>(tensor_half(i, j, k));
            }
        }
    }

    return nbytes_tensor;
}

static size_t load_single_tensor4d(FILE *f, std::string &name,
                                   Eigen::Tensor4dXf &tensor, int *ne,
                                   int32_t nelements)
{
    if (tensor.size() != nelements)
    {
        my_fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(
            stderr,
            "%s: model file shape: [%d, %d, %d, %d], demucs.cpp shape: [%d, "
            "%d, %d, %d]\n",
            __func__, ne[0], ne[1], ne[2], ne[3], (int)tensor.dimension(0),
            (int)tensor.dimension(1), (int)tensor.dimension(2),
            (int)tensor.dimension(3));
        return 0;
    }

    // loading weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_tensor = tensor.size() * bpe_half;

    // create a Eigen::half Eigen::Tensor to hold the quantized weights
    // of the same shape as the float tensor
    Eigen::Tensor<Eigen::half, 4, Eigen::RowMajor> tensor_half(ne[0], ne[1],
                                                               ne[2], ne[3]);
    fread(tensor_half.data(), bpe_half, nelements, f);

    // my_fprintf(stdout, "%16s: [%5d, %5d, %5d, %5d], type = float, %6.2f
    // MB\n", name.data(),
    //        ne[0], ne[1], ne[2], ne[3], nbytes_tensor / 1024.0 / 1024.0);

    // Manually copy the data from tensor_half to tensor
    for (int i = 0; i < ne[0]; ++i)
    {
        for (int j = 0; j < ne[1]; ++j)
        {
            for (int k = 0; k < ne[2]; ++k)
            {
                for (int l = 0; l < ne[3]; ++l)
                {
                    tensor(i, j, k, l) =
                        static_cast<float>(tensor_half(i, j, k, l));
                }
            }
        }
    }

    return nbytes_tensor;
}

static size_t load_single_tensor1d(FILE *f, std::string &name,
                                   Eigen::Tensor1dXf &tensor, int *ne,
                                   int32_t nelements)
{
    if (tensor.size() != nelements)
    {
        my_fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(stderr,
                   "%s: model file shape: [%d], demucs.cpp shape: [%d]\n",
                   __func__, ne[0], (int)tensor.dimension(0));
        return 0;
    }

    // loading weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_tensor = tensor.size() * bpe_half;

    // create a Eigen::half Eigen::Tensor to hold the quantized weights
    // of the same shape as the float tensor
    Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor> tensor_half(ne[0]);
    fread(tensor_half.data(), bpe_half, nelements, f);

    // my_fprintf(stdout, "%16s: [%5d], type = float, %6.2f MB\n", name.data(),
    //        ne[0], nbytes_tensor / 1024.0 / 1024.0);

    // Manually copy the data from tensor_half to tensor
    for (int i = 0; i < ne[0]; ++i)
    {
        tensor(i) = static_cast<float>(tensor_half(i));
    }

    return nbytes_tensor;
}

static size_t load_single_vector(FILE *f, std::string &name,
                                 Eigen::VectorXf &vector, int *ne,
                                 int32_t nelements)
{
    if (vector.size() != nelements)
    {
        my_fprintf(stderr, "%s: vector '%s' has wrong size in model file\n",
                   __func__, name.data());
        my_fprintf(stderr,
                   "%s: model file shape: [%d], demucs.cpp shape: [%d]\n",
                   __func__, ne[0], (int)vector.size());
        return 0;
    }

    // loading weights
    const size_t bpe_half = sizeof(Eigen::half);
    auto nbytes_vector = vector.size() * bpe_half;

    // create a Eigen::half Eigen::Vector to hold the quantized weights
    // of the same shape as the float vector
    Eigen::Matrix<Eigen::half, Eigen::Dynamic, 1> vector_half(ne[0]);
    fread(vector_half.data(), bpe_half, nelements, f);

    // my_fprintf(stdout, "%16s: [%5d], type = float, %6.2f MB\n", name.data(),
    //        ne[0], nbytes_vector / 1024.0 / 1024.0);

    // Manually copy the data from vector_half to vector
    for (int i = 0; i < ne[0]; ++i)
    {
        vector(i) = static_cast<float>(vector_half(i));
    }

    return nbytes_vector;
}

bool demucscpp_v3::load_demucs_v3_model(const std::string &model_file,
                                  struct demucscpp_v3::demucs_v3_model *model)
{
    my_fprintf(stderr, "%s: loading model\n", __func__);

    // compute t_start_us using C++ std::chrono
    const auto t_start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    std::cout << "Loading model_file... " << std::endl;

    FILE *f = fopen(model_file.c_str(), "rb");
    if (!f)
    {
        my_fprintf(stderr, "%s: failed to open %s\n", __func__,
                   model_file.c_str());
        return false;
    }

    // verify magic
    uint32_t magic;

    std::cout << "Checking the magic of model_file" << std::endl;

    // read the size of uint32_t bytes from f into magic
    fread(&magic, sizeof(uint32_t), 1, f);

    if (magic != 0x646d6333) // dmc3 = v3 mmi
    {
        fprintf(stderr, "%s: invalid model data (bad magic)\n", __func__);
        fclose(f);
        return false;
    }

    std::cout << "Model magic is Demucs V3 MMI" << std::endl;

    std::cout << "Loading demucs model... " << std::endl;

    // we dont need to prepare memory for the weights
    // they come preallocated in the hardcoded model

    size_t total_size = 0;
    uint32_t n_loaded = 0;

    // equivalent of with open(...) as f on each model_file
    std::cout << "Loading weights from model_file" << std::endl;

    // load weights from the file one tensor at a time

    for (;;)
    {
        int32_t n_dims;
        int32_t length;

        fread(&n_dims, sizeof(int32_t), 1, f);
        fread(&length, sizeof(int32_t), 1, f);

        int32_t nelements = 1;

        // we are loading up to 4d tensors, so allocate 4 dims
        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i)
        {
            fread(&ne[i], sizeof(int32_t), 1, f);
            nelements *= ne[i];
        }

        std::string name;
        std::vector<char> tmp(length);               // create a buffer
        fread(&tmp[0], sizeof(char), tmp.size(), f); // read to buffer
        name.assign(&tmp[0], tmp.size());

        // check if we reached eof of the open file f
        if (feof(f))
        {
            break;
        }

        //std::cout << "Loading tensor " << name << " with shape [" << ne[0]
        //            << ", " << ne[1] << ", " << ne[2] << ", " << ne[3] << "]"
        //            << std::endl;

        // match the tensor name to the correct tensor in the model
        size_t loaded_size = 0;

        // 4 Encoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "encoder." + std::to_string(i) + ".conv.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->encoder_conv_weight[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".conv.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_conv_bias[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->encoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "encoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                if (name == "encoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_0_conv1d_bias[0][0][i][j],
                        ne, nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_bias[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_3_conv1d_bias[0][0][i][j],
                        ne, nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_bias[0][0][i][j], ne,
                        nelements);
                }
                else if (name == "encoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_6_scale[0][0][i][j], ne,
                        nelements);
                }
            }
        }

        // Loop over encoders 4 and 5 non-dconv layers
        // dconv will be treated specially in the next block
        for (int encoder_index = 4; encoder_index <= 5; ++encoder_index) {
            int array_index = encoder_index - 4; // Maps 4 to 0, 5 to 1

            if (name == "encoder." + std::to_string(encoder_index) + ".conv.weight") {
                if (encoder_index == 4) {
                    loaded_size = load_single_tensor4d(
                            f, name, model->encoder_4_conv_weight, ne, nelements);
                } else {
                    loaded_size = load_single_tensor3d(
                        f, name, model->encoder_5_conv_weight, ne, nelements);
                }
            } else if (name == "encoder." + std::to_string(encoder_index) + ".conv.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_conv_bias[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".norm1.weight") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_norm1_weight[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".norm1.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_norm1_bias[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".rewrite.weight") {
                loaded_size = load_single_tensor3d(
                    f, name, model->encoder_4_5_rewrite_weight[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".rewrite.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_rewrite_bias[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".norm2.weight") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_norm2_weight[array_index], ne, nelements);
            } else if (name == "encoder." + std::to_string(encoder_index) + ".norm2.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->encoder_4_5_norm2_bias[array_index], ne, nelements);
            }

            // dconv time: 2 per layer
            for (int dconv_index = 0; dconv_index < 2; ++dconv_index) {
                if (name == "encoder." + std::to_string(encoder_index) + ".dconv.layers." +
                                std::to_string(dconv_index) + ".0.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_0_conv1d_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".0.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name, model->encoder_4_5_dconv_layers_0_conv1d_bias[array_index][dconv_index], ne, nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".1.weight") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_1_groupnorm_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".1.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_1_groupnorm_bias[array_index][dconv_index], ne,
                        nelements);
                }

                // dconv lstm for encoder 4, 5
                for (int lstm_index = 0; lstm_index < 2; ++lstm_index) {
                    for (int direction = 0; direction < 2; ++direction) {
                        std::string direction_suffix = direction == 0 ? "" : "_reverse";
                        std::string layer_suffix = "l" + std::to_string(lstm_index);

                        if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".3.lstm.weight_ih_" + layer_suffix + direction_suffix) {
                            loaded_size = load_single_matrix(
                                f, name,
                                model->encoder_4_5_dconv_layers_3_lstm_ih_w[array_index][dconv_index][lstm_index][direction], ne,
                                nelements);
                        } else if (name == "encoder." + std::to_string(encoder_index) +
                                        ".dconv.layers." + std::to_string(dconv_index) +
                                        ".3.lstm.weight_hh_" + layer_suffix + direction_suffix) {
                            loaded_size = load_single_matrix(
                                f, name,
                                model->encoder_4_5_dconv_layers_3_lstm_hh_w[array_index][dconv_index][lstm_index][direction], ne,
                                nelements);
                        } else if (name == "encoder." + std::to_string(encoder_index) +
                                        ".dconv.layers." + std::to_string(dconv_index) +
                                        ".3.lstm.bias_ih_" + layer_suffix + direction_suffix) {
                            loaded_size = load_single_matrix(
                                f, name,
                                model->encoder_4_5_dconv_layers_3_lstm_ih_b[array_index][dconv_index][lstm_index][direction], ne,
                                nelements);
                        } else if (name == "encoder." + std::to_string(encoder_index) +
                                        ".dconv.layers." + std::to_string(dconv_index) +
                                        ".3.lstm.bias_hh_" + layer_suffix + direction_suffix) {
                            loaded_size = load_single_matrix(
                                f, name,
                                model->encoder_4_5_dconv_layers_3_lstm_hh_b[array_index][dconv_index][lstm_index][direction], ne,
                                nelements);
                        }
                    }
                }

                // continue after the lstm with the attn etc.
                if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".3.linear.weight") {
                    loaded_size = load_single_matrix(
                        f, name,
                        model->encoder_4_5_dconv_layers_3_linear_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".3.linear.bias") {
                    loaded_size = load_single_vector(
                        f, name,
                        model->encoder_4_5_dconv_layers_3_linear_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.content.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_content_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.content.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_content_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.query.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_query_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.query.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_query_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.key.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_key_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.key.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_key_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.query_decay.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_query_decay_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.query_decay.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_query_decay_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.proj.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_proj_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".4.proj.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_4_proj_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".5.weight") {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->encoder_4_5_dconv_layers_5_conv1d_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".5.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_5_conv1d_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".6.weight") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_6_groupnorm_weight[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".6.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_6_groupnorm_bias[array_index][dconv_index], ne,
                        nelements);
                } else if (name == "encoder." + std::to_string(encoder_index) +
                                    ".dconv.layers." + std::to_string(dconv_index) +
                                    ".8.scale") {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->encoder_4_5_dconv_layers_8_scale[array_index][dconv_index], ne,
                        nelements);
                }
            }
        }

        // 4 TEncoders
        for (int i = 0; i < 4; ++i)
        {
            if (name == "tencoder." + std::to_string(i) + ".conv.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tencoder_conv_weight[i], ne, nelements);
            }
            else if (name == "tencoder." + std::to_string(i) + ".conv.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tencoder_conv_bias[i], ne, nelements);
            }
            else if (name ==
                     "tencoder." + std::to_string(i) + ".rewrite.weight")
            {
                loaded_size = load_single_tensor3d(
                    f, name, model->tencoder_rewrite_weight[i], ne, nelements);
            }
            else if (name == "tencoder." + std::to_string(i) + ".rewrite.bias")
            {
                loaded_size = load_single_tensor1d(
                    f, name, model->tencoder_rewrite_bias[i], ne, nelements);
            }

            // each sub-dconv is a stack of 2
            for (int j = 0; j < 2; ++j)
            {
                if (name == "tencoder." + std::to_string(i) + ".dconv.layers." +
                                std::to_string(j) + ".0.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_0_conv1d_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".0.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_0_conv1d_bias[1][0][i][j],
                        ne, nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".1.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_1_groupnorm_bias[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.weight")
                {
                    loaded_size = load_single_tensor3d(
                        f, name,
                        model->dconv_layers_3_conv1d_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".3.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_3_conv1d_bias[1][0][i][j],
                        ne, nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.weight")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_weight[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".4.bias")
                {
                    loaded_size = load_single_tensor1d(
                        f, name,
                        model->dconv_layers_4_groupnorm_bias[1][0][i][j], ne,
                        nelements);
                }
                else if (name == "tencoder." + std::to_string(i) +
                                     ".dconv.layers." + std::to_string(j) +
                                     ".6.scale")
                {
                    loaded_size = load_single_tensor1d(
                        f, name, model->dconv_layers_6_scale[1][0][i][j], ne,
                        nelements);
                }
            }
        }

        // 5th unique tencoder_4
        if (name == "tencoder.4.conv.weight")
        {
            loaded_size = load_single_tensor3d(
                f, name, model->tencoder_4_conv_weight, ne, nelements);
        } else if (name == "tencoder.4.conv.bias")
        {
            loaded_size = load_single_tensor1d(
                f, name, model->tencoder_4_conv_bias, ne, nelements);
        }

        // start with decoder 0,1 for frequency which
        // has its own arrays

        // next, tdecoder_0 which is unique

        // finally, decoder 2,3,4,5
        // and tdecoder 1,2,3,4
        // which are all grouped together in struct members
        // with arrays of size [8]

        // start with decoder 0,1 for frequency which has its own arrays
        for (int i = 0; i < 2; ++i) {
            if (name == "decoder." + std::to_string(i) + ".conv_tr.weight") {
                loaded_size = load_single_tensor3d(
                    f, name, model->decoder_0_1_conv_tr_weight[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".conv_tr.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_conv_tr_bias[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".norm2.weight") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_norm2_weight[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".norm2.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_norm2_bias[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".rewrite.weight") {
                loaded_size = load_single_tensor4d(
                    f, name, model->decoder_0_1_rewrite_weight[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".rewrite.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_rewrite_bias[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".norm1.weight") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_norm1_weight[i], ne, nelements);
            } else if (name == "decoder." + std::to_string(i) + ".norm1.bias") {
                loaded_size = load_single_tensor1d(
                    f, name, model->decoder_0_1_norm1_bias[i], ne, nelements);
            }
        }

        // next, tdecoder_0 which is unique
        if (name == "tdecoder.0.conv_tr.weight") {
            loaded_size = load_single_tensor3d(
                f, name, model->tdecoder_0_conv_tr_weight, ne, nelements);
        } else if (name == "tdecoder.0.conv_tr.bias") {
            loaded_size = load_single_tensor1d(
                f, name, model->tdecoder_0_conv_tr_bias, ne, nelements);
        } else if (name == "tdecoder.0.norm2.weight") {
            loaded_size = load_single_tensor1d(
                f, name, model->tdecoder_0_norm2_weight, ne, nelements);
        } else if (name == "tdecoder.0.norm2.bias") {
            loaded_size = load_single_tensor1d(
                f, name, model->tdecoder_0_norm2_bias, ne, nelements);
        }

        // finally, decoder 2,3,4,5 and tdecoder 1,2,3,4 which are all grouped together in struct members with arrays of size [8]
        // Loop over the first dimension [2] for freq, time
        for (int freq_time = 0; freq_time < 2; ++freq_time) {
            // Loop over the second dimension [4] for layers
            for (int layer = 0; layer < 4; ++layer) {
                // Construct the base name for current decoder/tdecoder
                std::string base_name = (freq_time == 0 ? "decoder." : "tdecoder.") + std::to_string(layer + (freq_time == 0 ? 2 : 1));

                // Load conv_tr.weight
                if (name == base_name + ".conv_tr.weight") {
                    loaded_size = load_single_tensor4d(
                        f, name, model->decoders_conv_tr_weight[freq_time][layer], ne, nelements);
                }
                // Load conv_tr.bias
                else if (name == base_name + ".conv_tr.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name, model->decoders_conv_tr_bias[freq_time][layer], ne, nelements);
                }
                // Load rewrite.weight
                else if (name == base_name + ".rewrite.weight") {
                    loaded_size = load_single_tensor4d(
                        f, name, model->decoders_rewrite_weight[freq_time][layer], ne, nelements);
                }
                // Load rewrite.bias
                else if (name == base_name + ".rewrite.bias") {
                    loaded_size = load_single_tensor1d(
                        f, name, model->decoders_rewrite_bias[freq_time][layer], ne, nelements);
                }
            }
        }

        if (name == "freq_emb.embedding.weight")
        {
            loaded_size = load_single_matrix(
                f, name, model->freq_emb_embedding_weight, ne, nelements);
        }

        if (loaded_size == 0)
        {
            my_fprintf(stderr, "%s: failed to load %s\n", __func__,
                       name.c_str());
            return false;
        }
        total_size += loaded_size;
        n_loaded++;
    }

    fclose(f);

    // compute finish time in microseconds using std::chrono

    const auto t_end_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    // print load time in seconds
    my_fprintf(stdout, "Loaded model (%u tensors, %6.2f MB) in %f s\n",
               n_loaded, total_size / 1024.0 / 1024.0,
               (float)(t_end_us - t_start_us) / 1000000.0f);

    return true;
}
