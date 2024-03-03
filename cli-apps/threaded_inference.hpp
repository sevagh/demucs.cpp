#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/*
    this is a multithreaded driver program of demucs.cpp
    which splits the input song into N segments and processes each independently

    javascript code here:
    https://github.com/sevagh/free-music-demixer/blob/main/docs/main.js#L23

    also similar to src/model_apply.cpp which implements the real
    demucs 7.8-second segmentation
*/
namespace demucscppthreaded
{
// bigger overlap from free-music-demixer
const int SAMPLE_RATE = 44100;
const float OVERLAP = 0.75;
const int OVERLAP_SAMPLES = ::floorf(SAMPLE_RATE * OVERLAP);

Eigen::Tensor3dXf
threaded_inference(const struct demucscpp::demucs_model &model,
                   const Eigen::MatrixXf &full_audio, int num_threads,
                   const std::string &prefix = "")
{
    // set output precision to 3 decimal places
    std::cout << std::fixed << std::setprecision(3);

    // create vector of progresscallbacks per-thread
    std::vector<demucscpp::ProgressCallback> cbs;
    for (int i = 0; i < num_threads; ++i)
    {
        cbs.push_back(
            [i, prefix](float progress, const std::string &log_message)
            {
                std::cout << prefix << "[THREAD " << i << "] (" << std::setw(3)
                          << std::setfill(' ') << progress * 100.0f << "%) "
                          << log_message << std::endl;
            });
    }

    // calculate segment length by dividing n_samples by num_threads
    int total_length = full_audio.cols();
    int segment_length = ::ceilf((float)total_length / (float)num_threads);

    std::vector<Eigen::MatrixXf> segments;
    // split the full audio into segments
    for (int i = 0; i < num_threads; ++i)
    {
        int start = i * segment_length;
        int end = std::min(total_length, start + segment_length);

        // Create a new segment with padding for overlap
        Eigen::MatrixXf segment =
            Eigen::MatrixXf::Zero(2, end - start + 2 * OVERLAP_SAMPLES);

        // Overlap-padding for the left and right channels
        // For the first segment, no padding at the start
        if (i == 0)
        {
            segment.block(0, 0, 2, OVERLAP_SAMPLES).colwise() =
                full_audio.col(0);
        }
        else
        {
            segment.block(0, 0, 2, OVERLAP_SAMPLES) = full_audio.block(
                0, start - OVERLAP_SAMPLES, 2, OVERLAP_SAMPLES);
        }

        // For the last segment, no padding at the end
        if (i == num_threads - 1)
        {
            int remaining_samples = total_length - end;
            segment.block(0, end - start + OVERLAP_SAMPLES, 2,
                          remaining_samples) =
                full_audio.block(0, end, 2, remaining_samples);
        }
        else
        {
            segment.block(0, end - start + OVERLAP_SAMPLES, 2,
                          OVERLAP_SAMPLES) =
                full_audio.block(0, end, 2, OVERLAP_SAMPLES);
        }

        // Assign the original segment data
        segment.block(0, OVERLAP_SAMPLES, 2, end - start) =
            full_audio.block(0, start, 2, end - start);
        segments.push_back(segment);
    }

    // insert parallel processing here
    // pretend like segment_outs contains:
    //   (4, 2, segment_samples)
    // which are 4 targets, stereo/2 channels, and the above segment length
    // and we want this to be recombined into a single tensor
    // i.e. Eigen::Tensor3dXf(4, 2, total_length)
    std::vector<Eigen::Tensor3dXf> segment_outs(num_threads);

    // This vector will hold the threads
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(
            [&model, &segments, &segment_outs, i, &cbs]() {
                segment_outs[i] =
                    demucscpp::demucs_inference(model, segments[i], cbs[i]);
            });
    }

    // Wait for all threads to finish
    for (auto &thread : threads)
    {
        thread.join();
    }

    // Calculate total output size and create the output tensor
    Eigen::Tensor3dXf final_output(4, 2, total_length);
    final_output.setZero();

    Eigen::VectorXf ramp(segment_length);
    for (int i = 0; i < segment_length; ++i)
    {
        ramp(i) = std::min(i + 1, segment_length - i);
    }
    ramp /= ramp.maxCoeff(); // Normalize the ramp

    Eigen::VectorXf sum_weight = Eigen::VectorXf::Zero(total_length);

    for (size_t i = 0; i < segment_outs.size(); ++i)
    {
        int segment_start = i * segment_length;
        for (int t = 0; t < 4; ++t)
        { // For each target
            for (int ch = 0; ch < 2; ++ch)
            { // For each channel
                for (int j = 0; j < segment_length + 2 * OVERLAP_SAMPLES; ++j)
                {
                    int global_idx = segment_start + j - OVERLAP_SAMPLES;
                    if (global_idx >= 0 && global_idx < total_length)
                    {
                        float weight = 1.0;
                        // Apply ramp weights at the beginning and end of the
                        // segment
                        if (j < OVERLAP_SAMPLES)
                        {
                            weight = ramp(j);
                        }
                        else if (j >= segment_length)
                        {
                            weight = ramp(segment_length + 2 * OVERLAP_SAMPLES -
                                          j - 1);
                        }
                        final_output(t, ch, global_idx) +=
                            segment_outs[i](t, ch, j) * weight;
                        sum_weight(global_idx) += weight;
                    }
                }
            }
        }
    }

    // Normalize the output by the sum of weights
    for (int t = 0; t < 4; ++t)
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int i = 0; i < total_length; ++i)
            {
                if (sum_weight(i) > 0)
                {
                    // account for summing per-target by dividing by n targets,
                    // 2 channels
                    final_output(t, ch, i) /= (sum_weight(i) / (2.0f * 4.0f));
                }
            }
        }
    }

    return final_output;
}
}; // namespace demucscppthreaded
