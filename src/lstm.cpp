#include "lstm.hpp"
#include "Eigen/Dense"
#include "model.hpp"
#include <iostream>

// preliminary shapes:
//
// input of shape (batch, input_size) or (input_size): tensor containing
// input features
//
// h_0 of shape (batch, hidden_size) or (hidden_size): tensor containing
// the initial hidden state c_0 of shape (batch, hidden_size) or
// (hidden_size): tensor containing the initial cell state
//
// weight_ih (torch.Tensor) – the learnable input-hidden weights, of
// shape (4*hidden_size, input_size)
//     presumably consisting of: W_ii, W_if, W_ig, W_io
// weight_hh (torch.Tensor) – the learnable hidden-hidden weights, of
// shape (4*hidden_size, hidden_size)
//     presumably consisting of: W_hi, W_hf, W_hg, W_ho
//
// similarly for biases:
//     bias_ih (torch.Tensor) – the learnable input-hidden bias, of
//     shape (4*hidden_size) bias_hh (torch.Tensor) – the learnable
//     hidden-hidden bias, of shape (4*hidden_size)
//
// it = sigmoid(W_ii x_t + b_ii + W_hi h_t + b_hi)
// ft = sigmoid(W_if x_t + b_if + W_hf h_t + b_hf)
// gt = tanh(W_ig x_t + b_ig + W_hg h_t + b_hg)
// ot = sigmoid(W_io x_t + b_io + W_ho h_t + b_ho)
// ct = f * c + i * g
//     eigen's array() multiplication is element-wise multiplication
//     i.e. Hadamard product
// ht = o * tanh(c)

static Eigen::MatrixXf sigmoid(const Eigen::MatrixXf &x)
{
    return 1.0 / (1.0 + (-x).array().exp());
}

void demucscpp_v3::lstm_forward(const struct demucscpp_v3::demucs_v3_model& model,
                                const int encoder_idx,
                                const int dconv_idx,
                                const Eigen::MatrixXf &input,
                                struct demucscpp_v3::demucs_v3_segment_buffers &data,
                                int hidden_size)
{
    int seq_len = input.rows();
    int hidden_state_size = hidden_size;

    Eigen::MatrixXf loop_input = input;

    for (int lstm_layer = 0; lstm_layer < 3; ++lstm_layer)
    {
        for (int direction = 0; direction < 2; ++direction)
        {
            // forward direction = 0: for t = 0 to seq_len - 1
            // backward direction = 1: for t = seq_len - 1 to 0
            for (int t = (direction == 0 ? 0 : seq_len - 1);
                 (direction == 0 ? t < seq_len : t > -1);
                 t += (direction == 0 ? 1 : -1))
            {
                // apply the inner input/hidden gate calculation for all gates
                // W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh
                //
                // at the end of the loop iteration, h[lstm_layer][direction]
                // will store h_t of this iteration at the beginning of the next
                // loop iteration, h[lstm_layer][direction] will be h_{t-1},
                // which is what we want similar for c[lstm_layer][direction]
                // and c_{t-1}
                //
                // the initial values for h and c are 0
                Eigen::MatrixXf gates =
                    model.encoder_4_5_dconv_layers_3_lstm_hh_w[encoder_idx][dconv_idx][lstm_layer][direction]
                            *
                        loop_input.row(t).transpose() +
                    model.encoder_4_5_dconv_layers_3_lstm_ih_b[encoder_idx][dconv_idx][lstm_layer][direction] +
                    model.encoder_4_5_dconv_layers_3_lstm_hh_w[encoder_idx][dconv_idx][lstm_layer][direction]
                            *
                        data.lstm_hidden[encoder_idx][dconv_idx][lstm_layer][direction] +
                    model.encoder_4_5_dconv_layers_3_lstm_hh_b[encoder_idx][dconv_idx][lstm_layer][direction];

                // slice up the gates into i|f|g|o-sized chunks
                Eigen::MatrixXf i_t =
                    sigmoid(gates.block(0, 0, hidden_state_size, 1));
                Eigen::MatrixXf f_t = sigmoid(
                    gates.block(hidden_state_size, 0, hidden_state_size, 1));
                Eigen::MatrixXf g_t = (gates.block(2 * hidden_state_size, 0,
                                                   hidden_state_size, 1))
                                          .array()
                                          .tanh();
                Eigen::MatrixXf o_t = sigmoid(gates.block(
                    3 * hidden_state_size, 0, hidden_state_size, 1));

                Eigen::MatrixXf c_t =
                    f_t.array() * data.lstm_cell[encoder_idx][dconv_idx][lstm_layer][direction].array() +
                    i_t.array() * g_t.array();
                Eigen::MatrixXf h_t = o_t.array() * (c_t.array().tanh());

                // store the hidden and cell states for later use
                data.lstm_hidden[encoder_idx][dconv_idx][lstm_layer][direction] = h_t;
                data.lstm_cell[encoder_idx][dconv_idx][lstm_layer][direction] = c_t;

                data.lstm_output_per_direction[encoder_idx][dconv_idx][lstm_layer][direction].row(t)
                    << h_t.transpose();
            }
        }

        // after both directions are done per LSTM layer, concatenate the
        // outputs
        data.lstm_output[encoder_idx][dconv_idx][lstm_layer] << data.lstm_output_per_direction[encoder_idx][dconv_idx][lstm_layer][0],
            data.lstm_output_per_direction[encoder_idx][dconv_idx][lstm_layer][1];

        loop_input = data.lstm_output[encoder_idx][dconv_idx][lstm_layer];
    }

    // the concatenated forward and backward hidden state is the final output
    // no need to return it, caller should access it
    //return data->output[2];
}
