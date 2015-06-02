#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//#define OLDVER
namespace caffe {

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->bias_term_ ? this->blobs_[1]->cpu_data() : NULL;
  int bottom_hw = this->height_ * this->width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int in_group_size = this->conv_in_channels_ / this->group_;
  // For each input, ...
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
      // Convolution
#ifndef OLDVER
    // For each image in input batch
    for (int nunused = 0; nunused < this->num_; ++nunused) {
      const Dtype *biasptr = bias;
      // if group_ > 1, restrict connectivity to a subset of inputs
      for (int g = 0; g < this->group_; ++g) {
        Dtype *outputp = top_data;
        const Dtype *wp_base = &weight[g * this->weight_offset_];
        // for each 'out_group', calculate convolution over input data
        for (int ounused = 0; ounused < out_group_size; ounused++) {
          const Dtype *bpg = bottom_data;
          const Dtype bias_val = bias ? *biasptr++ : 0;
          // Scan over source 2D input data
          for (int y = 0; y < this->height_out_; y++) {
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
            int p_limit = MIN(this->kernel_h_ - this->pad_h_,
                              this->height_ - y * this->stride_h_);
            const Dtype *bpy = bpg;
            for (int x = 0; x < this->width_out_; x++) {
              int q_limit = MIN(this->kernel_w_ - this->pad_w_,
                                this->width_ - x * this->stride_w_);
              Dtype temp = bias_val;
              const Dtype *bpx = bpy, *wpx = wp_base;
              // for each 'in_group', add contribution into convolution
              for (int k = 0; k < in_group_size; k++) {
                const Dtype *bpk = bpx, *wpk = wpx;
                // Calculate single 2D filter convolution
                for (int p = 0; p < p_limit; p++) {
                  const Dtype *bp = bpk, *wp = wpk;
                  for (int q = 0; q < q_limit; q++)
                    temp += *bp++ * *wp++;
                  bpk += this->width_;
                  wpk += this->kernel_w_;
                }
                bpx += bottom_hw;
                wpx += kernel_hw;
              }
              // Write convolution result into output (image, channel, y, x)
              *outputp++ = temp;
              bpy += this->stride_w_;
            }
            bpg += this->width_ * this->stride_h_;
          }
          wp_base += in_group_size * kernel_hw;
        }
        bottom_data += in_group_size * bottom_hw;
        top_data += this->output_offset_;
      }
    }
#else
    for (int n = 0; n < this->num_; ++n) {
      const Dtype* col_buff = bottom_data + bottom[i]->offset(n);
      if (!this->is_1x1_) {
        this->conv_im2col_cpu(col_buff, this->col_buffer_.mutable_cpu_data());
        col_buff = this->col_buffer_.cpu_data();
      }
      for (int g = 0; g < this->group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
            this->group_, this->conv_out_spatial_dim_, this->kernel_dim_ / this->group_,
            (Dtype)1., weight + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
            (Dtype)0., top_data + top[i]->offset(n) + this->output_offset_ * g);
      }
      // Bias
      if (this->bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
          this->height_out_ * this->width_out_, 1, (Dtype)1., bias, this->bias_multiplier_.cpu_data(),
          (Dtype)1., top_data + top[i]->offset(n));
      }
    }
#endif
  }
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int out_group_size = this->conv_out_channels_ / this->group_;
  int kernel_dim_group = this->kernel_dim_ / this->group_;
  if (this->param_propagate_down_[0])
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  if (this->bias_term_ && this->param_propagate_down_[1])
    caffe_set(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_cpu_diff());
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
#ifndef OLDVER
       int N = this->height_out_ * this->width_out_;
       const Dtype *A = top_diff + top[i]->offset(n);
       for (int j = 0; j < this->num_output_; j++) {
           Dtype temp = 0;
           for (int i = 0; i < N; i++)
               temp += A[j * N + i] * this->bias_multiplier_.cpu_data()[i];
           bias_diff[j] += temp;
       }
#else
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->height_out_ * this->width_out_, 1.,
            top_diff + top[i]->offset(n), this->bias_multiplier_.cpu_data(), 1., bias_diff);
#endif
      }
    }
    int height_col = (this->conv_in_height_ + 2 * this->pad_h_ - this->kernel_h_) / this->stride_h_ + 1;
    int width_col = (this->conv_in_width_ + 2 * this->pad_w_ - this->kernel_w_) / this->stride_w_ + 1;
    int channels_col = this->conv_in_channels_ * this->kernel_h_ * this->kernel_w_;
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
#ifndef OLDVER
          const Dtype* col_buff = bottom_data + bottom[i]->offset(n);
          if (!this->is_1x1_) {
            for (int c = 0; c < channels_col; ++c) {
              int w_offset = c % this->kernel_w_;
              int h_offset = (c / this->kernel_w_) % this->kernel_h_;
              int c_im = c / this->kernel_h_ / this->kernel_w_;
              for (int h = 0; h < height_col; ++h) {
                for (int w = 0; w < width_col; ++w) {
                  int h_pad = h * this->stride_h_ - this->pad_h_ + h_offset;
                  int w_pad = w * this->stride_w_ - this->pad_w_ + w_offset;
                  if (h_pad >= 0 && h_pad < this->conv_in_height_ && w_pad >= 0 && w_pad < this->conv_in_width_)
                    (this->col_buffer_.mutable_cpu_data())[(c * height_col + h) * width_col + w] =
                      (bottom_data + bottom[i]->offset(n))[(c_im * this->conv_in_height_ + h_pad) * this->conv_in_width_ + w_pad];
                  else
                    (this->col_buffer_.mutable_cpu_data())[(c * height_col + h) * width_col + w] = 0;
                }
              }
            }
            col_buff = this->col_buffer_.cpu_data();
          }
          for (int g = 0; g < this->group_; ++g) {
            for (int col = 0; col < kernel_dim_group; ++col) {
                for (int l = 0; l < this->conv_out_spatial_dim_; ++l) {
                    Dtype temp = (col_buff + this->col_offset_ * g)[col * this->conv_out_spatial_dim_ + l];
                    for (int row = 0; row < out_group_size; ++row)
                        (weight_diff + this->weight_offset_ * g)[row * kernel_dim_group + col]
                            += temp * (top_diff + top[i]->offset(n) + this->output_offset_ * g)
                               [row * this->conv_out_spatial_dim_ + l];
                }
            }
          }
#else
          const Dtype* col_buff = bottom_data + bottom[i]->offset(n);
          if (!this->is_1x1_) {
            this->conv_im2col_cpu(col_buff, this->col_buffer_.mutable_cpu_data());
            col_buff = this->col_buffer_.cpu_data();
          }
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, this->conv_out_channels_ / this->group_,
                this->kernel_dim_ / this->group_, this->conv_out_spatial_dim_,
                (Dtype)1., top_diff + top[i]->offset(n) + this->output_offset_ * g, col_buff + this->col_offset_ * g,
                (Dtype)1., weight_diff + this->weight_offset_ * g);
          }
#endif
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
#ifndef OLDVER
          Dtype* col_buff = this->col_buffer_.mutable_cpu_data();
          if (this->is_1x1_)
            col_buff = bottom_diff + bottom[i]->offset(n);
          for (int g = 0; g < this->group_; ++g) {
            for (int col = 0; col < this->conv_out_spatial_dim_; ++col) {
                for (int row = 0; row < kernel_dim_group; ++row) {
                    Dtype temp = 0;
                    for (int l = 0; l < out_group_size; ++l)
                       temp += (weight + this->weight_offset_ * g)[l * kernel_dim_group + row]
                          * (top_diff + top[i]->offset(n) + this->output_offset_ * g)[l * this->conv_out_spatial_dim_ + col];
                    (col_buff + this->col_offset_ * g)[row * this->conv_out_spatial_dim_ + col] = temp;
                }
            }
          }
          if (!this->is_1x1_) {
            caffe_set(this->conv_in_height_ * this->conv_in_width_ * this->conv_in_channels_, Dtype(0), (bottom_diff + bottom[i]->offset(n)));
            for (int c = 0; c < channels_col; ++c) {
              int w_offset = c % this->kernel_w_;
              int h_offset = (c / this->kernel_w_) % this->kernel_h_;
              int c_im = c / this->kernel_h_ / this->kernel_w_;
              for (int h = 0; h < height_col; ++h) {
                for (int w = 0; w < width_col; ++w) {
                  int h_pad = h * this->stride_h_ - this->pad_h_ + h_offset;
                  int w_pad = w * this->stride_w_ - this->pad_w_ + w_offset;
                  if (h_pad >= 0 && h_pad < this->conv_in_height_ && w_pad >= 0 && w_pad < this->conv_in_width_)
                    (bottom_diff + bottom[i]->offset(n))[(c_im * this->conv_in_height_ + h_pad) * this->conv_in_width_ + w_pad] +=
                        (col_buff)[(c * height_col + h) * width_col + w];
                }
              }
            }
          }
#else
          Dtype* col_buff = this->col_buffer_.mutable_cpu_data();
          if (this->is_1x1_)
            col_buff = bottom_diff + bottom[i]->offset(n);
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->kernel_dim_ / this->group_,
                this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
                (Dtype)1., weight + this->weight_offset_ * g, top_diff + top[i]->offset(n) + this->output_offset_ * g,
                (Dtype)0., col_buff + this->col_offset_ * g);
          }
          if (!this->is_1x1_)
            this->conv_col2im_cpu(col_buff, bottom_diff + bottom[i]->offset(n));
#endif
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConnectalConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConnectalConvolutionLayer);

}  // namespace caffe
