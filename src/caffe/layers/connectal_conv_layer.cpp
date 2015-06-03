#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->bias_term_ ? this->blobs_[1]->cpu_data() : NULL;
  int bottom_hw = this->conv_in_height_ * this->conv_in_width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int in_group_size = this->conv_in_channels_ / this->group_;
  // For each input, ...
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
      // Convolution
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
                              this->conv_in_height_ - y * this->stride_h_);
            const Dtype *bpy = bpg;
            for (int x = 0; x < this->width_out_; x++) {
              int q_limit = MIN(this->kernel_w_ - this->pad_w_,
                                this->conv_in_width_ - x * this->stride_w_);
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
                  bpk += this->conv_in_width_;
                  wpk += this->kernel_w_;
                }
                bpx += bottom_hw;
                wpx += kernel_hw;
              }
              // Write convolution result into output (image, channel, y, x)
              *outputp++ = temp;
              bpy += this->stride_w_;
            }
            bpg += this->conv_in_width_ * this->stride_h_;
          }
          wp_base += in_group_size * kernel_hw;
        }
        bottom_data += in_group_size * bottom_hw;
        top_data += this->output_offset_;
      }
    }
  }
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int in_group_size = this->conv_in_channels_ / this->group_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int bottom_hw = this->conv_in_height_ * this->conv_in_width_;
  int height_col = (this->conv_in_height_ + 2 * this->pad_h_ - this->kernel_h_) / this->stride_h_;
  int width_col = (this->conv_in_width_ + 2 * this->pad_w_ - this->kernel_w_) / this->stride_w_;
  int total_in_size = bottom_hw * this->conv_in_channels_;
  Dtype* bias_diff = NULL;

  if (this->param_propagate_down_[0])
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  // For all images
  for (int i = 0; i < top.size(); ++i) {
    // Bias gradient, if necessary.
    if (bias_diff) {
      for (int n = 0; n < this->num_; ++n) {
        const Dtype *top_diff_bp = top[i]->cpu_diff() + top[i]->offset(n);
        for (int j = 0; j < this->num_output_; j++)
          for (int i = 0; i < this->conv_out_spatial_dim_; i++)
            bias_diff[j] += top_diff_bp[j * this->conv_out_spatial_dim_ + i] * this->bias_multiplier_.cpu_data()[i];
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        const Dtype *top_diff_bp = top[i]->cpu_diff() + top[i]->offset(n);
        const Dtype *bottom_bp = bottom[i]->cpu_data() + bottom[i]->offset(n);
        Dtype *bottom_diff_bp =  bottom[i]->mutable_cpu_diff() + bottom[i]->offset(n);
        Dtype *cptr = this->col_buffer_.mutable_cpu_data();
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < this->group_; ++g) {
            for (int cchan = 0; cchan < in_group_size; ++cchan) {
              int garea = g * in_group_size + cchan;
              for (int p = 0; p < this->kernel_h_; ++p) {
                for (int q = 0; q < this->kernel_w_; ++q) {
                    int carea = garea * kernel_hw + p * this->kernel_w_ + q;
                  for (int h = 0; h < (height_col + 1); ++h) {
                    int h_pad = h * this->stride_h_ + p - this->pad_h_;
                    for (int w = 0; w < (width_col + 1); ++w) {
                      int w_pad = w * this->stride_w_ + q - this->pad_w_;
                      cptr[carea * (height_col + 1) * (width_col + 1) + h * (width_col + 1) + w] =
                        (h_pad >= 0 && h_pad < this->conv_in_height_
                         && w_pad >= 0 && w_pad < this->conv_in_width_) ?
                         bottom_bp[garea * bottom_hw + h_pad * this->conv_in_width_ + w_pad] : 0;
                    }
                  }
                  for (int xy = 0; xy < this->conv_out_spatial_dim_; ++xy) {
                    Dtype temp = this->col_buffer_.cpu_data()[carea * this->conv_out_spatial_dim_ + xy];
                    for (int oindex = 0; oindex < out_group_size; ++oindex)
                      weight_diff[this->weight_offset_ * g + (oindex * in_group_size + cchan) * kernel_hw + p * this->kernel_w_ + q]
                        += temp * top_diff_bp[(out_group_size * g + oindex) * this->conv_out_spatial_dim_ + xy];
                  }
                }
              }
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          caffe_set(total_in_size, Dtype(0), bottom_diff_bp);
          for (int g = 0; g < this->group_; ++g) {
            for (int cchan = 0; cchan < in_group_size; ++cchan) {
              for (int p = 0; p < this->kernel_h_; ++p) {
                for (int q = 0; q < this->kernel_w_; ++q) {
                  int garea = g * in_group_size + cchan;
                  int carea = garea * kernel_hw + p * this->kernel_w_ + q;
                  for (int xy = 0; xy < this->conv_out_spatial_dim_; ++xy) {
                    Dtype temp = 0;
                    for (int oindex = 0; oindex < out_group_size; ++oindex)
                      temp += weight[this->weight_offset_ * g + oindex * kernel_hw * in_group_size + cchan * kernel_hw + (p * this->kernel_w_ + q)]
                        * top_diff_bp[this->conv_out_spatial_dim_ * out_group_size * g + oindex * this->conv_out_spatial_dim_ + xy];
                    cptr[carea * this->conv_out_spatial_dim_ + xy] = temp;
                  }
                  for (int h = 0; h < (height_col + 1); ++h) {
                    for (int w = 0; w < (width_col + 1); ++w) {
                      int h_pad = h * this->stride_h_ + p - this->pad_h_;
                      int w_pad = w * this->stride_w_ + q - this->pad_w_;
                      if (h_pad >= 0 && h_pad < this->conv_in_height_
                       && w_pad >= 0 && w_pad < this->conv_in_width_)
                        bottom_diff_bp[garea * bottom_hw + h_pad * this->conv_in_width_ + w_pad]
                         += cptr[carea * (height_col + 1) * (width_col + 1) + h * (width_col + 1) + w];
                    }
                  }
                }
              }
            }
          }
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
