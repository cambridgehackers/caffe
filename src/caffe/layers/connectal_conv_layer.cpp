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
  const Dtype* bias = NULL;
  if (this->bias_term_)
    bias = this->blobs_[1]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    int top_channels = top[i]->channels();
    int top_height = top[i]->height();
    int top_width = top[i]->width();
    int bottom_channels = bottom[i]->channels();
    int bottom_height = bottom[i]->height();
    int bottom_width = bottom[i]->width();
    int weight_channels = this->blobs_[0]->channels();
    int weight_height = this->blobs_[0]->height();
    int weight_width = this->blobs_[0]->width();
    int kernel_pad_h = this->kernel_h_ - this->pad_h_;
    int kernel_pad_w = this->kernel_w_ - this->pad_w_;
    for (int n = 0; n < this->num_; ++n) {
#ifdef OLDVER
      const Dtype* col_buff = bottom_data + bottom[i]->offset(n);
      if (!this->is_1x1_) {
        this->conv_im2col_cpu(col_buff, this->col_buffer_.mutable_cpu_data());
        col_buff = this->col_buffer_.cpu_data();
      }
#endif
      // Convolution
      for (int g = 0; g < this->group_; ++g) {
#ifdef OLDVER
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
            this->group_, this->conv_out_spatial_dim_, this->kernel_dim_ / this->group_,
            (Dtype)1., weight + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
            (Dtype)0., top_data + top[i]->offset(n) + this->output_offset_ * g);
#else
#define BOFFSET(NAME, N, C, H, W) ((((N) * NAME ## _channels + (C)) * NAME ## _height + (H)) * NAME ## _width + (W))
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
        int o_g = top_channels / this->group_;
        int k_g = bottom_channels / this->group_;
        int kgg = k_g * g;
        Dtype *tp = &top_data[BOFFSET(top, n, o_g * g, 0, 0)];
        for (int o = 0; o < o_g; o++) {
          int o_head = o + o_g * g;
          Dtype bias_val = bias ? bias[o_head] : 0;
          const Dtype *wpy = &weight[BOFFSET(weight, o_head, 0, 0, 0)];
          for (int y = 0; y < top_height; y++) {
            int stride_y = y * this->stride_h_;
            int p_limit = MIN(kernel_pad_h, bottom_height - stride_y);
            const Dtype *bpy = &bottom_data[BOFFSET(bottom, n, kgg, stride_y, 0)];
            for (int x = 0; x < top_width; x++) {
              int stride_x = x * this->stride_w_;
              int q_limit = MIN(kernel_pad_w, bottom_width - stride_x);
              Dtype temp = bias_val;
              const Dtype *bpx = bpy;
              const Dtype *wpx = wpy;
              for (int k = 0; k < k_g; k++) {
                const Dtype *bpk = bpx;
                const Dtype *wpk = wpx;
                for (int p = 0; p < p_limit; p++) {
                  const Dtype *bp = bpk;
                  const Dtype *wp = wpk;
                  for (int q = 0; q < q_limit; q++)
                    temp += *bp++ * *wp++;
                  bpk += BOFFSET(bottom, 0, 0, 1, 0);
                  wpk += BOFFSET(weight, 0, 0, 1, 0);
                }
                bpx += BOFFSET(bottom, 0, 1, 0, 0);
                wpx += BOFFSET(weight, 0, 1, 0, 0);
              }
              bpy += BOFFSET(bottom, 0, 0, 0, this->stride_w_);
              *tp++ = temp;
            }
          }
        }
#endif
      }
      // Bias
#ifdef OLDVER
      if (this->bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
          this->height_out_ * this->width_out_, 1, (Dtype)1., bias, this->bias_multiplier_.cpu_data(),
          (Dtype)1., top_data + top[i]->offset(n));
      }
#endif
    }
  }
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#if 1
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
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
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->height_out_ * this->width_out_, 1.,
            top_diff + top[i]->offset(n), this->bias_multiplier_.cpu_data(), 1., bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
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
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
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
        }
      }
    }
  }
#else
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConnectalConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConnectalConvolutionLayer);

}  // namespace caffe
