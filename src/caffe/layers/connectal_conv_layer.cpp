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
  int bottom_hw = this->height_ * this->width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int per_out_group = this->conv_out_channels_ / this->group_;
  int per_in_group = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
      // Convolution
#ifdef OLDVER
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
#else
    int nblock = 0;
    for (int nunused = 0; nunused < this->num_; ++nunused) {
      const Dtype *biasptr = bias;
      for (int g = 0; g < this->group_; ++g) {
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
        Dtype *tp = &top_data[nblock * this->output_offset_];
        const Dtype *bp_base = &bottom_data[nblock * per_in_group * bottom_hw];
        const Dtype *wpy = &weight[g * this->weight_offset_];
        for (int ounused = 0; ounused < per_out_group; ounused++) {
          Dtype bias_val = bias ? *biasptr++ : 0;
          const Dtype *bpg = bp_base;
          for (int y = 0; y < this->height_out_; y++) {
            int p_limit = MIN(this->kernel_h_ - this->pad_h_, this->height_ - y * this->stride_h_);
            const Dtype *bpy = bpg;
            for (int x = 0; x < this->width_out_; x++) {
              int q_limit = MIN(this->kernel_w_ - this->pad_w_, this->width_ - x * this->stride_w_);
              Dtype temp = bias_val;
              const Dtype *bpx = bpy;
              const Dtype *wpx = wpy;
              for (int k = 0; k < per_in_group; k++) {
                const Dtype *bpk = bpx;
                const Dtype *wpk = wpx;
                for (int p = 0; p < p_limit; p++) {
                  const Dtype *bp = bpk;
                  const Dtype *wp = wpk;
                  for (int q = 0; q < q_limit; q++)
                    temp += *bp++ * *wp++;
                  bpk += this->width_;
                  wpk += this->kernel_w_;
                }
                bpx += bottom_hw;
                wpx += kernel_hw;
              }
              bpy += this->stride_w_;
              *tp++ = temp;
            }
            bpg += this->width_ * this->stride_h_;
          }
          wpy += per_in_group * kernel_hw;
        }
        nblock++;
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
#ifdef OLDVER
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->height_out_ * this->width_out_, 1.,
            top_diff + top[i]->offset(n), this->bias_multiplier_.cpu_data(), 1., bias_diff);
#else
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->height_out_ * this->width_out_, 1.,
            top_diff + top[i]->offset(n), this->bias_multiplier_.cpu_data(), 1., bias_diff);
#endif
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
#ifdef OLDVER
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
#ifdef OLDVER
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
