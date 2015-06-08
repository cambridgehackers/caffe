#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
#define MAX(A,B) (((A) > (B)) ? (A) : (B))
namespace caffe {
template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int bottom_hw = this->conv_in_height_ * this->conv_in_width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int in_group_size = this->conv_in_channels_ / this->group_;
  const Dtype* bias = this->bias_term_ ? this->blobs_[1]->cpu_data() : NULL;
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
  int bottom_hw = this->conv_in_height_ * this->conv_in_width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int in_group_size = this->conv_in_channels_ / this->group_;
  int usable_height = this->conv_in_height_ + 2 * this->pad_h_ - this->kernel_h_;
  int usable_width = this->conv_in_width_ + 2 * this->pad_w_ - this->kernel_w_;
  int total_in_kernel = in_group_size * kernel_hw;
  Dtype* weight_diff = NULL;
  Dtype* bias_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  // For all images
  for (int i = 0; i < top.size(); ++i) {
    for (int n = 0; n < this->num_; ++n) {
      int boff = n * this->conv_in_channels_ * bottom_hw;
      const Dtype *top_diff_bp = top[i]->cpu_diff()
          + n * this->conv_out_channels_ * this->conv_out_spatial_dim_;
      const Dtype *bottom_bp = bottom[i]->cpu_data() + boff;
      Dtype *bottom_diff_bp = NULL;
      // Bias gradient, if necessary.
      if (bias_diff) {
        const Dtype *tptr = top_diff_bp;
        for (int j = 0; j < this->num_output_; j++)
          for (int i = 0; i < this->conv_out_spatial_dim_; i++)
            bias_diff[j] += *tptr++ * this->bias_multiplier_.cpu_data()[i];
      }
      if (propagate_down[i]) {
        bottom_diff_bp =  bottom[i]->mutable_cpu_diff() + boff;
      }
      if (weight_diff || bottom_diff_bp) {
        for (int g = 0; g < this->group_; ++g) {
          for (int cchan = 0; cchan < in_group_size; ++cchan) {
            int gchan = (g * in_group_size + cchan) * bottom_hw;
            int wbase = g * this->weight_offset_ + cchan * kernel_hw;
#if 1
            if (weight_diff)
            for (int outindex = 0; outindex < out_group_size; ++outindex) {
              int wchan = wbase + outindex * total_in_kernel;
              const Dtype *topdptr = &top_diff_bp[g * this->output_offset_ + outindex * this->conv_out_spatial_dim_];
              for (int y = 0; y <= usable_height; y += this->stride_h_){
                for (int x = 0; x <= usable_width; x += this->stride_w_) {
                  Dtype chain_grad = topdptr[(y * (usable_width + this->stride_w_) / this->stride_h_ + x) / this->stride_w_ ];
//__builtin_prefetch(&bottom_bp[gchan + poffset * this->conv_in_width_]);
                  int p_start = MAX(0, this->pad_h_ - y);
                  int q_start = MAX(0, this->pad_w_ - x);
                  int p_limit = MIN(this->conv_in_height_ - y + this->pad_h_, this->kernel_h_ - 1);
                  int q_limit = MIN(this->conv_in_width_ - x + this->pad_w_, this->kernel_w_ - 1);
                  int bebase = gchan + (y - this->pad_h_) * this->conv_in_width_ + (x - this->pad_w_);
                  for (int p = p_start; p <= p_limit; ++p) {
                    for (int q = q_start; q <= q_limit; ++q) {
                      int belement = bebase + p * this->conv_in_width_ + q;
                      int welement = wchan + p * this->kernel_w_ + q;
                      // gradient w.r.t. weight. Note that we will accumulate diffs.
                      weight_diff[welement] += bottom_bp[belement] * chain_grad;
                    }
                  }
                }
              }
            }
            if (bottom_diff_bp)
            for (int poffset = this->pad_h_; poffset < this->conv_in_height_ + this->pad_h_; ++poffset) {
              for (int qoffset = this->pad_w_; qoffset < this->conv_in_width_ + this->pad_w_; ++qoffset) {
                int belement = gchan + poffset * this->conv_in_width_ + qoffset;
                Dtype temp = 0;
                int y_start = MAX(0, (poffset - this->kernel_h_ + this->stride_h_) / this->stride_h_);
                int x_start = MAX(0, (qoffset - this->kernel_w_ + this->stride_w_) / this->stride_w_);
                int y_limit = MIN(usable_height, poffset)/ this->stride_h_;
                int x_limit = MIN(usable_width, qoffset) / this->stride_w_;
                for (int outindex = 0; outindex < out_group_size; ++outindex) {
                  int wchan = wbase + outindex * total_in_kernel;
                  const Dtype *topdptr = &top_diff_bp[g * this->output_offset_ + outindex * this->conv_out_spatial_dim_];
                  for (int y = y_start; y <= y_limit; y++) {
                    for (int x = x_start; x <= x_limit; x++) {
                      int welement = wchan + (poffset - y * this->stride_h_)
                         * this->kernel_w_ + (qoffset - x * this->stride_w_);
                      // gradient w.r.t. bottom data, if necessary.
                      temp += weight[welement] * topdptr[y * (usable_width / this->stride_w_ + 1) + x];
                    }
                  }
                }
                bottom_diff_bp[belement] = temp;
              }
            }
#else
            // zero out gradient wrt bottom data, we're about to fill it
            if (bottom_diff_bp)
              caffe_set(bottom_hw, Dtype(0), &bottom_diff_bp[gchan]);
            for (int outindex = 0; outindex < out_group_size; ++outindex) {
              int wchan = g * this->weight_offset_ + (cchan + outindex * in_group_size) * kernel_hw;
              const Dtype *topdptr = &top_diff_bp[g * this->output_offset_ + outindex * this->conv_out_spatial_dim_];
              for (int y = 0; y <= usable_height; y += this->stride_h_){
                for (int x = 0; x <= usable_width; x += this->stride_w_) {
                  for (int p = 0; p < this->kernel_h_; ++p) {
                    for (int q = 0; q < this->kernel_w_; ++q) {
                      int poffset = y + p - this->pad_h_;
                      int qoffset = x + q - this->pad_w_;
                      if (poffset >= 0 && poffset < this->conv_in_height_ && qoffset >= 0 && qoffset < this->conv_in_width_) {
                        int belement = gchan + poffset * this->conv_in_width_ + qoffset;
                        int welement = wchan + p * this->kernel_w_ + q;
                        Dtype chain_grad = topdptr[(y * (usable_width + this->stride_w_) / this->stride_h_ + x) / this->stride_w_ ];
                        // gradient w.r.t. weight. Note that we will accumulate diffs.
                        if (weight_diff)
                          weight_diff[welement] += bottom_bp[belement] * chain_grad;
                        // gradient w.r.t. bottom data, if necessary.
                        if (bottom_diff_bp)
                          bottom_diff_bp[belement] += weight[welement] * chain_grad;
                      }
                    }
                  }
                }
              }
            }
#endif
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
