#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "papi.h"
#include <assert.h>
#define MIN(A,B) (((A) < (B)) ? (A) : (B))
#define MAX(A,B) (((A) > (B)) ? (A) : (B))

//#define PERFSTAT
#define NUM_EVENTS 4
static void perfpinit(void)
{
#ifdef PERFSTAT
  static int once = 1;
  int event[NUM_EVENTS] = {PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L1_DCM };
  if (once) {
    once = 0;
    /* Start counting events */
    if (PAPI_start_counters(event, NUM_EVENTS) != PAPI_OK) {
        fprintf(stderr, "PAPI_start_counters - FAILED\n");
        exit(1);
    }
  }
#endif
}
static void perfread(long long *perfvalues)
{
#ifdef PERFSTAT
    if (PAPI_read_counters(perfvalues, NUM_EVENTS) != PAPI_OK) {
        fprintf(stderr, "PAPI_read_counters - FAILED\n");
        exit(1);
    }
#endif
}
#ifdef PERFSTAT
static void perfperf(long long *perfvalues, const char *name)
{
    printf("%s: Total instructions: %6lld;", name, perfvalues[0]);
    printf("Total cycles: %6lld;", perfvalues[1]);
    printf("Instr per cycle: %2.3f;", (double)perfvalues[0] / (double) perfvalues[1]);
    printf("Branches mispredicted: %6lld;", perfvalues[2]);
    printf("L1 Cache misses: %6lld\n", perfvalues[3]);
}
#endif
template <typename Dtype>
class ParamType {
public:
    const Dtype* weight;
    const Dtype* bias;
    const Dtype **bottom;
    Dtype **top;
    const Dtype *bias_multiplier_;
    Dtype **bottom_diff;
    const Dtype **top_diff;
    Dtype *weight_diff;
    Dtype *bias_diff;
    int top_size;
    int bottom_size;
    int weight_diff_count;
    int bias_diff_count;
    int num_;
    int num_output_;
    int group_;
    int height_out_, width_out_;
    int kernel_h_, kernel_w_;
    int conv_in_height_, conv_in_width_;
    int conv_in_channels_, conv_out_channels_;
    int conv_out_spatial_dim_;
    int weight_offset_;
    int output_offset_;
    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    //const 
    bool *propagate_down;
};
#define CASSERT assert
namespace caffe {
template <typename Dtype>
    ParamType<Dtype> *forward_init(ConnectalConvolutionLayer<Dtype> *base,
             const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top)
{
    typedef Dtype *dptr;
    typedef const Dtype *cdptr;
    ParamType<Dtype> *param = static_cast<ParamType<Dtype> *>(base->paramPtr);
    if (param) {
        CASSERT(param->weight == base->blobs_[0]->cpu_data());
        CASSERT(param->top_size == top.size());
        CASSERT(param->bottom_size == bottom.size());
        for (int i = 0; i < param->bottom_size; i++) {
             CASSERT(param->bottom[i] == bottom[i]->cpu_data());
             CASSERT(param->bottom_diff[i] == bottom[i]->mutable_cpu_diff());
        }
        for (int i = 0; i < param->top_size; i++) {
             CASSERT(param->top[i] == top[i]->mutable_cpu_data());
             CASSERT(param->top_diff[i] == top[i]->cpu_diff());
        }
        CASSERT(param->weight_diff == base->blobs_[0]->mutable_cpu_diff());
        CASSERT(param->weight_diff_count == base->blobs_[0]->count());
        if (base->bias_term_) {
            CASSERT(param->bias == base->blobs_[1]->cpu_data());
            CASSERT(param->bias_diff == base->blobs_[1]->mutable_cpu_diff());
            CASSERT(param->bias_diff_count == base->blobs_[1]->count());
            CASSERT(param->bias_multiplier_ == base->bias_multiplier_.cpu_data());
        }
        CASSERT(param->num_ == base->num_);
        CASSERT(param->num_output_ == base->num_output_);
        CASSERT(param->group_ == base->group_);
        CASSERT(param->height_out_ == base->height_out_);
        CASSERT(param->width_out_ == base->width_out_);
        CASSERT(param->kernel_h_ == base->kernel_h_);
        CASSERT(param->kernel_w_ == base->kernel_w_);
        CASSERT(param->conv_in_height_ == base->conv_in_height_);
        CASSERT(param->conv_in_width_ == base->conv_in_width_);
        CASSERT(param->conv_in_channels_ == base->conv_in_channels_);
        CASSERT(param->conv_out_channels_ == base->conv_out_channels_);
        CASSERT(param->conv_out_spatial_dim_ == base->conv_out_spatial_dim_);
        CASSERT(param->weight_offset_ == base->weight_offset_);
        CASSERT(param->output_offset_ == base->output_offset_);
        CASSERT(param->pad_h_ == base->pad_h_);
        CASSERT(param->pad_w_ == base->pad_w_);
        CASSERT(param->stride_h_ == base->stride_h_);
        CASSERT(param->stride_w_ == base->stride_w_);
        return param;
    }
    param = new ParamType<Dtype>;
printf("[%s:%d] param %p\n", __FUNCTION__, __LINE__, param);
    base->paramPtr = param;
    memset(param, 0, sizeof(*param));
    param->weight = base->blobs_[0]->cpu_data();
    param->top_size = top.size();
    param->bottom_size = bottom.size();
    param->bottom = new cdptr[param->bottom_size];
    param->bottom_diff = new dptr[param->bottom_size];
    for (int i = 0; i < param->bottom_size; i++) {
         param->bottom[i] = bottom[i]->cpu_data();
         param->bottom_diff[i] = bottom[i]->mutable_cpu_diff();
    }
    param->top = new dptr[param->top_size];
    param->top_diff = new cdptr[param->top_size];
    for (int i = 0; i < param->top_size; i++) {
         param->top[i] = top[i]->mutable_cpu_data();
         param->top_diff[i] = top[i]->cpu_diff();
    }
    param->weight_diff = base->blobs_[0]->mutable_cpu_diff();
    param->weight_diff_count = base->blobs_[0]->count() * sizeof(Dtype);
    if (base->bias_term_) {
        param->bias = base->blobs_[1]->cpu_data();
        param->bias_diff = base->blobs_[1]->mutable_cpu_diff();
        param->bias_diff_count = base->blobs_[1]->count() * sizeof(Dtype);
        param->bias_multiplier_ = base->bias_multiplier_.cpu_data();
    }
    param->num_ = base->num_;
    param->num_output_ = base->num_output_;
    param->group_ = base->group_;
    param->height_out_ = base->height_out_;
    param->width_out_ = base->width_out_;
    param->kernel_h_ = base->kernel_h_;
    param->kernel_w_ = base->kernel_w_;
    param->conv_in_height_ = base->conv_in_height_;
    param->conv_in_width_ = base->conv_in_width_;
    param->conv_in_channels_ = base->conv_in_channels_;
    param->conv_out_channels_ = base->conv_out_channels_;
    param->conv_out_spatial_dim_ = base->conv_out_spatial_dim_;
    param->weight_offset_ = base->weight_offset_;
    param->output_offset_ = base->output_offset_;
    param->pad_h_ = base->pad_h_;
    param->pad_w_ = base->pad_w_;
    param->stride_h_ = base->stride_h_;
    param->stride_w_ = base->stride_w_;
    return param;
}

template <typename Dtype>
    ParamType<Dtype> *backward_init(ConnectalConvolutionLayer<Dtype> *base,
             const vector<Blob<Dtype>*>& top,
             const vector<bool>& propagate_down,
             const vector<Blob<Dtype>*>& bottom)
{
    ParamType<Dtype> *param = forward_init<Dtype>(base, bottom, top);
    if (!param->propagate_down) {
printf("[%s:%d]\n", __FUNCTION__, __LINE__);
        param->propagate_down = new bool[propagate_down.size()];
        for (int i = 0; i < propagate_down.size(); i++)
            param->propagate_down[i] = propagate_down[i];
    }
    else {
        for (int i = 0; i < propagate_down.size(); i++)
            CASSERT(param->propagate_down[i] == propagate_down[i]);
    }
    return param;
}
}

typedef ParamType<float> Fparam;
typedef ParamType<double> Dparam;

template <typename Dtype>
void forward_process(void *aparam)
{
  ParamType<Dtype> *param = static_cast<ParamType<Dtype> *>(aparam);
  perfpinit();
  long long perfvalues1[NUM_EVENTS];
  const Dtype* weight = param->weight;
  int bottom_hw = param->conv_in_height_ * param->conv_in_width_;
  int kernel_hw = param->kernel_h_ * param->kernel_w_;
  int out_group_size = param->conv_out_channels_ / param->group_;
  int in_group_size = param->conv_in_channels_ / param->group_;
  const Dtype* bias = param->bias;
  // For each input, ...
  for (int i = 0; i < param->bottom_size; ++i) {
    const Dtype* bottom_data = param->bottom[i];
    Dtype* top_data = param->top[i];
      // Convolution
    // For each image in input batch
    for (int nunused = 0; nunused < param->num_; ++nunused) {
      const Dtype *biasptr = bias;
      // if group_ > 1, restrict connectivity to a subset of inputs
      for (int g = 0; g < param->group_; ++g) {
        Dtype *outputp = top_data;
        const Dtype *wp_base = &weight[g * param->weight_offset_];
        // for each 'out_group', calculate convolution over input data
        for (int ounused = 0; ounused < out_group_size; ounused++) {
          const Dtype *bpg = bottom_data;
          const Dtype bias_val = bias ? *biasptr++ : 0;
          // Scan over source 2D input data
          for (int y = 0; y < param->height_out_; y++) {
            int p_limit = MIN(param->kernel_h_ - param->pad_h_,
                              param->conv_in_height_ - y * param->stride_h_);
            const Dtype *bpy = bpg;
            for (int x = 0; x < param->width_out_; x++) {
              int q_limit = MIN(param->kernel_w_ - param->pad_w_,
                                param->conv_in_width_ - x * param->stride_w_);
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
                  bpk += param->conv_in_width_;
                  wpk += param->kernel_w_;
                }
                bpx += bottom_hw;
                wpx += kernel_hw;
              }
              // Write convolution result into output (image, channel, y, x)
              *outputp++ = temp;
              bpy += param->stride_w_;
            }
            bpg += param->conv_in_width_ * param->stride_h_;
          }
          wp_base += in_group_size * kernel_hw;
          perfread(perfvalues1);
        }
        bottom_data += in_group_size * bottom_hw;
        top_data += param->output_offset_;
      }
    }
  }
#ifdef PERFSTAT
  static int jcacount = 0;
  if (jcacount++ > 300 && jcacount < 310)
    perfperf(perfvalues1, "forward");
#endif
}
template <typename Dtype>
void backward_process(void *aparam)
{
  ParamType<Dtype> *param = static_cast<ParamType<Dtype> *>(aparam);
  perfpinit();
  long long perfvalues2[NUM_EVENTS];
  const Dtype* weight = param->weight;
  int bottom_hw = param->conv_in_height_ * param->conv_in_width_;
  int kernel_hw = param->kernel_h_ * param->kernel_w_;
  int out_group_size = param->conv_out_channels_ / param->group_;
  int in_group_size = param->conv_in_channels_ / param->group_;
  int usable_height = param->conv_in_height_ + 2 * param->pad_h_ - param->kernel_h_;
  int usable_width = param->conv_in_width_ + 2 * param->pad_w_ - param->kernel_w_;
  Dtype* weight_diff = param->weight_diff;
  Dtype* bias_diff = param->bias_diff;
  if (weight_diff)
    memset(weight_diff, 0, param->weight_diff_count);
  if (bias_diff)
    memset(bias_diff, 0, param->bias_diff_count);
  // For all images
  for (int i = 0; i < param->top_size; ++i) {
    for (int n = 0; n < param->num_; ++n) {
      int boff = n * param->conv_in_channels_ * bottom_hw;
      const Dtype *top_diff_bp = param->top_diff[i]
          + n * param->conv_out_channels_ * param->conv_out_spatial_dim_;
      const Dtype *bottom_bp = param->bottom[i] + boff;
      Dtype *bottom_diff_bp = NULL;
      // Bias gradient, if necessary.
      if (bias_diff) {
        const Dtype *tptr = top_diff_bp;
        for (int j = 0; j < param->num_output_; j++)
          for (int i = 0; i < param->conv_out_spatial_dim_; i++)
            bias_diff[j] += *tptr++ * param->bias_multiplier_[i];
      }
      if (param->propagate_down[i]) {
        bottom_diff_bp =  param->bottom_diff[i] + boff;
      }
      if (weight_diff || bottom_diff_bp) {
        for (int g = 0; g < param->group_; ++g) {
          for (int cchan = 0; cchan < in_group_size; ++cchan) {
            int gchan = (g * in_group_size + cchan) * bottom_hw;
            // zero out gradient wrt bottom data, we're about to fill it
            if (bottom_diff_bp)
              memset(&bottom_diff_bp[gchan], 0, bottom_hw * sizeof(Dtype));
            for (int outindex = 0; outindex < out_group_size; ++outindex) {
              int wchan = g * param->weight_offset_ + (cchan + outindex * in_group_size) * kernel_hw;
              const Dtype *topdptr = &top_diff_bp[g * param->output_offset_ + outindex * param->conv_out_spatial_dim_];
              for (int y = 0; y <= usable_height; y += param->stride_h_){
                for (int x = 0; x <= usable_width; x += param->stride_w_) {
                  Dtype chain_grad = topdptr[(y * (usable_width + param->stride_w_) / param->stride_h_ + x) / param->stride_w_ ];
                  int pad_y = param->pad_h_ - y;
                  int pad_x = param->pad_w_ - x;
                  int p_start = MAX(0, pad_y);
                  int p_limit = MIN(param->kernel_h_, param->conv_in_height_ + pad_y);
                  int q_start = MAX(0, pad_x);
                  int q_limit = MIN(param->kernel_w_, param->conv_in_width_ + pad_x);
                  int bbase = gchan - pad_y * param->conv_in_width_ - pad_x;
                  if (chain_grad != 0.0)
                  for (int p = p_start; p < p_limit; ++p) {
                    for (int q = q_start; q < q_limit; ++q) {
                      int belement = bbase + p * param->conv_in_width_ + q;
                      int welement = wchan + p * param->kernel_w_ + q;
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
              perfread(perfvalues2);
            }
#ifdef PERFSTAT
            static int jcacount = 0;
            if (jcacount++ > 300) {
                perfperf(perfvalues2, "second");
                exit(-1);
            }
#endif
          }
        }
      }
    }
  }
}
namespace caffe {
template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  forward_init(this, bottom, top);
  forward_process<Dtype>(this->paramPtr);
#if 0
  perfpinit();
  long long perfvalues1[NUM_EVENTS];
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
          perfread(perfvalues1);
        }
        bottom_data += in_group_size * bottom_hw;
        top_data += this->output_offset_;
      }
    }
  }
#ifdef PERFSTAT
  static int jcacount = 0;
  if (jcacount++ > 300 && jcacount < 310)
    perfperf(perfvalues1, "forward");
#endif
#endif
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  backward_init(this, bottom, propagate_down, top);
  backward_process<Dtype>(this->paramPtr);
#if 0
  perfpinit();
  long long perfvalues2[NUM_EVENTS];
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int bottom_hw = this->conv_in_height_ * this->conv_in_width_;
  int kernel_hw = this->kernel_h_ * this->kernel_w_;
  int out_group_size = this->conv_out_channels_ / this->group_;
  int in_group_size = this->conv_in_channels_ / this->group_;
  int usable_height = this->conv_in_height_ + 2 * this->pad_h_ - this->kernel_h_;
  int usable_width = this->conv_in_width_ + 2 * this->pad_w_ - this->kernel_w_;
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
            // zero out gradient wrt bottom data, we're about to fill it
            if (bottom_diff_bp)
              caffe_set(bottom_hw, Dtype(0), &bottom_diff_bp[gchan]);
            for (int outindex = 0; outindex < out_group_size; ++outindex) {
              int wchan = g * this->weight_offset_ + (cchan + outindex * in_group_size) * kernel_hw;
              const Dtype *topdptr = &top_diff_bp[g * this->output_offset_ + outindex * this->conv_out_spatial_dim_];
              for (int y = 0; y <= usable_height; y += this->stride_h_){
                for (int x = 0; x <= usable_width; x += this->stride_w_) {
                  Dtype chain_grad = topdptr[(y * (usable_width + this->stride_w_) / this->stride_h_ + x) / this->stride_w_ ];
                  int pad_y = this->pad_h_ - y;
                  int pad_x = this->pad_w_ - x;
                  int p_start = MAX(0, pad_y);
                  int p_limit = MIN(this->kernel_h_, this->conv_in_height_ + pad_y);
                  int q_start = MAX(0, pad_x);
                  int q_limit = MIN(this->kernel_w_, this->conv_in_width_ + pad_x);
                  int bbase = gchan - pad_y * this->conv_in_width_ - pad_x;
                  if (chain_grad != 0.0)
                  for (int p = p_start; p < p_limit; ++p) {
                    for (int q = q_start; q < q_limit; ++q) {
                      int belement = bbase + p * this->conv_in_width_ + q;
                      int welement = wchan + p * this->kernel_w_ + q;
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
              perfread(perfvalues2);
            }
#ifdef PERFSTAT
            static int jcacount = 0;
            if (jcacount++ > 300) {
                perfperf(perfvalues2, "second");
                exit(-1);
            }
#endif
          }
        }
      }
    }
  }
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConnectalConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConnectalConvolutionLayer);

}  // namespace caffe
