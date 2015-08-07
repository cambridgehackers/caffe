#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <assert.h>
#include "connectal_conv.h"

#define CASSERT assert
namespace caffe {
template <typename Dtype>
    ParamType<Dtype> *paramStructInit(ConnectalConvolutionLayer<Dtype> *base,
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
        CASSERT(param->weight_offset_ == base->weight_offset_);
        CASSERT(param->pad_h_ == base->pad_h_);
        CASSERT(param->pad_w_ == base->pad_w_);
        CASSERT(param->stride_h_ == base->stride_h_);
        CASSERT(param->stride_w_ == base->stride_w_);
        // legacy
        CASSERT(param->col_buffer_ == base->col_buffer_.mutable_cpu_data());
        CASSERT(param->is_1x1_ == base->is_1x1_);
        CASSERT(param->bottom_mult == bottom[0]->offset(1));
        CASSERT(param->top_mult == top[0]->offset(1));
        CASSERT(param->param_propagate_down_[0] == base->param_propagate_down_[0]);
        CASSERT(param->param_propagate_down_[1] == base->param_propagate_down_[1]);
        return param;
    }
    param = static_cast<ParamType<Dtype> *>(init_connectal_conv_library(sizeof(Dtype)));
    base->paramPtr = param;
    param->weight = base->blobs_[0]->cpu_data();
    param->top_size = top.size();
    param->bottom_size = bottom.size();
    param->bottom = new cdptr[param->bottom_size];
    for (int i = 0; i < param->bottom_size; i++) {
         param->bottom[i] = bottom[i]->cpu_data();
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
    param->weight_offset_ = base->weight_offset_;
    param->pad_h_ = base->pad_h_;
    param->pad_w_ = base->pad_w_;
    param->stride_h_ = base->stride_h_;
    param->stride_w_ = base->stride_w_;
    // legacy
    param->col_buffer_ = base->col_buffer_.mutable_cpu_data();
    param->is_1x1_ = base->is_1x1_;
    param->bottom_mult = bottom[0]->offset(1);
    param->top_mult = top[0]->offset(1);
    param->param_propagate_down_[0] = base->param_propagate_down_[0];
    param->param_propagate_down_[1] = base->param_propagate_down_[1];
    param->col_offset_ = base->col_offset_;
    return param;
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  ParamType<Dtype> *param = paramStructInit<Dtype>(this, bottom, top);
  param->forward_process();
}

template <typename Dtype>
void ConnectalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  typedef Dtype *dptr;
  ParamType<Dtype> *param = paramStructInit<Dtype>(this, bottom, top);
  if (!param->bottom_diff) {
      param->bottom_diff = new dptr[param->bottom_size];
      for (int i = 0; i < param->bottom_size; i++) {
          if (propagate_down[i])
              param->bottom_diff[i] = bottom[i]->mutable_cpu_diff();
      }
  }
  param->backward_process();
}

#ifdef CPU_ONLY
STUB_GPU(ConnectalConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConnectalConvolutionLayer);

}  // namespace caffe
