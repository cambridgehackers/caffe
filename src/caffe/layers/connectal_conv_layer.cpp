#define DECLARE_CONNECTAL_CONV
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
    typedef CPtr dptr;
    typedef CPtr cdptr;
    ParamType<Dtype> *param = static_cast<ParamType<Dtype> *>(base->paramPtr);
    if (param) {
        return param;
    }
    param = static_cast<ParamType<Dtype> *>(connectal_conv_library_param(sizeof(Dtype)));
    base->paramPtr = param;
    param->top_size = top.size();
    param->bottom_size = bottom.size();
    param->bottom = new cdptr[param->bottom_size];
    param->top = new dptr[param->top_size];
    param->top_diff = new cdptr[param->top_size];
    param->bottom_diff = new dptr[param->bottom_size];
    param->weight_diff_count = base->blobs_[0]->count() * sizeof(Dtype);
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

    // memory
    size_t len = 0, currentIndex = 0;
#define ROUNDLEN(A) ((((A)->size() + sizeof(double) - 1)/sizeof(double)) * sizeof(double))
    len += ROUNDLEN(base->blobs_[0]->data());
    len += ROUNDLEN(base->blobs_[0]->diff());
    for (int i = 0; i < param->bottom_size; i++) {
        len += ROUNDLEN(bottom[i]->data());
        len += ROUNDLEN(bottom[i]->diff());
    }
    for (int i = 0; i < param->top_size; i++) {
        len += ROUNDLEN(top[i]->data());
        len += ROUNDLEN(top[i]->diff());
    }
    if (base->bias_term_) {
        len += ROUNDLEN(base->blobs_[1]->data());
        len += ROUNDLEN(base->blobs_[1]->diff());
        len += ROUNDLEN(base->bias_multiplier_.data());
    }
    //
    len++;
    param->basePtr = (uint8_t *)alloc_portal_memory(len * sizeof(Dtype), 1/*cached*/, &param->portalFd_);
    currentIndex += sizeof(Dtype);
printf("[%s:%d] len %ld\n", __FUNCTION__, __LINE__, (long)len);
#define PTRINC(TARGET, A) { \
        (A)->set_cpu_data(CACCESS(currentIndex)); \
        TARGET = currentIndex;\
        currentIndex += ROUNDLEN(A); \
        }
    PTRINC(param->weight, base->blobs_[0]->data());
    PTRINC(param->weight_diff, base->blobs_[0]->diff());
    for (int i = 0; i < param->bottom_size; i++) {
        PTRINC(param->bottom[i], bottom[i]->data());
        PTRINC(param->bottom_diff[i], bottom[i]->diff());
    }
    for (int i = 0; i < param->top_size; i++) {
        PTRINC(param->top[i], top[i]->data());
        PTRINC(param->top_diff[i], top[i]->diff());
    }
    if (base->bias_term_) {
        PTRINC(param->bias, base->blobs_[1]->data());
        PTRINC(param->bias_diff, base->blobs_[1]->diff());
        PTRINC(param->bias_multiplier_, base->bias_multiplier_.data());
    }
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
  ParamType<Dtype> *param = paramStructInit<Dtype>(this, bottom, top);
  if (!param->propdone_) {
      param->propdone_ = 1;
      for (int i = 0; i < param->bottom_size; i++) {
          if (!propagate_down[i])
              param->bottom_diff[i] = 0;
      }
  }
  param->backward_process();
}

#ifdef CPU_ONLY
STUB_GPU(ConnectalConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConnectalConvolutionLayer);

}  // namespace caffe
