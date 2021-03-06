// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
                        width_, kernel_size_, pad_, stride_, col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            bias_multiplier_->gpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_->gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_gpu_data();
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                   width_, kernel_size_, pad_, stride_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down[i]) {
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
              stride_, bottom_diff + (*bottom)[i]->offset(n));
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void CopyDataFrom(const int nthreads, Dtype* dst, const Dtype* src){
	CUDA_KERNEL_LOOP(index, nthreads) {
		*(dst+index) = *(src+index);
	}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Reconstruct_gpu(const vector<Blob<Dtype>*>& top,
     vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  weight = this->blobs_[0]->gpu_data();
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
	  const Dtype* top_diff = NULL;
	  top_diff = top[i]->gpu_diff();
	  Blob<Dtype>*  top_t= new Blob<Dtype>(top[i]->num(), top[i]->channels(), top[i]->height(), top[i]->width());
	  Dtype* top_t_data = top_t->mutable_gpu_data();
	  CopyDataFrom<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(top[i]->count(), top_t_data, top_diff);
	  Dtype* col_data = col_buffer_.mutable_gpu_data();
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
	  for (int n = 0; n < num_; ++n) {
		  if (bias_term_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
				N_, 1, (Dtype)0., this->blobs_[1]->gpu_data(),
				bias_multiplier_->gpu_data(),
				(Dtype)1., top_t_data + top_t->offset(n));
		  }
		  im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                   width_, kernel_size_, pad_, stride_, col_data);
		  for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_t_data + top_t->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
		  col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
              stride_, bottom_diff + (*bottom)[i]->offset(n));
	  }
	  delete top_t;
  }
}
/*
template <typename Dtype>
void ConvolutionLayer<Dtype>::SetTop_gpu(vector<Blob<Dtype>*>* bottom){
	int num = (*bottom)[0]->num();
	int channels = (*bottom)[0]->channels();
	int height = (*bottom)[0]->height();
	int width = (*bottom)[0]->width();
	
	for(int n=0;n<num;++n){
		const Dtype* feature_data = (*bottom)[0]->cpu_data()+(*bottom)[0]->offset(n);
		Dtype* feature_diff = (*bottom)[0]->mutable_cpu_diff()+(*bottom)[0]->offset(n);
		int x=0, y=0, ch=0;
		Dtype maxact = feature_data[0]-1;
		for(int c=0;c<channels;++c){
			for(int h=0;h<height;++h){
				for(int w=0;w<width;++w){
					int pixelIdx = (c * height + h ) * width + w;
					feature_diff[pixelIdx]= Dtype(0.);
					if(maxact < feature_data[pixelIdx]){
						x = h;
						y = w;
						ch = c;
						maxact = feature_data[pixelIdx];
					}
				}
			}
		}
		int maxIdx = (ch * height + x) * width + y;
		feature_diff[maxIdx] = maxact;
	}
}
*/
INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
