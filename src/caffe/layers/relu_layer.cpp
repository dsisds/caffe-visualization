// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
Dtype ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = max(bottom_data[i], Dtype(0));
  }
  return Dtype(0);
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::SetTop_cpu(vector<Blob<Dtype>*>* bottom){
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

template <typename Dtype>
void ReLULayer<Dtype>::Reconstruct_cpu(const vector<Blob<Dtype>*>& top,
	vector<Blob<Dtype>*>* bottom){
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		for(int i=0;i<count;++i){
			bottom_diff[i] = max(top_diff[i], Dtype(0));
		}
}


INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
