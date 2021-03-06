// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/insert_splits.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename TypeParam>
class SplitLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SplitLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
  }
  virtual ~SplitLayerTest() {
    delete blob_bottom_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SplitLayerTest, TestDtypesAndDevices);

TYPED_TEST(SplitLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_a_->num(), 2);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 6);
  EXPECT_EQ(this->blob_top_a_->width(), 5);
  EXPECT_EQ(this->blob_top_b_->num(), 2);
  EXPECT_EQ(this->blob_top_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_b_->height(), 6);
  EXPECT_EQ(this->blob_top_b_->width(), 5);
}

TYPED_TEST(SplitLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    Dtype bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_a_->cpu_data()[i]);
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    Dtype bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(SplitLayerTest, TestGradientInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  this->blob_top_vec_[0] = this->blob_bottom_vec_[0];
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


class SplitLayerInsertionTest : public ::testing::Test {
 protected:
  void RunInsertionTest(
      const string& input_param_string, const string& output_param_string) {
    // Test that InsertSplits called on the proto specified by
    // input_param_string results in the proto specified by
    // output_param_string.
    NetParameter input_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        input_param_string, &input_param));
    NetParameter expected_output_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        output_param_string, &expected_output_param));
    NetParameter actual_output_param;
    InsertSplits(input_param, &actual_output_param);
    EXPECT_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
    // Also test idempotence.
    NetParameter double_split_insert_param;
    InsertSplits(actual_output_param, &double_split_insert_param);
    EXPECT_EQ(actual_output_param.DebugString(),
       double_split_insert_param.DebugString());
  }
};

TEST_F(SplitLayerInsertionTest, TestNoInsertion1) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestNoInsertion2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'data_split' "
      "  type: SPLIT "
      "  bottom: 'data' "
      "  top: 'data_split_0' "
      "  top: 'data_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_split_0' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestNoInsertionImageNet) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  name: 'data' "
      "  type: DATA "
      "  data_param { "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    mean_file: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batch_size: 256 "
      "    crop_size: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  name: 'conv1' "
      "  type: CONVOLUTION "
      "  convolution_param { "
      "    num_output: 96 "
      "    kernel_size: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  name: 'relu1' "
      "  type: RELU "
      "  bottom: 'conv1' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  name: 'pool1' "
      "  type: POOLING "
      "  pooling_param { "
      "    pool: MAX "
      "    kernel_size: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'pool1' "
      "} "
      "layers { "
      "  name: 'norm1' "
      "  type: LRN "
      "  lrn_param { "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool1' "
      "  top: 'norm1' "
      "} "
      "layers { "
      "  name: 'conv2' "
      "  type: CONVOLUTION "
      "  convolution_param { "
      "    num_output: 256 "
      "    group: 2 "
      "    kernel_size: 5 "
      "    pad: 2 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'norm1' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  name: 'relu2' "
      "  type: RELU "
      "  bottom: 'conv2' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  name: 'pool2' "
      "  type: POOLING "
      "  pooling_param { "
      "    pool: MAX "
      "    kernel_size: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv2' "
      "  top: 'pool2' "
      "} "
      "layers { "
      "  name: 'norm2' "
      "  type: LRN "
      "  lrn_param { "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool2' "
      "  top: 'norm2' "
      "} "
      "layers { "
      "  name: 'conv3' "
      "  type: CONVOLUTION "
      "  convolution_param { "
      "    num_output: 384 "
      "    kernel_size: 3 "
      "    pad: 1 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'norm2' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  name: 'relu3' "
      "  type: RELU "
      "  bottom: 'conv3' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  name: 'conv4' "
      "  type: CONVOLUTION "
      "  convolution_param { "
      "    num_output: 384 "
      "    group: 2 "
      "    kernel_size: 3 "
      "    pad: 1 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'conv3' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  name: 'relu4' "
      "  type: RELU "
      "  bottom: 'conv4' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  name: 'conv5' "
      "  type: CONVOLUTION "
      "  convolution_param { "
      "    num_output: 256 "
      "    group: 2 "
      "    kernel_size: 3 "
      "    pad: 1 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'conv4' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  name: 'relu5' "
      "  type: RELU "
      "  bottom: 'conv5' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  name: 'pool5' "
      "  type: POOLING "
      "  pooling_param { "
      "    kernel_size: 3 "
      "    pool: MAX "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv5' "
      "  top: 'pool5' "
      "} "
      "layers { "
      "  name: 'fc6' "
      "  type: INNER_PRODUCT "
      "  inner_product_param { "
      "    num_output: 4096 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.005 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'pool5' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  name: 'relu6' "
      "  type: RELU "
      "  bottom: 'fc6' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  name: 'drop6' "
      "  type: DROPOUT "
      "  dropout_param { "
      "    dropout_ratio: 0.5 "
      "  } "
      "  bottom: 'fc6' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  name: 'fc7' "
      "  type: INNER_PRODUCT "
      "  inner_product_param { "
      "    num_output: 4096 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.005 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'fc6' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  name: 'relu7' "
      "  type: RELU "
      "  bottom: 'fc7' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  name: 'drop7' "
      "  type: DROPOUT "
      "  dropout_param { "
      "    dropout_ratio: 0.5 "
      "  } "
      "  bottom: 'fc7' "
      "  top: 'fc7' "
      "} "
      "layers { "
      "  name: 'fc8' "
      "  type: INNER_PRODUCT "
      "  inner_product_param { "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "  } "
      "  blobs_lr: 1. "
      "  blobs_lr: 2. "
      "  weight_decay: 1. "
      "  weight_decay: 0. "
      "  bottom: 'fc7' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestNoInsertionWithInPlace) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'relu' "
      "  type: RELU "
      "  bottom: 'innerprod' "
      "  top: 'innerprod' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: SOFTMAX_LOSS "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestInsertion) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'innerprod3' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod3' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'data_data_0_split' "
      "  type: SPLIT "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "  top: 'data_data_0_split_2' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_data_0_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'innerprod2_innerprod2_0_split' "
      "  type: SPLIT "
      "  bottom: 'innerprod2' "
      "  top: 'innerprod2' "
      "  top: 'innerprod2_innerprod2_0_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod3' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_data_0_split_2' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2_innerprod2_0_split_1' "
      "  bottom: 'innerprod3' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TEST_F(SplitLayerInsertionTest, TestInsertionTwoTop) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'label' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'innerprod3' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'innerprod4' "
      "  type: INNER_PRODUCT "
      "  bottom: 'label' "
      "  top: 'innerprod4' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod4' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'data_data_0_split' "
      "  type: SPLIT "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "} "
      "layers: { "
      "  name: 'label_data_1_split' "
      "  type: SPLIT "
      "  bottom: 'label' "
      "  top: 'label' "
      "  top: 'label_data_1_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'label' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'innerprod3' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_data_0_split_1' "
      "  top: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'innerprod4' "
      "  type: INNER_PRODUCT "
      "  bottom: 'label_data_1_split_1' "
      "  top: 'innerprod4' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod3' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2' "
      "  bottom: 'innerprod4' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TEST_F(SplitLayerInsertionTest, TestInputInsertion) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "input: 'data' "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "input: 'data' "
      "input_dim: 10 "
      "input_dim: 3 "
      "input_dim: 227 "
      "input_dim: 227 "
      "layers: { "
      "  name: 'data_input_0_split' "
      "  type: SPLIT "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_input_0_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data_input_0_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

TEST_F(SplitLayerInsertionTest, TestWithInPlace) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'relu1' "
      "  type: RELU "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1' "
      "  bottom: 'label' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2' "
      "  bottom: 'data' "
      "} ";
  const string& expected_output_proto =
      "name: 'TestNetwork' "
      "layers: { "
      "  name: 'data' "
      "  type: DATA "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers: { "
      "  name: 'data_data_0_split' "
      "  type: SPLIT "
      "  bottom: 'data' "
      "  top: 'data' "
      "  top: 'data_data_0_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod1' "
      "  type: INNER_PRODUCT "
      "  bottom: 'data' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'relu1' "
      "  type: RELU "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "} "
      "layers: { "
      "  name: 'innerprod1_relu1_0_split' "
      "  type: SPLIT "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod1' "
      "  top: 'innerprod1_relu1_0_split_1' "
      "} "
      "layers: { "
      "  name: 'innerprod2' "
      "  type: INNER_PRODUCT "
      "  bottom: 'innerprod1' "
      "  top: 'innerprod2' "
      "} "
      "layers: { "
      "  name: 'loss1' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod1_relu1_0_split_1' "
      "  bottom: 'label' "
      "} "
      "layers: { "
      "  name: 'loss2' "
      "  type: EUCLIDEAN_LOSS "
      "  bottom: 'innerprod2' "
      "  bottom: 'data_data_0_split_1' "
      "} ";
  this->RunInsertionTest(input_proto, expected_output_proto);
}

}  // namespace caffe
