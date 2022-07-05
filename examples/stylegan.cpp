// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include "datareader.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <math.h>

void pretty_print(const ncnn::Mat& m)
{
    int w = m.w;
    int h = m.h;
    int d = m.d;
    int channels = m.c;
    int size = w * h * d;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr = m.channel(q);
        for (int i = 0; i < size; i++)
        {
            float x = ptr[i];
            printf("%f ", x);
        }
        printf("------------------------\n");
    }
}

void save_data(const ncnn::Mat& m, char* name)
{
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    FILE* fp = fopen(name, "wb");
    int size = W * H * D;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int i = 0; i < size; i++)
        {
            fprintf(fp, "%e,", ptr[i]);
        }
    }
}

void print_shape(const ncnn::Mat& m, const char* name)
{
    int dims = m.dims;
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    printf("%s shape dims=%d\n", name, dims);
    printf("C=%d\n", C);
    printf("D=%d\n", D);
    printf("H=%d\n", H);
    printf("W=%d\n", W);
}



class Square : public ncnn::Layer
{
public:
    Square()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;
        int dims = bottom_top_blob.dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(x * x);
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Square)



class Rsqrt : public ncnn::Layer
{
public:
    Rsqrt()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        eps = pd.get(0, 0.f);
        scale = pd.get(1, 1.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
//                printf(" %f\n", x);
//              out = 1.f / sqrt(eps + x * scale) = 1.f / sqrt(scale) * 1.f / sqrt(eps/scale + x)
                if (scale == 1.f)
                {
                    ptr[i] = static_cast<float>(1.f / sqrt(eps + x));
                }else
                {
                    float A = static_cast<float>(1.f / sqrt(scale));
                    float B = static_cast<float>(1.f / sqrt(eps/scale + x));
                    ptr[i] = static_cast<float>(A * B);
                }
            }
        }

        return 0;
    }
public:
    float eps;
    float scale;
};

DEFINE_LAYER_CREATOR(Rsqrt)


class Lerp : public ncnn::Layer
{
public:
    Lerp()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob0 = bottom_blobs[0];
        const ncnn::Mat& bottom_blob1 = bottom_blobs[1];
        const ncnn::Mat& coeffs = bottom_blobs[2];
        int w = bottom_blob1.w;
        int h = bottom_blob1.h;
        int d = bottom_blob1.d;
        int channels = bottom_blob1.c;
        int size = w * h * d;

        size_t elemsize = bottom_blob1.elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = bottom_blob0.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            const float* coeffs_ptr = coeffs.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x0 = ptr0[i];
                float x1 = ptr1[i];
                float coeff = coeffs_ptr[0];
                out_ptr[i] = static_cast<float>(x0 + coeff * (x1 - x0));
            }
        }

        top_blob.dims = bottom_blob1.dims;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Lerp)


class StyleMixingSwitcher : public ncnn::Layer
{
public:
    StyleMixingSwitcher()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        ws_i = pd.get(0, 0);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& ws0 = bottom_blobs[0];
        const ncnn::Mat& ws1 = bottom_blobs[1];
        const ncnn::Mat& mixing = bottom_blobs[2];
        int w = ws0.w;
        int h = ws0.h;
        int d = ws0.d;
        int channels = ws0.c;
        int size = w * h * d;

        size_t elemsize = ws0.elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ws0.channel(q);
            const float* ptr1 = ws1.channel(q);
            const float* mixing_ptr = mixing.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x0 = ptr0[i];
                float x1 = ptr1[i];
                float mixing_ = mixing_ptr[ws_i];
                if (mixing_ > 0.5f)
                {
                    out_ptr[i] = x1;
                }else
                {
                    out_ptr[i] = x0;
                }
            }
        }

        top_blob.dims = ws0.dims;

        return 0;
    }
public:
    int ws_i;
};

DEFINE_LAYER_CREATOR(StyleMixingSwitcher)


// refer to Convolution layer.
class Shell : public ncnn::Layer
{
public:
    Shell()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        C = pd.get(2, 1);
        D = pd.get(11, 1);
        H = pd.get(1, 1);
        W = pd.get(0, 1);
        target_dims = pd.get(3, 4);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
            return -100;

        if (bias_term)
        {
            bias_data = mb.load(C, 1);
            if (bias_data.empty())
                return -100;
        }
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // refer to Split layer.
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const size_t elemsize = bottom_blob.elemsize;

        top_blobs[0].create(W, H, D, C, elemsize, opt.blob_allocator);
        if (top_blobs[0].empty())
            return -100;
        top_blobs[0] = weight_data;
//        print_shape(top_blobs[0], "top_blobs[0]");


        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Reshape);

        // set param
        ncnn::ParamDict pd;
        pd.set(2, C);
        pd.set(11, D);
        pd.set(1, H);
        pd.set(0, W);
        op->load_param(pd);
        op->create_pipeline(opt);
        op->forward(top_blobs[0], top_blobs[0], opt);
        op->destroy_pipeline(opt);
        delete op;

        top_blobs[0].dims = target_dims;

        if (bias_term)
        {
            top_blobs[1] = bias_data;
        }
//        print_shape(top_blobs[0], "top_blobs[0]");
//        print_shape(bias_data, "bias_data");
        return 0;
    }
public:
    // param
    int C;
    int D;
    int H;
    int W;
    int target_dims;
    int bias_term;

    int weight_data_size;

    // model
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

DEFINE_LAYER_CREATOR(Shell)


class Fmatmul : public ncnn::Layer
{
public:
    Fmatmul()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        bias_term = pd.get(1, 0);
        weight_data_size = pd.get(2, 0);
        int8_scale_term = pd.get(8, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());

//        if (int8_scale_term)
//        {
//#if NCNN_INT8
//            support_int8_storage = true;
//#else
//            NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
//            return -1;
//#endif
//        }

        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // call InnerProduct
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, bias_term);
        pd.set(2, weight_data_size);
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        ncnn::Mat weights[4];
        weights[0] = bottom_blobs[1];
        if (bias_term)
        {
            weights[1] = bottom_blobs[2];
        }

//#if NCNN_INT8
//        if (int8_scale_term)
//        {
//            weights[2] = weight_data_int8_scales;
//            weights[3] = bottom_blob_int8_scales;
//        }
//#endif

        op->load_model(ncnn::ModelBinFromMatArray(weights));

        op->create_pipeline(opt);

        // forward
        op->forward(bottom_blobs[0], top_blobs[0], opt);

        op->destroy_pipeline(opt);

        delete op;

        return 0;
    }
public:
    // param
    int num_output;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    ncnn::Mat activation_params;
//#if NCNN_INT8
//    ncnn::Mat weight_data_int8_scales;
//    ncnn::Mat bottom_blob_int8_scales;
//#endif
};

DEFINE_LAYER_CREATOR(Fmatmul)


class BiasAct : public ncnn::Layer
{
public:
    BiasAct()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        act_type = pd.get(0, 0);
        alpha = pd.get(1, 0.f);
        gain = pd.get(2, 0.f);
        clamp = pd.get(3, 0.f);
        return 0;
    }

    float bias_act(float x, float bias, float elu_alpha, float selu_alpha, float selu_lambda, float alphaxlambda) const
    {
        x = x + bias;
        if (act_type == 1)
        {  // relu
            if (x < 0)
                x = 0.f;
        }else if (act_type == 2)
        {  // lrelu
            if (x < 0)
                x *= alpha;
        }else if (act_type == 3)
        {  // tanh
            x = static_cast<float>(tanh(x));
        }else if (act_type == 4)
        {  // sigmoid
            x = static_cast<float>(1.f / (1.f + expf(-x)));
        }else if (act_type == 5)
        {  // elu
            if (x < 0.f)
                x = static_cast<float>(elu_alpha * (exp(x) - 1.f));
        }else if (act_type == 6)
        {  // selu
            if (x < 0.f)
                x = static_cast<float>((exp(x) - 1.f) * alphaxlambda);
            else
                x *= selu_alpha;
        }else if (act_type == 7)
        {  // softplus
            x = log(exp(x) + 1.0f);
        }else if (act_type == 8)
        {  // swish
            x = static_cast<float>(x / (1.f + expf(-x)));
        }

        // scale
        if (gain != 1.f)
            x *= gain;
        // clamp
        if (clamp >= 0.0f)
            x = std::min(clamp, std::max(-clamp, x));
        return x;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& bias_data = bottom_blobs[1];
        // miemie2013: bottom_blob.dims may be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: bias_data.dims must be 1 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "bottom_blob");
//        print_shape(bias_data, "bias_data");

        int b_w = bias_data.w;
        int b_h = bias_data.h;
        int b_d = bias_data.d;
        int b_channels = bias_data.c;

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;

        int _111W_array_add_111W_bias = 0;
        int _C1HW_array_add_111W_bias = 0;
        if (channels == 1 && d == 1 && h == 1 && w != 1) {
            if (b_channels == 1 && b_d == 1 && b_h == 1 && b_w != 1) {
                _111W_array_add_111W_bias = 1;
            }
        }else if (channels != 1 && d == 1 && h != 1 && w != 1) {
            if (b_channels == 1 && b_d == 1 && b_h == 1 && b_w != 1) {
                _C1HW_array_add_111W_bias = 1;
            }
        }else {
            printf("not implemented.\n");
            return -100;
        }

        size_t elemsize = bottom_blobs[0].elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float elu_alpha = 0.1f;
        float selu_alpha = 1.67326324f;
        float selu_lambda = 1.050700987f;
        float alphaxlambda = selu_alpha * selu_lambda;


        if (_111W_array_add_111W_bias)
        {
            const float* bias_ptr = bias_data.channel(0);
            const float* in_ptr = bottom_blob.channel(0);
            float* out_ptr = top_blob.channel(0);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                float bias = bias_data[i];
                float x = in_ptr[i];
                x = bias_act(x, bias, elu_alpha, selu_alpha, selu_lambda, alphaxlambda);
                out_ptr[i] = x;
            }
            // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
            top_blobs[0].dims = 1;
            return 0;
        }else if (_C1HW_array_add_111W_bias)
        {
            const float* bias_ptr = bias_data.channel(0);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* in_ptr = bottom_blob.channel(q);
                float* out_ptr = top_blob.channel(q);
                float bias = bias_ptr[q];

                for (int i = 0; i < size; i++)
                {
                    float x = in_ptr[i];
                    x = bias_act(x, bias, elu_alpha, selu_alpha, selu_lambda, alphaxlambda);
                    out_ptr[i] = x;
                }
            }
            // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
            top_blobs[0].dims = bottom_blob.dims;
            return 0;
        }else {
            printf("not implemented.\n");
            return -100;
        }
        return 0;
    }
public:
    // param
    int act_type;
    float alpha;
    float gain;
    float clamp;
};

DEFINE_LAYER_CREATOR(BiasAct)



class F4DMul1D : public ncnn::Layer
{
public:
    F4DMul1D()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        dim = pd.get(0, 0);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& bias_data = bottom_blobs[1];

        int b_w = bias_data.w;
        int b_h = bias_data.h;
        int b_d = bias_data.d;
        int b_channels = bias_data.c;
        if (b_channels == 1 && b_d == 1 && b_h == 1)
        {
            ;
        }else {
            printf("not implemented.\n");
            return -100;
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;
        int wh = w * h;


        int in_C = b_w;
        int out_C = channels / in_C;

        size_t elemsize = bottom_blobs[0].elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (dim == 1)
        {
            const float* bias_ptr = bias_data.channel(0);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* in_ptr = bottom_blob.channel(q);
                float* out_ptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float bias = bias_ptr[i / wh];
                    float x = in_ptr[i] * bias;
                    out_ptr[i] = x;
                }
            }
            return 0;
        }else if (dim == 0)
        {
            const float* bias_ptr = bias_data.channel(0);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* in_ptr = bottom_blob.channel(q);
                float* out_ptr = top_blob.channel(q);
                float bias = bias_ptr[q];

                for (int i = 0; i < size; i++)
                {
                    float x = in_ptr[i] * bias;
                    out_ptr[i] = x;
                }
            }
            return 0;
        }else {
            printf("not implemented.\n");
            return -100;
        }
        return 0;
    }
public:
    // param
    int dim;
};

DEFINE_LAYER_CREATOR(F4DMul1D)



class AddNoise : public ncnn::Layer
{
public:
    AddNoise()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& noise_data = bottom_blobs[1];
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: noise_data.dims must be 4 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "bottom_blob");
//        print_shape(noise_data, "noise_data");

        int b_w = noise_data.w;
        int b_h = noise_data.h;
        int b_d = noise_data.d;
        int b_channels = noise_data.c;

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;


        int in_C = b_w;
        int out_C = channels / in_C;

        size_t elemsize = bottom_blobs[0].elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* noise_ptr = noise_data.channel(0);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float bias = noise_ptr[i];
                float x = in_ptr[i] + bias;
                out_ptr[i] = x;
            }
        }
        // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
        top_blobs[0].dims = bottom_blob.dims;
        return 0;
    }
};

DEFINE_LAYER_CREATOR(AddNoise)


class MulConstant : public ncnn::Layer
{
public:
    MulConstant()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        scale = pd.get(0, 1.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(scale * x);
            }
        }

        return 0;
    }
public:
    float scale;
};

DEFINE_LAYER_CREATOR(MulConstant)



class FconvTranspose2d : public ncnn::Layer
{
public:
    FconvTranspose2d()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        num_input = pd.get(31, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        pad_left = pd.get(4, 0);
        pad_right = pd.get(15, pad_left);
        pad_top = pd.get(14, pad_left);
        pad_bottom = pd.get(16, pad_top);
        output_pad_right = pd.get(18, 0);
        output_pad_bottom = pd.get(19, output_pad_right);
        output_w = pd.get(20, 0);
        output_h = pd.get(21, output_w);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& weight_data = bottom_blobs[1];
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: weight_data.dims must be 4 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "FconvTranspose2d bottom_blob");
//        print_shape(weight_data, "FconvTranspose2d weight_data");

        // miemie2013: In ncnn's Deconvolution Layer, weight must be flattened.
        ncnn::Mat weight_data2 = weight_data.reshape(kernel_w*kernel_h*num_input*num_output, 1, 1, 1);
        weight_data2.dims = 1;
//        print_shape(weight_data2, "FconvTranspose2d weight_data");

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, pad_left);
        pd.set(15, pad_right);
        pd.set(14, pad_top);
        pd.set(16, pad_bottom);
        pd.set(18, output_pad_right);
        pd.set(19, output_pad_bottom);
        pd.set(20, output_w);
        pd.set(21, output_h);
        pd.set(5, bias_term);
        pd.set(6, weight_data_size);
        pd.set(9, activation_type);
        pd.set(10, activation_params);
        op->load_param(pd);

        // set weights
        ncnn::Mat weights[2];
        weights[0] = weight_data2;
        if (bias_term)
        {
            weights[1] = bottom_blobs[2];
        }
        op->load_model(ncnn::ModelBinFromMatArray(weights));

        op->create_pipeline(opt);
        op->forward(bottom_blob, top_blobs[0], opt);
        op->destroy_pipeline(opt);
        delete op;
        return 0;
    }
public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int output_pad_right;
    int output_pad_bottom;
    int output_w;
    int output_h;
    int bias_term;
    int num_input;

    int weight_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    ncnn::Mat activation_params;
};

DEFINE_LAYER_CREATOR(FconvTranspose2d)


class Down2 : public ncnn::Layer
{
public:
    Down2()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
//        print_shape(bottom_blob, "Down2 bottom_blob");
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = in_W / 2;
        int out_H = in_H / 2;
        int out_C = in_C;
        if (in_W % 2 == 1)
            out_W++;
        if (in_H % 2 == 1)
            out_H++;

        top_blob.create(out_W, out_H, out_C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // miemie2013:
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_C; p++)
        {
            const float* input_ptr = bottom_blob.channel(p);
            float* output_ptr = top_blob.channel(p);

            for (int i = 0; i < out_H; i++)
            {
                for (int j = 0; j < out_W; j++)
                {
                    output_ptr[i * out_W + j] = input_ptr[i * 2 * in_W + j * 2];
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Down2)



class Up2 : public ncnn::Layer
{
public:
    Up2()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
//        print_shape(bottom_blob, "Up2 bottom_blob");
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = in_W * 2;
        int out_H = in_H * 2;
        int out_C = in_C;

        top_blob.create(out_W, out_H, out_C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // miemie2013:
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_C; p++)
        {
            const float* input_ptr = bottom_blob.channel(p);
            float* output_ptr = top_blob.channel(p);

            for (int i = 0; i < out_H; i++)
            {
                for (int j = 0; j < out_W; j++)
                {
                    if (i % 2 == 0)
                    {
                        if (j % 2 == 0)
                        {
                            output_ptr[i * out_W + j] = input_ptr[i / 2 * in_W + j / 2];
                        }else if (j % 2 == 1)
                        {
                            output_ptr[i * out_W + j] = 0.f;
                        }
                    }else if (i % 2 == 1)
                    {
                        output_ptr[i * out_W + j] = 0.f;
                    }
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Up2)


class StyleganPost : public ncnn::Layer
{
public:
    StyleganPost()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;
        int dims = bottom_top_blob.dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                x = x * 127.5f + 128.f;
                x = std::min(255.f, std::max(0.f, x));
                ptr[i] = x;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(StyleganPost)




static int stylegan_gen(const char* z0_path, const char* mapping_param_path, const char* mapping_bin_path, const char* synthesis_param_path, const char* synthesis_bin_path, int z_dim, int num_ws, double ws_coeff)
{
    // get input z.
    FILE* fp = fopen(z0_path, "rb");
    if (!fp)
    {
        printf("fopen %s failed", z0_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    ncnn::Mat z = mb.load(z_dim, 0);
    fclose(fp);

    print_shape(z, "z");

    ncnn::Net mapping;
    ncnn::Net synthesis;

//    mapping.opt.use_vulkan_compute = true;
//    synthesis.opt.use_vulkan_compute = true;

    mapping.opt.use_vulkan_compute = false;
    mapping.opt.use_fp16_storage = false;
    mapping.opt.use_fp16_packed = false;
    mapping.opt.use_fp16_storage = false;
    mapping.opt.use_fp16_arithmetic = false;
    synthesis.opt.use_vulkan_compute = false;
    synthesis.opt.use_fp16_storage = false;
    synthesis.opt.use_fp16_packed = false;
    synthesis.opt.use_fp16_storage = false;
    synthesis.opt.use_fp16_arithmetic = false;

    mapping.register_custom_layer("Square", Square_layer_creator);
    mapping.register_custom_layer("Rsqrt", Rsqrt_layer_creator);
    mapping.register_custom_layer("Lerp", Lerp_layer_creator);
    mapping.register_custom_layer("Shell", Shell_layer_creator);
    mapping.register_custom_layer("Fmatmul", Fmatmul_layer_creator);
    mapping.register_custom_layer("BiasAct", BiasAct_layer_creator);
    mapping.register_custom_layer("F4DMul1D", F4DMul1D_layer_creator);
    mapping.register_custom_layer("AddNoise", AddNoise_layer_creator);
    mapping.register_custom_layer("MulConstant", MulConstant_layer_creator);
    mapping.register_custom_layer("FconvTranspose2d", FconvTranspose2d_layer_creator);
    mapping.register_custom_layer("Down2", Down2_layer_creator);
    mapping.register_custom_layer("Up2", Up2_layer_creator);

    synthesis.register_custom_layer("Square", Square_layer_creator);
    synthesis.register_custom_layer("Rsqrt", Rsqrt_layer_creator);
    synthesis.register_custom_layer("StyleMixingSwitcher", StyleMixingSwitcher_layer_creator);
    synthesis.register_custom_layer("Shell", Shell_layer_creator);
    synthesis.register_custom_layer("Fmatmul", Fmatmul_layer_creator);
    synthesis.register_custom_layer("BiasAct", BiasAct_layer_creator);
    synthesis.register_custom_layer("F4DMul1D", F4DMul1D_layer_creator);
    synthesis.register_custom_layer("AddNoise", AddNoise_layer_creator);
    synthesis.register_custom_layer("MulConstant", MulConstant_layer_creator);
    synthesis.register_custom_layer("FconvTranspose2d", FconvTranspose2d_layer_creator);
    synthesis.register_custom_layer("Down2", Down2_layer_creator);
    synthesis.register_custom_layer("Up2", Up2_layer_creator);
    synthesis.register_custom_layer("StyleganPost", StyleganPost_layer_creator);

    mapping.load_param(mapping_param_path);
    mapping.load_model(mapping_bin_path);
    synthesis.load_param(synthesis_param_path);
    synthesis.load_model(synthesis_bin_path);

    ncnn::Extractor mapping_ex0 = mapping.create_extractor();
    ncnn::Extractor synthesis_ex = synthesis.create_extractor();

    float* coeff_data = new float[1];
    coeff_data[0] = (float)ws_coeff;
    ncnn::Mat coeff(1, coeff_data);

    mapping_ex0.input("z", z);
    mapping_ex0.input("coeff", coeff);
    ncnn::Mat ws0;
    mapping_ex0.extract("output", ws0);
    print_shape(ws0, "ws0");

    float* mixing_data = new float[num_ws];
    for(int i=0;i<num_ws;i++)
        mixing_data[i] = 0.0f;
    ncnn::Mat mixing(num_ws, mixing_data);

    synthesis_ex.input("ws0", ws0);
    synthesis_ex.input("ws1", ws0);
    synthesis_ex.input("mixing", mixing);
    ncnn::Mat fake_img;
    synthesis_ex.extract("output", fake_img);
    print_shape(fake_img, "fake_img");
    save_data(fake_img, "output.txt");


    // ncnn to opencv
    // ncnn::Mat in(w, h, 3);
    cv::Mat fake_img_(fake_img.h, fake_img.w, CV_8UC3);
    fake_img.to_pixels(fake_img_.data, ncnn::Mat::PIXEL_BGR2RGB);

    cv::imshow("fake_img", fake_img_);
    cv::waitKey(0);

    return 0;
}



static int style_mixing(const char* z0_path, const char* z1_path, const char* mapping_param_path, const char* mapping_bin_path, const char* synthesis_param_path, const char* synthesis_bin_path, int z_dim, int num_ws, double ws_coeff, int* mixing_ws_idx, int mixing_ws_num)
{
    // get input z.
    FILE* fp = fopen(z0_path, "rb");
    if (!fp)
    {
        printf("fopen %s failed", z0_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    ncnn::Mat z = mb.load(z_dim, 0);
    fclose(fp);

    FILE* fp2 = fopen(z1_path, "rb");
    if (!fp2)
    {
        printf("fopen %s failed", z1_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr2(fp2);
    ncnn::ModelBinFromDataReader mb2(dr2);
    ncnn::Mat z2 = mb2.load(z_dim, 0);
    fclose(fp2);

    print_shape(z, "z");

    ncnn::Net mapping;
    ncnn::Net synthesis;

//    mapping.opt.use_vulkan_compute = true;
//    synthesis.opt.use_vulkan_compute = true;

    mapping.opt.use_vulkan_compute = false;
    mapping.opt.use_fp16_storage = false;
    mapping.opt.use_fp16_packed = false;
    mapping.opt.use_fp16_storage = false;
    mapping.opt.use_fp16_arithmetic = false;
    synthesis.opt.use_vulkan_compute = false;
    synthesis.opt.use_fp16_storage = false;
    synthesis.opt.use_fp16_packed = false;
    synthesis.opt.use_fp16_storage = false;
    synthesis.opt.use_fp16_arithmetic = false;

    mapping.register_custom_layer("Square", Square_layer_creator);
    mapping.register_custom_layer("Rsqrt", Rsqrt_layer_creator);
    mapping.register_custom_layer("Lerp", Lerp_layer_creator);
    mapping.register_custom_layer("Shell", Shell_layer_creator);
    mapping.register_custom_layer("Fmatmul", Fmatmul_layer_creator);
    mapping.register_custom_layer("BiasAct", BiasAct_layer_creator);
    mapping.register_custom_layer("F4DMul1D", F4DMul1D_layer_creator);
    mapping.register_custom_layer("AddNoise", AddNoise_layer_creator);
    mapping.register_custom_layer("MulConstant", MulConstant_layer_creator);
    mapping.register_custom_layer("FconvTranspose2d", FconvTranspose2d_layer_creator);
    mapping.register_custom_layer("Down2", Down2_layer_creator);
    mapping.register_custom_layer("Up2", Up2_layer_creator);

    synthesis.register_custom_layer("Square", Square_layer_creator);
    synthesis.register_custom_layer("Rsqrt", Rsqrt_layer_creator);
    synthesis.register_custom_layer("StyleMixingSwitcher", StyleMixingSwitcher_layer_creator);
    synthesis.register_custom_layer("Shell", Shell_layer_creator);
    synthesis.register_custom_layer("Fmatmul", Fmatmul_layer_creator);
    synthesis.register_custom_layer("BiasAct", BiasAct_layer_creator);
    synthesis.register_custom_layer("F4DMul1D", F4DMul1D_layer_creator);
    synthesis.register_custom_layer("AddNoise", AddNoise_layer_creator);
    synthesis.register_custom_layer("MulConstant", MulConstant_layer_creator);
    synthesis.register_custom_layer("FconvTranspose2d", FconvTranspose2d_layer_creator);
    synthesis.register_custom_layer("Down2", Down2_layer_creator);
    synthesis.register_custom_layer("Up2", Up2_layer_creator);
    synthesis.register_custom_layer("StyleganPost", StyleganPost_layer_creator);

    mapping.load_param(mapping_param_path);
    mapping.load_model(mapping_bin_path);
    synthesis.load_param(synthesis_param_path);
    synthesis.load_model(synthesis_bin_path);

    ncnn::Extractor mapping_ex0 = mapping.create_extractor();
    ncnn::Extractor mapping_ex1 = mapping.create_extractor();
    ncnn::Extractor synthesis_ex = synthesis.create_extractor();

    float* coeff_data = new float[1];
    coeff_data[0] = (float)ws_coeff;
    ncnn::Mat coeff(1, coeff_data);

    mapping_ex0.input("z", z);
    mapping_ex0.input("coeff", coeff);
    ncnn::Mat ws0;
    mapping_ex0.extract("output", ws0);
    print_shape(ws0, "ws0");

    mapping_ex1.input("z", z2);
    mapping_ex1.input("coeff", coeff);
    ncnn::Mat ws1;
    mapping_ex1.extract("output", ws1);
    print_shape(ws1, "ws1");

    float* mixing_data = new float[num_ws];
    for(int i=0;i<num_ws;i++)
        mixing_data[i] = 0.0f;
    for(int i=0; i<mixing_ws_num; i++)
        mixing_data[mixing_ws_idx[i]] = 1.0f;
    ncnn::Mat mixing(num_ws, mixing_data);

    synthesis_ex.input("ws0", ws0);
    synthesis_ex.input("ws1", ws1);
    synthesis_ex.input("mixing", mixing);
    ncnn::Mat fake_img;
    synthesis_ex.extract("output", fake_img);
    print_shape(fake_img, "fake_img");


    // ncnn to opencv
    // ncnn::Mat in(w, h, 3);
    cv::Mat fake_img_(fake_img.h, fake_img.w, CV_8UC3);
    fake_img.to_pixels(fake_img_.data, ncnn::Mat::PIXEL_BGR2RGB);

    cv::imshow("fake_img", fake_img_);
    cv::waitKey(0);

    return 0;
}


int main(int argc, char** argv)
{
//    if (argc != 7)
//    {
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        return -1;
//    }

    // 0 means image gen; 1 means style mixing;
    int func_id = atoi(argv[1]);
    int z_dim = atoi(argv[2]);
    int num_ws = atoi(argv[3]);
    double ws_coeff = atof(argv[4]);

    if (func_id == 0)
    {
        const char* z_path = argv[5];
        const char* mapping_param_path = argv[6];
        const char* mapping_bin_path = argv[7];
        const char* synthesis_param_path = argv[8];
        const char* synthesis_bin_path = argv[9];
        stylegan_gen(z_path, mapping_param_path, mapping_bin_path, synthesis_param_path, synthesis_bin_path, z_dim, num_ws, ws_coeff);
    }else if (func_id == 1)
    {
        const char* z0_path = argv[5];
        const char* z1_path = argv[6];
        const char* mapping_param_path = argv[7];
        const char* mapping_bin_path = argv[8];
        const char* synthesis_param_path = argv[9];
        const char* synthesis_bin_path = argv[10];
        int mixing_ws_num = argc - 11;
        int* mixing_ws_idx = new int[mixing_ws_num];
        for(int i=0; i<mixing_ws_num; i++)
            mixing_ws_idx[i] = atoi(argv[11 + i]);
        style_mixing(z0_path, z1_path, mapping_param_path, mapping_bin_path, synthesis_param_path, synthesis_bin_path, z_dim, num_ws, ws_coeff, mixing_ws_idx, mixing_ws_num);
    }

    return 0;
}
