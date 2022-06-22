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

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

void pretty_print(const ncnn::Mat& m)
{
    FILE* fp = fopen("output.txt", "wb");
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                fprintf(fp, "%e,", ptr[x]);
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* param_path, const char* bin_path)
{
    ncnn::Net squeezenet;

    squeezenet.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
//    squeezenet.load_param("squeezenet_v1.1.param");
//    squeezenet.load_model("squeezenet_v1.1.bin");
    squeezenet.load_param(param_path);
    squeezenet.load_model(bin_path);

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);
//    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_RGB2BGR, bgr.cols, bgr.rows);
//    pretty_print(in);

//    mean = [117.3, 126.5, 130.2]
//    std = [108.4, 117.3, 127.6]
    const float mean_vals[3] = {117.3f, 126.5f, 130.2f};
    const float norm_vals[3] = {1.0f/108.4f, 1.0f/117.3f, 1.0f/127.6f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("images", in);

    ncnn::Mat out;
    ex.extract("output", out);
    pretty_print(out);

    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* param_path = argv[2];
    const char* bin_path = argv[3];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores, param_path, bin_path);

    return 0;
}
