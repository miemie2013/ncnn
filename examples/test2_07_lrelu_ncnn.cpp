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

static int detect_PPYOLOE(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* param_path, const char* bin_path)
{
    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

    model.load_param(param_path);
    model.load_model(bin_path);

    // get ncnn::Mat with RGB format like PPYOLOE do.
    ncnn::Mat in_rgb = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    ncnn::Mat in_resize;
    // Interp image with cv2.INTER_CUBIC like PPYOLOE do.
    ncnn::resize_bicubic(in_rgb, in_resize, 6, 6);

    // Normalize image with the same mean and std like PPYOLOE do.
//    mean=[123.675, 116.28, 103.53]
//    std=[58.395, 57.12, 57.375]
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f/58.395f, 1.0f/57.12f, 1.0f/57.375f};
    in_resize.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in_resize);

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
    detect_PPYOLOE(m, cls_scores, param_path, bin_path);

    return 0;
}
