#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

// 析构函数，清理网络资源
AngleNet::~AngleNet() {
    net.clear();
}

// 设置线程数量
void AngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

// 初始化模型，加载模型参数和权重文件
bool AngleNet::initModel(AAssetManager *mgr) {
    int ret_param = net.load_param(mgr, "angle_op.param");
    int ret_bin = net.load_model(mgr, "angle_op.bin");
    if (ret_param != 0 || ret_bin != 0) {
        LOGE("# %d  %d", ret_param, ret_bin);
        return false;
    }
    return true;
}

// 从输出数据中计算角度得分
Angle scoreToAngle(const float *outputData, int w) {
    int maxIndex = 0;
    float maxScore = -1000.0f;
    for (int i = 0; i < w; i++) {
        if (i == 0) maxScore = outputData[i];
        else if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return {maxIndex, maxScore};
}

// 获取单个图像的角度
Angle AngleNet::getAngle(cv::Mat &src) {
    // 图像数据预处理
    ncnn::Mat input = ncnn::Mat::from_pixels(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows);
    input.substract_mean_normalize(meanValues, normValues);

    // 创建提取器并进行推理
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("out", out);

    // 计算并返回角度
    return scoreToAngle((float *) out.data, out.w);
}

// 批量处理图像，获取角度信息
std::vector<Angle>
AngleNet::getAngles(std::vector<cv::Mat> &partImgs, bool doAngle, bool mostAngle) {
    int size = partImgs.size();
    std::vector<Angle> angles(size);

    // 是否进行角度检测
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            auto angleImg = adjustTargetImg(partImgs[i], dstWidth, dstHeight);
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;
            angles[i] = angle;
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }

    // 计算最可能的角度
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {
            mostAngleIndex = 0; // 所有角度设为0
        } else {
            mostAngleIndex = 1; // 所有角度设为1
        }
        Logger("Set All Angle to mostAngleIndex(%d)", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}
