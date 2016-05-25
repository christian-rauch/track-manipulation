#ifndef LCM_DEPTH_PROVIDER_H
#define LCM_DEPTH_PROVIDER_H

#include <depth_sources/depth_source.h>
#include <util/mirrored_memory.h>

#include <lcm/lcm-cpp.hpp>
#include <bot_core/images_t.hpp>

struct StereoCameraParameter {
    float2 focal_length;    // f_x, f_y in pxl
    float2 camera_center;   // c_x, c_y in pxl
    float baseline;         // b in meter
    uint64_t width;         // image width in pxl
    uint64_t height;        // image height in pxl
};

template <typename DepthType, typename ColorType>
class LCM_DepthSource : public dart::DepthSource<DepthType,ColorType> {
private:
    lcm::LCM *lcm = NULL;

#ifdef CUDA_BUILD
    dart::MirroredVector<DepthType> * _depthData;
#else
    DepthType * _depthData;
#endif // CUDA_BUILD

    uint64_t _depthTime;

    StereoCameraParameter _cam_param;

    float _ScaleToMeters;

public:
    LCM_DepthSource(const StereoCameraParameter &param, const float scale = 1.0);

    ~LCM_DepthSource();

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const;

#ifdef CUDA_BUILD
    const DepthType * getDepth() const { return _depthData->hostPtr(); }
    const DepthType * getDeviceDepth() const { return _depthData->devicePtr(); }
#else
    const DepthType * getDepth() const { return _depthData; }
    const DepthType * getDeviceDepth() const { return 0; }
#endif // CUDA_BUILD

    float getScaleToMeters() const { return _ScaleToMeters; }

    uint64_t getDepthTime() const { return _depthTime; }

    bool initLCM(const std::string channel);

    // handle lcm images_t message and save their content
    void imgHandle(const lcm::ReceiveBuffer* rbuf, const std::string& channel,
                   const bot_core::images_t* msg);

    // convert a disparity image to a distance image using camera parameters
    void disparity_to_depth(DepthType *disparity_img);
};

#endif
