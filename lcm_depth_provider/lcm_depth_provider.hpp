#ifndef LCM_DEPTH_PROVIDER_H
#define LCM_DEPTH_PROVIDER_H

#include <depth_sources/depth_source.h>

#include <lcm/lcm-cpp.hpp>
#include <bot_core/images_t.hpp>

template <typename DepthType, typename ColorType>
class LCM_DepthSource : public dart::DepthSource<DepthType,ColorType> {
private:
    lcm::LCM *lcm = NULL;

    DepthType *depth_data = NULL;

public:
    LCM_DepthSource();

    ~LCM_DepthSource();

    //void init();

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const;

    const DepthType* getDepth() const;

    const DepthType* getDeviceDepth() const;

    bool initLCM(const std::string channel);

    bool handleLCM();

    void imgHandle(const lcm::ReceiveBuffer* rbuf, const std::string& channel,
                   const bot_core::images_t* msg);
};

#endif
