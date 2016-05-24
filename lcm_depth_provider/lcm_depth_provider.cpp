#include <lcm_depth_provider.hpp>

#include <iostream>

#include <vector_functions.hpp>

template <typename DepthType, typename ColorType>
LCM_DepthSource<DepthType,ColorType>::LCM_DepthSource() {
    this->_isLive = false; // no way to control LCM playback from here
    this->_hasColor = false; // only depth for now
    this->_colorWidth = 0;
    this->_colorHeight = 0;
    this->_hasTimestamps = true;
    this->_frame = 0; // initialize first frame

    // camera properties
    // [fx fy skew cx cy] => [ 556.183166504, 556.183166504, 0, 512.0, 512.0 ]
    // rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    // translation = [-0.0700, 0.0, 0.0]

    this->_focalLength = make_float2(556.183166504, 556.183166504);

    // TODO: set dynamically
    this->_depthWidth = 1024;
    this->_depthHeight = 1024;
    this->_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);
}

template <typename DepthType, typename ColorType>
LCM_DepthSource<DepthType,ColorType>::~LCM_DepthSource() {
    if(lcm!=NULL)
        delete lcm;
}

//template <typename DepthType, typename ColorType>
//void LCM_DepthSource<DepthType,ColorType>::init() {
//    this->_frame = 0;
//}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::setFrame(const uint frame) {
    if (this->_isLive)
        return;
}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::advance() {
    lcm->handle();
}

template <typename DepthType, typename ColorType>
bool LCM_DepthSource<DepthType,ColorType>::hasRadialDistortionParams() const {
    return false;
}

template <typename DepthType, typename ColorType>
const DepthType* LCM_DepthSource<DepthType,ColorType>::getDepth() const {

}

template <typename DepthType, typename ColorType>
const DepthType* LCM_DepthSource<DepthType,ColorType>::getDeviceDepth() const {

}

template <typename DepthType, typename ColorType>
bool LCM_DepthSource<DepthType,ColorType>::initLCM(const std::string channel) {

    lcm = new lcm::LCM();
    if(!lcm->good())
        return false;

    lcm->subscribe(channel, &LCM_DepthSource<short unsigned int, uchar3>::imgHandle, this);
}

template <typename DepthType, typename ColorType>
bool LCM_DepthSource<DepthType,ColorType>::handleLCM() {
    return lcm->handle();
}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::imgHandle(const lcm::ReceiveBuffer* rbuf, const std::string& channel,
               const bot_core::images_t* msg) {
    std::cout<<"received message!!"<<std::endl;

    //uint _depthWidth;
    //uint _depthHeight;
    //_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);
}
