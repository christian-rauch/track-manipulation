#include <lcm_depth_provider.hpp>

#include <iostream>

#include <vector_functions.hpp>

#include <zlib.h>

template <typename DepthType, typename ColorType>
LCM_DepthSource<DepthType,ColorType>::LCM_DepthSource(const StereoCameraParameter &param) {
    this->_isLive = true; // no way to control LCM playback from here
    this->_hasColor = false; // only depth for now
    this->_colorWidth = 0;
    this->_colorHeight = 0;
    this->_hasTimestamps = true;
    this->_frame = 0; // initialize first frame

    // camera properties
    // [fx fy skew cx cy] => [ 556.183166504, 556.183166504, 0, 512.0, 512.0 ]
    // rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    // translation = [-0.0700, 0.0, 0.0]

    _cam_param = param;

    //this->_focalLength = make_float2(556.183166504, 556.183166504);
    this->_focalLength = param.focal_length;

//    // set principal point / camera centre if given
//    if(param.camera_center.x > 0 && param.camera_center.y > 0)
//        this->_principalPoint = param.camera_center;

//    // set image dimensions if given
//    if(param.width > 0 && param.height > 0) {
//        this->_depthWidth = param.width;
//        this->_depthHeight = param.height;
//    }

    this->_principalPoint = param.camera_center;
    this->_depthWidth = param.width;
    this->_depthHeight = param.height;

    //_ScaleToMeters = 0.001; // depth_in_m = _ScaleToMeters * depth_in_data ??
    _ScaleToMeters = 0.02;

    // TODO: set dynamically
    //this->_depthWidth = 1024;
    //this->_depthHeight = 1024;
    //this->_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);
    //this->_principalPoint = make_float2(512, 512);

//    _cam_param.baseline = 0.07;
//    _cam_param.focal_length = make_float2(556.183166504, 556.183166504);
//    _cam_param.camera_center = make_float2(512, 512);
//    _cam_param.width = 1024;
//    _cam_param.height = 1024;

    // allocate memory for depth image
#ifdef CUDA_BUILD
    _depthData = new dart::MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
#else
    _depthData = new DepthType[this->_depthWidth*this->_depthHeight];
#endif // CUDA_BUILD
}

template <typename DepthType, typename ColorType>
LCM_DepthSource<DepthType,ColorType>::~LCM_DepthSource() {
    if(lcm!=NULL)
        delete lcm;
#ifdef CUDA_BUILD
    delete _depthData;
#else
    delete [] _depthData;
#endif // CUDA_BUILD
}

//template <typename DepthType, typename ColorType>
//void LCM_DepthSource<DepthType,ColorType>::init() {
//    this->_frame = 0;
//}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::setFrame(const uint frame) {
    // nothing to do, we cannot control LCM playbag from here
    if(this->_isLive) return;
}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::advance() {
    this->_frame++;

    // wait (block) for new messages
    lcm->handle();

#ifdef CUDA_BUILD
    _depthData->syncHostToDevice();
#endif // CUDA_BUILD
}

template <typename DepthType, typename ColorType>
bool LCM_DepthSource<DepthType,ColorType>::hasRadialDistortionParams() const {
    return false;
}

//template <typename DepthType, typename ColorType>
//const DepthType* LCM_DepthSource<DepthType,ColorType>::getDepth() const {

//}

//template <typename DepthType, typename ColorType>
//const DepthType* LCM_DepthSource<DepthType,ColorType>::getDeviceDepth() const {

//}

template <typename DepthType, typename ColorType>
bool LCM_DepthSource<DepthType,ColorType>::initLCM(const std::string img_channel) {

    lcm = new lcm::LCM();
    if(!lcm->good())
        return false;

    lcm->subscribe(img_channel, &LCM_DepthSource<short unsigned int, uchar3>::imgHandle, this);
}

//template <typename DepthType, typename ColorType>
//bool LCM_DepthSource<DepthType,ColorType>::handleLCM() {
//    return lcm->handle();
//}

template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::imgHandle(const lcm::ReceiveBuffer* rbuf, const std::string& channel,
               const bot_core::images_t* msg) {
    std::cout<<"received message!!"<<std::endl;

    // TODO: set dynamically
    //uint _depthWidth;
    //uint _depthHeight;
    //_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);

    // read data
    const int64_t n_img = msg->n_images;
    const std::vector<int16_t> image_types = msg->image_types;
    std::cout<<"received "<<n_img<<" images at "<<msg->utime<<std::endl;

    // pointer to constant values of raw data, we are not going to change the image
    const uint8_t * raw_data = NULL;
    //uint8_t * raw_data = NULL;

    // go over all images
    for(unsigned int i=0; i<n_img; i++) {
        const int16_t img_type = image_types[i];
        // number of bytes of raw data
        const int32_t image_size_raw = msg->images[i].size;
        // real expected image size (width*height*byte_per_pixel)
        uint64_t image_size_real = msg->images[i].row_stride * msg->images[i].height;

        // dbg: show information about disparity image
        if(img_type==bot_core::images_t::DISPARITY || img_type==bot_core::images_t::DISPARITY_ZIPPED) {
            std::cout<<"img size raw "<<image_size_raw<<" byte"<<std::endl;
            std::cout<<"img size real "<<image_size_real<<" byte"<<std::endl;
            //std::cout<<"img size real2 "<<msg->images[i].data.size()<<" byte"<<std::endl;
            const int bpp = msg->images[i].row_stride / msg->images[i].width;
            std::cout<<"using "<<bpp<<" byte per pixel"<<std::endl;
        }


        // process image data
        switch(img_type) {
        // raw disparity image data
        case bot_core::images_t::DISPARITY:
            raw_data = msg->images[i].data.data();
            //raw_data = msg->images[i].data;
            break;

        // zlib compressed disparity image data
        case bot_core::images_t::DISPARITY_ZIPPED:
            // decompress data
            const uint8_t * raw_data_compressed = msg->images[i].data.data();
            // allocate memory for uncompressed image
            raw_data = new uint8_t[image_size_real];
            // uncompress OF((Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen));
            const int zip_res = uncompress((Bytef *)raw_data, &image_size_real, (Bytef *)raw_data_compressed, image_size_raw);
            if(zip_res!=Z_OK) {
                std::cerr<<"something went wrong in decompressing data (err: "<<zip_res<<")"<<std::endl;
                switch(zip_res) {
                case Z_MEM_ERROR:
                    std::cerr<<"(Z_MEM_ERROR) not enough memory"<<std::endl;
                    break;
                case Z_BUF_ERROR:
                    std::cerr<<"(Z_BUF_ERROR) not enough room in the output buffer"<<std::endl;
                    break;
                case Z_DATA_ERROR:
                    std::cerr<<"(Z_DATA_ERROR) input data is corrupted or incomplete"<<std::endl;
                    break;
                case Z_STREAM_ERROR:
                    std::cerr<<"(Z_STREAM_ERROR) data stream error"<<std::endl;
                    break;
                }
            }
            std::cout<<"img size raw "<<image_size_raw<<std::endl;
            std::cout<<"img size real "<<image_size_real<<std::endl;
            break;
        }
    } // for i over images_t

    if(raw_data!=NULL) {
        // found disparity image
        // convert
        disparity_to_depth((DepthType*)raw_data);
    }
    else {
        std::cerr<<"no disparity image found"<<std::endl;
    }

    // cast time from signed to unsigned
    _depthTime = (msg->utime >= 0) ? (uint64_t)msg->utime : 0;

    // sync
#ifdef CUDA_BUILD
    //std::cout<<"copying "<<sizeof(DepthType)<<" x "<<_depthData->length()<<" bytes of data"<<std::endl;
    memcpy(_depthData->hostPtr(), raw_data, sizeof(DepthType)*_depthData->length());
    delete [] raw_data;
#else
    _depthData = raw_data;
    // TODO: who is freeing 'raw_data'?
#endif // CUDA_BUILD

}

// replace disparity by depth, values are changes inplace (in memory)
template <typename DepthType, typename ColorType>
void LCM_DepthSource<DepthType,ColorType>::disparity_to_depth(DepthType * disparity_img) {
    // Z = (f*b)/d
    // distance = (focal_length_pxl * baseline_meter) / disparity_pxl
    const float factor = _cam_param.focal_length.x * _cam_param.baseline;
    const float scale = 1.0/(_ScaleToMeters);
    for(unsigned int i=0; i<_depthData->length(); i++) {
        // deal with disparity 0 (avoid FPE / division by zero)
        disparity_img[i] = (disparity_img[i]!=0) ? (factor / disparity_img[i]) * scale : 0;
        //disparity_img[i] *= 1.0/(_ScaleToMeters);
    }
}
