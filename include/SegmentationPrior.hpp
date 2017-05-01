#ifndef SEGMENTATIONPRIOR_HPP
#define SEGMENTATIONPRIOR_HPP

#include <dart/tracker.h>
#include <img_classif_msgs/image_class_t.hpp>
#include <lcm/lcm-cpp.hpp>
#include <mutex>

using namespace dart;

class SegmentationPrior : public Prior {
public:
    typedef Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi16r;
    SegmentationPrior(Tracker &tracker);

//    virtual int getNumPriorParams() const { return 0; }

//    virtual float * getPriorParams() { return 0; }

//    virtual void updatePriorParams(const float * update, const std::vector<MirroredModel *> & models) { }

    virtual void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                                     Eigen::VectorXf & JTe,
                                     const int * modelOffsets,
                                     const int priorParamOffset,
                                     const std::vector<MirroredModel *> & models,
                                     const std::vector<Pose> & poses,
                                     const OptimizationOptions & opts);

    void onImgClass(const lcm::ReceiveBuffer* /*rbuf*/, const std::string& /*chan*/,  const img_classif_msgs::image_class_t* img_class_id);


private:
    Tracker &tracker;
    lcm::LCM lcm;
    std::mutex mutex;
    MatrixXi16r img_class;
};

#endif // SEGMENTATIONPRIOR_HPP
