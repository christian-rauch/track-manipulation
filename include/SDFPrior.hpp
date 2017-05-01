#ifndef SDFPRIOR_HPP
#define SDFPRIOR_HPP

#include <dart/tracker.h>

using namespace dart;

class SDFPrior : public Prior {
public:
    //SegmentationPrior(const Optimizer *optim);
    SDFPrior(Tracker &tracker) : tracker(tracker) { }

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
private:
    //const Optimizer *optim;
    Tracker &tracker;
};

#endif // SDFPRIOR_HPP
