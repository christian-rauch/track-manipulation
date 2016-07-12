#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <dart/tracker.h>

namespace dart {

/**
 * @brief The NoCameraMovementPrior class
 * Prior to prevent movement of camera, e.g. set transformation of model to camera to 0.
 * This prior needs to be added last to enforce no transformation.
 */
class NoCameraMovementPrior : public Prior {
private:
    int _srcModelID;

public:
    NoCameraMovementPrior(const int srcModelID);

    void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                             Eigen::VectorXf & JTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts);
};

class ReportedJointsPrior : public Prior {
private:
    // references to both pose sources for continuous updates
    const Pose &_reported;
    const Pose &_estimated;
    const int _modelID;
    const double _weight;

public:
    ReportedJointsPrior(const int modelID, const Pose &reported, const Pose &current, const double weight=1.0);

    void computeContribution(Eigen::SparseMatrix<float> & fullJTJ,
                                 Eigen::VectorXf & fullJTe,
                                 const int * modelOffsets,
                                 const int priorParamOffset,
                                 const std::vector<MirroredModel *> & models,
                                 const std::vector<Pose> & poses,
                                 const OptimizationOptions & opts);
};

}

#endif // PRIORS_HPP
