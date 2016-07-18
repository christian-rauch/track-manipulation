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

    /**
     * @brief computeGNParam compute parameter for Gauss-Newton
     * @param diff vector of differences in joint angles
     * @return tuple with Jacobian J and the gradient J^T*e
     */
    virtual std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff) = 0;

protected:
    const double _weight;

public:
    explicit ReportedJointsPrior(const int modelID, const Pose &reported, const Pose &current, const double weight=1.0);

    void computeContribution(Eigen::SparseMatrix<float> & fullJTJ,
                                 Eigen::VectorXf & fullJTe,
                                 const int * modelOffsets,
                                 const int priorParamOffset,
                                 const std::vector<MirroredModel *> & models,
                                 const std::vector<Pose> & poses,
                                 const OptimizationOptions & opts);
};

class WeightedL2NormOfError : public ReportedJointsPrior {
    using ReportedJointsPrior::ReportedJointsPrior;
private:
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff);
};

class L2NormOfWeightedError : public ReportedJointsPrior {
    using ReportedJointsPrior::ReportedJointsPrior;
private:
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff);
};

}

#endif // PRIORS_HPP
