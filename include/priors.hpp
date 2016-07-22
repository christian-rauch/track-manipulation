#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <dart/tracker.h>

// for publishing debugging information
#define LCM_DEBUG_GRADIENT

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
#ifdef LCM_DEBUG_GRADIENT
    unsigned int _skipped;
#endif

    /**
     * @brief computeGNParam compute parameter for Gauss-Newton
     * @param diff vector of differences in joint angles
     * @return tuple with Jacobian J and the gradient J^T*e
     */
    virtual std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff) = 0;

#ifdef LCM_DEBUG_GRADIENT
    void setup();
#endif

protected:
    const double _weight;
    const Eigen::MatrixXf _Q;

public:
    /**
     * @brief ReportedJointsPrior constructor for scalar weights
     * @param modelID ID of model in DART tracker
     * @param reported reported pose
     * @param current estimated pose
     * @param weight scalar weight
     */
    explicit ReportedJointsPrior(const int modelID, const Pose &reported, const Pose &current, const double weight);

    /**
     * @brief ReportedJointsPrior constructor for weights in matrix form
     * @param modelID ID of model in DART tracker
     * @param reported reported pose
     * @param current estimated pose
     * @param Q square weight matrix with dimensions like joint vector
     */
    explicit ReportedJointsPrior(const int modelID, const Pose &reported, const Pose &current, const Eigen::MatrixXf Q);

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

class QWeightedError : public ReportedJointsPrior {
    using ReportedJointsPrior::ReportedJointsPrior;
private:
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff);
};

class SimpleWeightedError : public ReportedJointsPrior {
    using ReportedJointsPrior::ReportedJointsPrior;
private:
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf> computeGNParam(const Eigen::VectorXf &diff);
};

}

#endif // PRIORS_HPP
