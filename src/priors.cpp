#include <priors.hpp>

dart::NoCameraMovementPrior::NoCameraMovementPrior(const int srcModelID) : _srcModelID(srcModelID) {}

void dart::NoCameraMovementPrior::computeContribution(Eigen::SparseMatrix<float> & JTJ,
                             Eigen::VectorXf & JTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts)
{
    // get offsets for selected model in the full parameter space
    const Pose & srcPose = poses[_srcModelID];
    const int srcDims = srcPose.getReducedDimensions();
    const int srcOffset = modelOffsets[_srcModelID];

    // get step in parameter space at current optimization state
    Eigen::VectorXf paramUpdate = JTJ.block(srcOffset,srcOffset,srcDims,srcDims).triangularView<Eigen::Upper>().solve(JTe.segment(srcOffset,srcDims));

    // set camera to model transformation (first 6 parameters) to zero
    paramUpdate.head<6>() = Eigen::VectorXf::Zero(6);

    // compute jacobian and error to achieve no movement of camera frame
    JTe.segment(srcOffset,srcDims) = JTJ.block(srcOffset,srcOffset,srcDims,srcDims) * paramUpdate;
}

dart::ReportedJointsPrior::ReportedJointsPrior(const int modelID, const Pose &reported, const Pose &current, const double weight)
    : _modelID(modelID), _reported(reported), _estimated(current), _weight(weight) { }

void dart::ReportedJointsPrior::computeContribution(Eigen::SparseMatrix<float> & fullJTJ,
                             Eigen::VectorXf & fullJTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts)
{
    // get mapping of reported joint names and values
    std::map<std::string, float> rep_map;
    for(unsigned int i=0; i<_reported.getReducedArticulatedDimensions(); i++) {
        rep_map[_reported.getReducedName(i)] = _reported.getReducedArticulation()[i];
    }

    // compute difference of reported to estimated joint value
    Eigen::VectorXf diff = Eigen::VectorXf::Zero(_estimated.getReducedArticulatedDimensions());
    for(unsigned int i=0; i<_estimated.getReducedArticulatedDimensions(); i++) {
        const std::string jname = _estimated.getReducedName(i);
        float rep = rep_map.at(jname);
        float est = _estimated.getReducedArticulation()[i];
        diff[i] = rep_map.at(jname) - _estimated.getReducedArticulation()[i];
    }

    // set nan values to 0, e.g. comparison of nan values always yields false
    diff = (diff.array()!=diff.array()).select(0,diff);

    // L2 norm of error vector
    const double e = _weight*diff.norm();

    // Jacobian, e.g. the partial derivation of the error w.r.t. to each joint value
    const Eigen::VectorXf J = - diff.array() / e;
    const Eigen::MatrixXf JTJ = J*J.transpose();
    const Eigen::VectorXf JTe = - diff.transpose();

    for(unsigned int r=0; r<JTJ.rows(); r++)
        for(unsigned int c=0; c<JTJ.cols(); c++)
            if(JTJ(r,c)!=0)
                fullJTJ.coeffRef(modelOffsets[_modelID]+6+r, modelOffsets[_modelID]+6+c) += JTJ(r,c);

    for(unsigned int r=0; r<JTe.rows(); r++)
            if(JTe[r]!=0)
                fullJTe[modelOffsets[_modelID]+6+r] += JTe[r];
}
