#include "SDFPrior.hpp"
#include <dart/optimization/kernels/obsToMod.h>

void SDFPrior::computeContribution(
        Eigen::SparseMatrix<float> & JTJ,
        Eigen::VectorXf & JTe,
        const int * modelOffsets,
        const int priorParamOffset,
        const std::vector<MirroredModel *> & models,
        const std::vector<Pose> & poses,
        const OptimizationOptions & opts)
{
    if(models.size()!=poses.size()) {
        throw std::runtime_error("models!=poses");
    }

    for(uint i(0); i<models.size(); i++) {
        errorAndDataAssociation(
                    tracker.getPointCloudSource().getDeviceVertMap(),
                    tracker.getPointCloudSource().getDeviceNormMap(),
                    int(tracker.getPointCloudSource().getDepthWidth()),
                    int(tracker.getPointCloudSource().getDepthHeight()),
                    *models[i],
                    opts,
                    //_dPts->hostPtr()[0],
                    tracker.getOptimizer()->_dPts->hostPtr()[i],
                    //_lastElements->devicePtr(),
                    tracker.getOptimizer()->_lastElements->devicePtr(),
                    //_lastElements->hostPtr(),
                    tracker.getOptimizer()->_lastElements->hostPtr(),
                    NULL, NULL, NULL);

        // compute gradient / Hessian

        float obsToModError = 0;
        Observation observation(tracker.getPointCloudSource().getDeviceVertMap(),
                                tracker.getPointCloudSource().getDeviceNormMap(),
                                int(tracker.getPointCloudSource().getDepthWidth()),
                                int(tracker.getPointCloudSource().getDepthHeight()));

        // enforce lambdaObsToMod = 1.0 for the unpack in 'computeObsToModContribution'
        // 'computeObsToModContribution' is private anyway
        OptimizationOptions fake_opts(opts);
        fake_opts.lambdaObsToMod = 1.0;

        Eigen::MatrixXf denseJTJ(JTJ);
        tracker.getOptimizer()->computeObsToModContribution(JTe,denseJTJ,obsToModError,*models[i],poses[i],fake_opts,observation);
        JTJ = denseJTJ.sparseView();
    }
}
