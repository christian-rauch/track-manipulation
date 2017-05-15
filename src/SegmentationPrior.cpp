#include "SegmentationPrior.hpp"
#include <dart/optimization/kernels/obsToMod.h>

//#define DBG_DA
#include <fstream>
//#define PRNT_DA

SegmentationPrior::SegmentationPrior(Tracker &tracker) : tracker(tracker), lcm() {
    if(lcm.good())
        lcm.subscribe("SEGMENTED_IMAGE", &SegmentationPrior::onImgClass, this);
    else
        throw std::runtime_error("LCM error in SegmentationPrior");
}

void SegmentationPrior::computeContribution(
        Eigen::SparseMatrix<float> & JTJ,
        Eigen::VectorXf & JTe,
        const int * modelOffsets,
        const int priorParamOffset,
        const std::vector<MirroredModel *> & models,
        const std::vector<Pose> & poses,
        const OptimizationOptions & opts)
{
    CheckCudaDieOnError();

    // check for new classified images, ignore prior otherwise
    if(lcm.handleTimeout(10)<=0)
        return;

    if(models.size()!=poses.size()) {
        throw std::runtime_error("models!=poses");
    }

    const uint h = tracker.getPointCloudSource().getDepthHeight();
    const uint w = tracker.getPointCloudSource().getDepthWidth();
    const uint ndata = w*h;

    for(uint i(0); i<models.size(); i++) {
        // reset _lastElements counter on host and device
        cudaMemset(tracker.getOptimizer()->_lastElements->devicePtr(),0,sizeof(int));
        tracker.getOptimizer()->_lastElements->syncDeviceToHost();
        CheckCudaDieOnError();

        // host copy of observed points
        std::vector<float4> points(ndata);
        cudaMemcpy(points.data(), tracker.getPointCloudSource().getDeviceVertMap(), ndata*sizeof(float4), cudaMemcpyDefault);
        CheckCudaDieOnError();

        std::vector<DataAssociatedPoint> dpoints;
        mutex.lock();
        for(uint iw(0); iw<w; iw++) {
            for(uint ih(0); ih<h; ih++) {
                const uint index = iw + ih*w;
                if(points[index].w>0) { // valid observation
                    if(img_class(ih,iw)>0) { // no background
                        DataAssociatedPoint da;
                        da.index = index;
                        da.dataAssociation = img_class(ih,iw); // get predicted class
                        da.error = 0.0; // we will compute error in other kernel
                        dpoints.push_back(da);
                    }
                }
            }
        }
        mutex.unlock();

        // set number of points, only used at host
        (*tracker.getOptimizer()->_lastElements)[i] = dpoints.size();
        tracker.getOptimizer()->_lastElements->syncHostToDevice();
        // sync data
        cudaMemcpy((*tracker.getOptimizer()->_dPts)[i],
                   dpoints.data(),
                   dpoints.size()*sizeof(DataAssociatedPoint),
                   cudaMemcpyHostToDevice);

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
        CheckCudaDieOnError();
    }


#ifdef PRNT_DA
    // DBG
    for(uint imod(0); imod<models.size(); imod++) {
        auto le = tracker.getOptimizer()->_lastElements->hostPtr()[imod];
        std::cout << "le: " << le << std::endl;

        std::cout << "######## dPts DA " << imod << std::endl;
        DataAssociatedPoint *da;
        cudaMallocHost(&da, ndata*sizeof(DataAssociatedPoint));
        CheckCudaDieOnError();
        cudaMemcpy(da, tracker.getOptimizer()->_dPts->hostPtr()[imod], ndata*sizeof(DataAssociatedPoint),cudaMemcpyDeviceToHost);
        CheckCudaDieOnError();
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> da_mat(h,w);
        da_mat.setConstant(-1);
        // index = x + y*width;
        for(uint j(0); j<le; j++) {
//            std::cout << da[j].index << " " << da[j].dataAssociation << " " << da[j].error << std::endl;
//            if(da[j].index!=0) {
//                std::cout << da[j].index << " " << da[j].dataAssociation << " " << da[j].error << std::endl;
//            }
            const uint x = da[j].index%w;
            const uint y = (x==0) ? da[j].index : da[j].index/w;
            da_mat(y,x) = da[j].dataAssociation;
        }
        cudaFreeHost(da);
        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
        const std::string fname = "dPts.csv";
        std::ofstream csvfile(fname.c_str());
        csvfile << da_mat.format(CSVFormat);
        csvfile.close();
    }
    CheckCudaDieOnError();
#endif
}

void SegmentationPrior::onImgClass(const lcm::ReceiveBuffer* /*rbuf*/, const std::string& /*chan*/,  const img_classif_msgs::image_class_t* img_class_msg) {
    mutex.lock();
    img_class.resize(img_class_msg->height, img_class_msg->width);
    // copy row by row
    for(uint r(0); r<img_class.rows(); r++) {
        std::memcpy(img_class.row(r).data(), img_class_msg->class_id[r].data(), img_class_msg->width*sizeof(decltype(img_class)::Scalar));
    }
    mutex.unlock();
}
