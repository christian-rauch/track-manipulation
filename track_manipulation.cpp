#include <iostream>
#include <stdio.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pangolin/pangolin.h>
#include <pangolin/utils/timer.h>

#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <dart/depth_sources/image_depth_source.h>
#include <dart/geometry/plane_fitting.h>
#include <dart/img_proc/img_ops.h>
#include <dart/optimization/priors.h>
#include <dart/tracker.h>
#include <dart/util/dart_io.h>
#include <dart/util/gl_dart.h>
#include <dart/util/image_io.h>
#include <dart/util/ostream_operators.h>
#include <dart/util/string_format.h>
#include <dart/visualization/color_ramps.h>
#include <dart/visualization/data_association_viz.h>
#include <dart/visualization/gradient_viz.h>
#include <dart/visualization/sdf_viz.h>

#include <priors.hpp>

#define EIGEN_DONT_ALIGN

// select a robot
//#define JUSTIN      // using XML model and data files
#define VALKYRIE    // using URDF model and LCM subscriptions

//#define WITH_BOTTLE
//#define WITH_BOX
//#define WITH_RECT

// switch depth sources
#ifdef JUSTIN
    #define DEPTH_SOURCE_IMAGE // demo files
#endif

#ifdef VALKYRIE
    // read depth images from LCM topic
    #define DEPTH_SOURCE_LCM
    #define DEPTH_SOURCE_LCM_MULTISENSE
    //#define DEPTH_SOURCE_LCM_XTION
    // read depth (not disparity) images from MultiSense SL, use specific camera parameters
    //#define DEPTH_SOURCE_IMAGE_MULTISENSE
    #include <dart_lcm/lcm_state_publish.hpp>
    #include <dart_lcm/lcm_frame_pose_publish.hpp>
#endif

#ifdef DEPTH_SOURCE_LCM
    #include <dart_lcm/dart_lcm_depth_provider.hpp>
#endif

#ifdef JUSTIN
    #define ENABLE_JUSTIN
    // filter point below table plane
#endif

#ifdef VALKYRIE
    // FIXME: set by cmake
    #define ENABLE_URDF
    #define ENABLE_LCM_JOINTS
#endif

#ifdef ENABLE_URDF
    #include <dart_urdf/read_model_urdf.h>
#endif

#ifdef ENABLE_LCM_JOINTS
    #include <dart_lcm/dart_lcm_joints.hpp>
#endif

#ifdef JUSTIN
    // switch use of contact information
    #define USE_CONTACT_PRIOR
#endif

#define LCM_CHANNEL_ROBOT_STATE "EST_ROBOT_STATE"
//#define LCM_CHANNEL_ROBOT_STATE "EST_ROBOT_STATE_ORG"
#define LCM_CHANNEL_DART_PREFIX "DART_"
//#define LCM_CHANNEL_DART_PREFIX "EST_ROBOT"

using namespace std;

enum PointColorings {
    PointColoringNone = 0,
    PointColoringRGB,
    PointColoringErr,
    PointColoringDA,
    NumPointColorings
};

enum DebugImgs {
    DebugColor=0,
    DebugObsDepth,
    DebugPredictedDepth,
    DebugObsToModDA,
    DebugModToObsDA,
    DebugObsToModErr,
    DebugModToObsErr,
    DebugJTJ,
    DebugN
};

#ifdef ENABLE_JUSTIN
enum TrackingMode {
    ModeObjOnTable,
    ModeIntermediate,
    ModeObjGrasped,
    ModeObjGraspedLeft
};

std::string getTrackingModeString(const TrackingMode mode) {
    switch (mode) {
    case ModeObjOnTable:
        return "object on table";
    case ModeIntermediate:
        return "intermediate";
    case ModeObjGrasped:
        return "object grasped";
    case ModeObjGraspedLeft:
        return "object grasped in left hand";
    }
}

static const int fullArmFingerTipFrames[10] = { 11, 15, 19, 23, 27,  38, 42, 46, 50, 54 };
static const int handFingerTipFrames[5] = { 4, 8, 12, 16, 20 };
#endif

const static int panelWidth = 180;

void setSlidersFromTransform(dart::SE3& transform, pangolin::Var<float>** sliders) {
    *sliders[0] = transform.r0.w; transform.r0.w = 0;
    *sliders[1] = transform.r1.w; transform.r1.w = 0;
    *sliders[2] = transform.r2.w; transform.r2.w = 0;
    dart::se3 t = dart::se3FromSE3(transform);
    *sliders[3] = t.p[3];
    *sliders[4] = t.p[4];
    *sliders[5] = t.p[5];
}
void setSlidersFromTransform(const dart::SE3& transform, pangolin::Var<float>** sliders) {
    dart::SE3 mutableTransform = transform;
    setSlidersFromTransform(mutableTransform,sliders);
}
#ifdef ENABLE_JUSTIN
const static dart::SE3 T_wh = dart::SE3FromRotationY(M_PI)*dart::SE3FromRotationX(-M_PI_2)*dart::SE3FromTranslation(make_float3(0,0,0.138));//dart::SE3Fromse3(dart::se3(0,0,0.1,0,2.22144,2.22144)); //dart::SE3art::SE3Fromse3(dart::se3(0, 0.108385,-0.108385, 1.5708, 0, 0)); // = dart::SE3Invert(dart::SE3Fromse3(dart::se3(0, 0.115, -0.115, 1.5708, 0, 0)));
const static dart::SE3 T_hw = dart::SE3Invert(T_wh);
const static dart::SE3 T_wc = dart::SE3FromTranslation(make_float3(-0.2,0.8,0))*
        dart::SE3Fromse3(dart::se3(0,0,0,0,M_PI_2,0))*
        dart::SE3Fromse3(dart::se3(0,0,0, 2.1,0,0));

const static int rightShoulderFrame = 1;
const static int rightPalmFrame = 7;
const static int leftShoulderFrame = 28;
const static int leftPalmFrame = 34;
const static int headFrame = 56;

void loadReportedJointAngles(std::string jointAngleFile, std::vector<float *> & jointAngles) {
    int nFrames;
    int nJoints;

    std::ifstream jointAngleStream;
    jointAngleStream.open(jointAngleFile);

    assert(jointAngleStream.is_open());

    jointAngleStream >> nFrames;

    jointAngleStream >> nJoints;

    for (int i=0; i<nFrames; ++i) {

        float * frameAngles = new float[nJoints];
        for (int j=0; j<nJoints; ++j) {
            jointAngleStream >> frameAngles[j];
        }
        jointAngles.push_back(frameAngles);
    }

    jointAngleStream.close();
}

void loadReportedContacts(std::string contactFile, std::vector<int *> & contacts) {
    int nFrames, nContactFrames;
    int nJoints;

    std::ifstream contactStream;
    contactStream.open(contactFile);

    contactStream >> nContactFrames;

    for (int i=0; i<nContactFrames; ++i) {
        int * frameContacts = new int[10];
        for (int j=0; j<10; ++j) {
            contactStream >> frameContacts[j];
        }
        contacts.push_back(frameContacts);
    }

    contactStream.close();
}
#endif

dart::Pose nullReductionPose(const dart::HostOnlyModel &model) {
    std::vector<float> jointMins, jointMaxs;
    std::vector<std::string> jointNames;
    for (int j=0; j<model.getNumJoints(); ++j) {
        jointMins.push_back(model.getJointMin(j));
        jointMaxs.push_back(model.getJointMax(j));
        jointNames.push_back(model.getJointName(j));
    }
    return dart::Pose(new dart::NullReduction(model.getNumJoints(),
                    jointMins.data(), jointMaxs.data(), jointNames.data()));
}

// debugging function to print all joint positions stored in a pose
void printPoseJoints(dart::Pose &pose) {
    const uint ndims = pose.getReducedArticulatedDimensions();
    std::cout<<"reduced joints ("<<ndims<<"):"<<std::endl;
    for(uint i=0; i<ndims; i++) {
        std::cout<<i<<" "<<pose.getReducedName(i)<<": "<<pose.getReducedArticulation()[i]<<std::endl;
    }
}

#ifdef ENABLE_JUSTIN
static const dart::SE3 initialT_cj(make_float4(-0.476295, -0.0945505, -0.874187, -0.22454),
                                   make_float4(-0.625852, 0.734788, 0.26152, -0.305038   ),
                                   make_float4(0.617613, 0.671677, -0.409147, -0.105219  ));

static const dart::SE3 initialT_co(make_float4(0.262348, -0.955909, -0.131952, 0.0238097),
                                   make_float4(-0.620357, -0.271813, 0.735714, -0.178571),
                                   make_float4(-0.739142, -0.111156, -0.664314, 0.702381));
#endif

static float3 initialTableNorm = make_float3(0.0182391, 0.665761, -0.745942);
static float initialTableIntercept = -0.705196;

int main(int argc, char *argv[]) {

#ifdef ENABLE_JUSTIN
    const std::string objectModelFile = "../models/ikeaMug/ikeaMug.xml";
    const float objObsSdfRes = 0.0025;
    const float3 objObsSdfOffset = make_float3(0,0,0);

    const std::string videoLoc = "../video/";
#endif

    // -=-=-=- initializations -=-=-=-
    cudaSetDevice(0);
    cudaDeviceReset();

    pangolin::CreateWindowAndBind("Main",640+4*panelWidth+1,2*480+1);

    glewInit();
    glutInit(&argc, argv);
    dart::Tracker tracker;

    // -=-=-=- pangolin window setup -=-=-=-

    pangolin::CreatePanel("lim").SetBounds(0.0,1.0,1.0,pangolin::Attach::Pix(-panelWidth));
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("opt").SetBounds(0.0,1.0,pangolin::Attach::Pix(panelWidth), pangolin::Attach::Pix(2*panelWidth));
    pangolin::CreatePanel("pose").SetBounds(0.0,1.0,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));

    int glWidth = 640;
    int glHeight= 480;
    int glFL = 400;
    int glPPx = glWidth/2;
    int glPPy = glHeight/2;
#ifdef ENABLE_JUSTIN
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);
    pangolin::OpenGlRenderState camState(glK);
#endif
#ifdef ENABLE_URDF
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRUB_BottomLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);
    pangolin::OpenGlMatrix viewpoint = pangolin::ModelViewLookAt(0, 0, 0.05, 0, -0.1, 0.2, pangolin::AxisY);
    //pangolin::OpenGlMatrix viewpoint = pangolin::ModelViewLookAt(0, 0, 0.05, 0, 0, 0.2, pangolin::AxisY);
    // the MultiSense and the Xtion have different oriented frames
#ifdef DEPTH_SOURCE_LCM_MULTISENSE
    // Z forward, Y up
    pangolin::OpenGlRenderState camState(glK, viewpoint);
#endif
#ifdef DEPTH_SOURCE_LCM_XTION
    // Z forward, Y down
    pangolin::OpenGlRenderState camState(pangolin::OpenGlMatrix::RotateZ(M_PI)*glK, viewpoint);
#endif
#endif
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(640.0f/480.0f).SetHandler(new pangolin::Handler3D(camState));

    pangolin::View & imgDisp = pangolin::Display("img").SetAspect(640.0f/480.0f);
    pangolin::GlTexture imgTexDepthSize(320,240);
    pangolin::GlTexture imgTexPredictionSize(160,120);

    pangolin::DataLog infoLog;
    {
        std::vector<std::string> infoLogLabels;
        infoLogLabels.push_back("errObsToMod");
        infoLogLabels.push_back("errModToObs");
        infoLogLabels.push_back("stabilityThreshold");
        infoLogLabels.push_back("resetThreshold");
        infoLog.SetLabels(infoLogLabels);
    }

    pangolin::Display("multi")
            .SetBounds(1.0, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(camDisp)
            //            .AddDisplay(infoPlotter)
            .AddDisplay(imgDisp);

//    imgDisp.Show(false);

    float defaultModelSdfPadding = 0.07;

    std::vector<pangolin::Var<float> *> sizeVars;

#ifdef DEPTH_SOURCE_IMAGE
    // initialize depth source

    dart::ImageDepthSource<ushort,uchar3> * depthSource = new dart::ImageDepthSource<ushort,uchar3>();
    depthSource->initialize(videoLoc+"/depth",dart::IMAGE_PNG,
                            make_float2(525/2,525/2),make_float2(160,120),
                            320,240,0.001,0);
//                            true,videoLoc+"/color",dart::IMAGE_PNG,320,240);

    // ----
#endif

#ifdef DEPTH_SOURCE_IMAGE_TEST
    dart::ImageDepthSource<uint16_t,uchar3> * depthSource = new dart::ImageDepthSource<uint16_t,uchar3>();
    depthSource->initialize("../test_video/",dart::IMAGE_PNG, make_float2(525/2,525/2),make_float2(320,240), 640,480,0.001);
    //depthSource->initialize("../test_video2/",dart::IMAGE_PNG, make_float2(525/2,525/2),make_float2(2,2), 5,4,0.001);
#endif

#ifdef DEPTH_SOURCE_IMAGE_MULTISENSE
    dart::ImageDepthSource<uint16_t,uchar3> * depthSource = new dart::ImageDepthSource<uint16_t,uchar3>();
    depthSource->initialize("../depth_grasping_bottle",dart::IMAGE_PNG, make_float2(556.183166504, 556.183166504),make_float2(512,512), 1024, 1024, 0.001);
#endif

#ifdef DEPTH_SOURCE_LCM
    // example of using log file
    // exit after failure, used to stop publishing when end of logfile is reached
//    LCM_CommonBase::exitOnFailure(true);
    //LCM_CommonBase::setProvider("file:///home/christian/Downloads/logs/20160623_cr_arm_moving_IR_pattern/bottle_move.lcmlog");
//    LCM_CommonBase::setProvider("file:///home/christian/Downloads/logs/20160623_cr_arm_moving_IR_pattern/bottle_move-filtered_b120.lcmlog");
    //LCM_CommonBase::setProvider("file:///home/christian/Downloads/logs/20160623_cr_arm_moving_IR_pattern/box_move.lcmlog");
//    LCM_CommonBase::setProvider("file:///home/christian/Downloads/logs/20160727_cr-hand-movement-with-vicon-marker/vicon-arm_movement.lcmlog");
//    LCM_CommonBase::setProvider("file:///home/christian/Downloads/logs/20160727_cr-hand-movement-with-vicon-marker/vicon-finger_movement.lcmlog");
#endif

#ifdef DEPTH_SOURCE_LCM_MULTISENSE
    // Valkyrie Unit D, MultiSense SL
    dart::StereoCameraParameter val_multisense;
    val_multisense.focal_length = make_float2(556.183166504, 556.183166504);
    val_multisense.camera_center = make_float2(512, 512);
    val_multisense.baseline = 0.07;
    val_multisense.width = 1024;
    val_multisense.height = 1024;
    val_multisense.subpixel_resolution = 1.0/16.0;

    // initialise LCM depth source and listen on channel "CAMERA" in a separate thread
    dart::LCM_DepthSource<float,uchar3> *depthSource = new dart::LCM_DepthSource<float,uchar3>(val_multisense);
    depthSource->subscribe_images("CAMERA");
    //depthSource->subscribe_images("CAMERA_FILTERED");
    depthSource->setMaxDepthDistance(1.0); // meter

    const std::string cam_frame_name = "left_camera_optical_frame_joint";
#endif

#ifdef DEPTH_SOURCE_LCM_XTION
    // Valkyrie Unit D, Asus Xtion PRO Live
    dart::StereoCameraParameter val_xtion;
    val_xtion.focal_length = make_float2(528.01442863461716, 528.01442863461716);
    val_xtion.camera_center = make_float2(320, 267);
    val_xtion.baseline = 0.075;
    val_xtion.width = 640;
    val_xtion.height = 480;
    val_xtion.depth_resolution = 1.0/1000.0;

    // initialise LCM depth source and listen on channel "CAMERA" in a separate thread
    dart::LCM_DepthSource<float,uchar3> *depthSource = new dart::LCM_DepthSource<float,uchar3>(val_xtion);
    depthSource->subscribe_images("OPENNI_FRAME");

    const std::string cam_frame_name = "head_xtion_joint";
#endif

    tracker.addDepthSource(depthSource);
    dart::Optimizer & optimizer = *tracker.getOptimizer();

    const static int obsSdfSize = 64;
    const static float obsSdfResolution = 0.01*32/obsSdfSize;
    const static float defaultModelSdfResolution = 2e-3; //1.5e-3;
    const static float3 obsSdfOffset = make_float3(0,0,0.1);

    pangolin::Var<float> modelSdfResolution("lim.modelSdfResolution",defaultModelSdfResolution,defaultModelSdfResolution/2,defaultModelSdfResolution*2);
    pangolin::Var<float> modelSdfPadding("lim.modelSdfPadding",defaultModelSdfPadding,defaultModelSdfPadding/2,defaultModelSdfPadding*2);

#ifdef ENABLE_JUSTIN
    dart::ParamMapPoseReduction * handPoseReduction = dart::loadParamMapPoseReduction("../models/spaceJustin/justinHandParamMap.txt");

    tracker.addModel("../models/spaceJustin/spaceJustinHandRight.xml",
                     modelSdfResolution,
                     modelSdfPadding,
                     obsSdfSize,
                     obsSdfResolution,
                     make_float3(-0.5*obsSdfSize*obsSdfResolution) + obsSdfOffset, //);
                     handPoseReduction);


    dart::PosePrior reportedPosePrior(tracker.getPose(0).getReducedDimensions());
    memset(reportedPosePrior.getWeights(),0,6*sizeof(float));

    dart::HostOnlyModel spaceJustin;
    dart::readModelXML("../models/spaceJustin/spaceJustinArmsAndHead.xml",spaceJustin);

    spaceJustin.computeStructure();

    dart::LinearPoseReduction * justinPoseReduction = dart::loadLinearPoseReduction("../models/spaceJustin/spaceJustinPoseReduction.txt");

    dart::Pose spaceJustinPose(justinPoseReduction);
//    std::cout << spaceJustinPose.getReducedArticulatedDimensions() << " full justin articulated dimensions" << std::endl;

    tracker.addModel(objectModelFile,
                     0.5*modelSdfResolution,
                     modelSdfPadding,
                     64);
//                     objObsSdfRes,
//                     objObsSdfOffset);

    tracker.addModel("../models/spaceJustin/spaceJustinHandLeft.xml",
                     //"../models/spaceJustinArms.xml",
                     //0.1,
                     modelSdfResolution,
                     modelSdfPadding,
                     obsSdfSize,
                     obsSdfResolution,
                     make_float3(-0.5*obsSdfSize*obsSdfResolution) + obsSdfOffset,
                     handPoseReduction);
#endif

#ifdef ENABLE_URDF
    // original Valkyrie model
    //const std::string urdf_model_path = "../models/val_description/urdf/valkyrie_sim.urdf";
    // Valkyrie with attached Asus Xtion PRO LIVE
    //const std::string urdf_model_path = "../models/val_description/urdf/valkyrie_with_xtion.urdf";
    const std::string urdf_model_path = "../models/val_description/urdf/valkyrie_sim.urdf";

    // add Valkyrie
    dart::HostOnlyModel val = dart::readModelURDF(urdf_model_path, "pelvis");

    std::cout<<"found robot: "<<val.getName()<<std::endl;

    // initialize pose with 0 joint values and no full to reduced mapping
    dart::Pose val_pose = nullReductionPose(val);
    val_pose.zero();

    // joints/frame IDs for finding transformations
    const int val_cam_frame_id = val.getJointFrame(val.getJointIdByName(cam_frame_name));

#ifdef WITH_BOTTLE
    // track bottle
    dart::HostOnlyModel bottle = dart::readModelURDF("../models/bottle/bottle.urdf");
    tracker.addModel(bottle, 0.5*modelSdfResolution, modelSdfPadding, 64);
    // initial bottle pose in camera coordinate system, transformation camera to bottle
    // rotation according to Tait-Bryan angles: Z_1 Y_2 X_3
    // e.g. first: rotation around Z-axis, second: rotation around Y-axis, third: rotation around X-axis

    // multisense
#ifdef DEPTH_SOURCE_LCM_MULTISENSE
    // lcmlog__2016-04-20__18-51-14-226028__yy-wxm-table-grasping-left-hand
    // t>1150s
    const dart::SE3 T_cb = dart::SE3FromTranslation(0.244, -0.3036, 0.5952) * dart::SE3FromEuler(make_float3(0.2244, -0.594, -0.6732));

    // lcmlog__2016-06-23__13-36-34-184696__cr-ir_pattern2
//    const dart::SE3 T_cb = dart::SE3FromTranslation(-0.0357, -0.1905, 0.5833) * dart::SE3FromEuler(make_float3(-0.4862, 0.1333, -0.9519));

    // dbg: place object far away
    // workaround for data points no being assigned to single object when tracking
//    const dart::SE3 T_cb = dart::SE3FromTranslation(-1, -1, 1) * dart::SE3FromEuler(make_float3(-0.4862, 0.1333, -0.9519));
#endif
#ifdef DEPTH_SOURCE_LCM_XTION
    // asus xtion
    //const dart::SE3 T_cb = dart::SE3FromTranslation(0.1131, -0.07738, 0.6071) * dart::SE3FromEuler(make_float3(2.02, -2.02, 0.4862));
    //const dart::SE3 T_cb = dart::SE3FromTranslation(0.1131, -0.07738, 0.6071) * dart::SE3FromRotationX(0.4862) * dart::SE3FromRotationY(-2.02) * dart::SE3FromRotationZ(2.02);
    const dart::SE3 T_cb = dart::SE3FromTranslation(0.1131, -0.07738, 0.6071) * dart::SE3FromRotationX(0.4862) * dart::SE3FromRotationY(-2.02+1.04) * dart::SE3FromRotationZ(2.02);
#endif
#endif // BOTTLE

#ifdef WITH_BOX
    // track box
    dart::HostOnlyModel box = dart::readModelURDF("../models/box/box.urdf");
    tracker.addModel(box, 0.5*modelSdfResolution, modelSdfPadding, 64);
    // initial bottle pose in camera coordinate system, transformation camera to bottle
    // rotation according to Tait-Bryan angles: Z_1 Y_2 X_3
    // e.g. first: rotation around Z-axis, second: rotation around Y-axis, third: rotation around X-axis

    // lcmlog__2016-06-23__13-41-58-648227__cr-ir_pattern3
    const dart::SE3 T_cb = dart::SE3FromTranslation(-0.1012, -0.2143, 0.5655) * dart::SE3FromEuler(make_float3(1.122, 0.5984, -1.085));
#endif

#ifdef WITH_RECT
    dart::HostOnlyModel object = dart::readModelURDF("../models/rect_bar/rect.urdf");
    tracker.addModel(object, 0.5*modelSdfResolution, modelSdfPadding, 64);
    const dart::SE3 T_cb = dart::SE3FromTranslation(0.2381, -0.2917, 0.5774) * dart::SE3FromEuler(make_float3(0.4488, -0.3366, 0.7854));
#endif

    // track subparts of Valkyrie
    //const std::vector<uint8_t> colour_estimated_model = {255, 127, 0}; // orange
    const std::vector<uint8_t> colour_estimated_model = {255, 200, 0}; // yellow-orange
    dart::HostOnlyModel val_torso = dart::readModelURDF(urdf_model_path, "torso", "obj", colour_estimated_model);

    const int val_torso_cam_frame_id = val_torso.getJointFrame(val_torso.getJointIdByName(cam_frame_name));

    tracker.addModel(val_torso,
                     modelSdfResolution,    // modelSdfResolution, def = 0.002
                     modelSdfPadding,       // modelSdfPadding, def = 0.07
                     obsSdfSize,
                     obsSdfResolution,
                     make_float3(-0.5*obsSdfSize*obsSdfResolution) + obsSdfOffset,
                     0,         // poseReduction
                     1e5,       // collisionCloudDensity (def = 1e5)
                     true      // cacheSdfs
                     );

    // position priors
    // define 4 corresponding points in world camera and valkyrie camera frame
    // to fix head to reported head pose
//    const float point_weight = 1000000000;
//    dart::Point3D3DPrior val_camera_origin0(tracker.getModelIDbyName("valkyrie"), val_torso_cam_frame_id, make_float3(0, 0, 0), make_float3(0, 0, 0), point_weight);
//    dart::Point3D3DPrior val_camera_origin1(tracker.getModelIDbyName("valkyrie"), val_torso_cam_frame_id, make_float3(1, 0, 0), make_float3(1, 0, 0), point_weight);
//    dart::Point3D3DPrior val_camera_origin2(tracker.getModelIDbyName("valkyrie"), val_torso_cam_frame_id, make_float3(0, 1, 0), make_float3(0, 1, 0), point_weight);
//    dart::Point3D3DPrior val_camera_origin3(tracker.getModelIDbyName("valkyrie"), val_torso_cam_frame_id, make_float3(0, 0, 1), make_float3(0, 0, 1), point_weight);

//    tracker.addPrior(&val_camera_origin0);
//    tracker.addPrior(&val_camera_origin1);
//    tracker.addPrior(&val_camera_origin2);
//    tracker.addPrior(&val_camera_origin3);

    // weighted L2 norm
//    dart::WeightedL2NormOfError val_rep(tracker.getModelIDbyName("valkyrie"), val_pose, tracker.getPose("valkyrie"), 1);

    // individually weighted joints
//    const unsigned int val_torso_dims = tracker.getPose(tracker.getModelIDbyName("valkyrie")).getReducedArticulatedDimensions();
//    // TODO: replace Q by sparse matrix
//    Eigen::MatrixXf Q = 1 * Eigen::MatrixXf::Identity(val_torso_dims, val_torso_dims);

//    // change weights for left fingers in index 11..23
//    for(unsigned int i=11; i<=23; i++) {
//        Q(i,i) = 0.2;
//        //Q(i,i) = 2.0;
//    }
//    // change weights for left wrist roll/pitch in index 6,7
//    for(unsigned int i=6; i<=7; i++) {
//        Q(i,i) = 1;
//        //Q(i,i) = 5;
//        //Q(i,i) = 25;
//    }

//    dart::QWeightedError val_rep(tracker.getModelIDbyName("valkyrie"), val_pose, tracker.getPose("valkyrie"), Q);

//    tracker.addPrior(&val_rep);

    // prevent movement of the camera frame by enforcing no transformation
    dart::NoCameraMovementPrior val_cam(tracker.getModelIDbyName("valkyrie"));
    tracker.addPrior(&val_cam);
#endif

    std::cout<<"added models: "<<tracker.getNumModels()<<std::endl;

    std::vector<pangolin::Var<float> * *> poseVars;

    pangolin::Var<bool> sliderControlled("pose.sliderControl",false,true);
    for (int m=0; m<tracker.getNumModels(); ++m) {

        const int dimensions = tracker.getModel(m).getPoseDimensionality();

        pangolin::Var<float> * * vars = new pangolin::Var<float> *[dimensions];
        poseVars.push_back(vars);
        poseVars[m][0] = new pangolin::Var<float>(dart::stringFormat("pose.%d x",m),0,-0.5,0.5);
        poseVars[m][1] = new pangolin::Var<float>(dart::stringFormat("pose.%d y",m),0,-0.5,0.5);
        poseVars[m][2] = new pangolin::Var<float>(dart::stringFormat("pose.%d z",m),0.3,0.5,1.5);
        poseVars[m][3] = new pangolin::Var<float>(dart::stringFormat("pose.%d wx",m),    0,-M_PI,M_PI);
        poseVars[m][4] = new pangolin::Var<float>(dart::stringFormat("pose.%d wy",m),    0,-M_PI,M_PI);
        poseVars[m][5] = new pangolin::Var<float>(dart::stringFormat("pose.%d wz",m), M_PI,-M_PI,M_PI);

        const dart::Pose & pose = tracker.getPose(m);
        for (int i=0; i<pose.getReducedArticulatedDimensions(); ++i) {
            poseVars[m][i+6] = new pangolin::Var<float>(dart::stringFormat("pose.%d %s",m,pose.getReducedName(i).c_str()),0,pose.getReducedMin(i),pose.getReducedMax(i));
        }

    }

    // pangolin variables
//    static pangolin::Var<bool> trackFromVideo("ui.track",false,false,true);
    static pangolin::Var<bool> trackFromVideo("ui.track",true,false,true);
    static pangolin::Var<bool> stepVideo("ui.stepVideo",false,false);
    static pangolin::Var<bool> stepVideoBack("ui.stepVideoBack",false,false);
#ifdef ENABLE_URDF
    static pangolin::Var<bool> resetRobotPose("ui.resetRobotPose",false,false);
    static pangolin::Var<bool> useReportedPose("ui.useReportedPose",false,true);
#endif

    static pangolin::Var<float> sigmaPixels("ui.sigmaPixels",3.0,0.01,4);
    static pangolin::Var<float> sigmaDepth("ui.sigmaDepth",0.1,0.001,1);
    static pangolin::Var<float> focalLength("ui.focalLength",depthSource->getFocalLength().x,0.8*depthSource->getFocalLength().x,1.2*depthSource->getFocalLength().x);//475,525); //525.0,450.0,600.0);
    //static pangolin::Var<float> focalLength_y("ui.focalLength_y",depthSource->getFocalLength().y, 500, 1500);
    static pangolin::Var<bool> showCameraPose("ui.showCameraPose",false,true);
    static pangolin::Var<bool> showEstimatedPose("ui.showEstimate",true,true);
    //static pangolin::Var<bool> showEstimatedPose("ui.showEstimate",false,true);
    //static pangolin::Var<bool> showReported("ui.showReported",false,true);
    static pangolin::Var<bool> showReported("ui.showReported",true,true);

#ifdef JUSTIN
    static pangolin::Var<bool> showTablePlane("ui.showTablePlane",false,true);
#endif

    static pangolin::Var<bool> showVoxelized("ui.showVoxelized",false,true);
    static pangolin::Var<float> levelSet("ui.levelSet",0.0,-10.0,10.0);
    static pangolin::Var<bool> showTrackedPoints("ui.showPoints",true,true);
#ifdef DEPTH_SOURCE_LCM
    static pangolin::Var<bool> showPointColour("ui.showColour",true,true);
#else
    static pangolin::Var<bool> showPointColour("ui.showColour",false,true);
#endif
    static pangolin::Var<int> pointColoringObs("ui.pointColoringObs",0,0,NumPointColorings-1);
    static pangolin::Var<int> pointColoringPred("ui.pointColoringPred",0,0,NumPointColorings-1);
#ifdef JUSTIN
    static pangolin::Var<float> planeOffset("ui.planeOffset",-0.03,-0.05,0);
#endif

    static pangolin::Var<int> debugImg("ui.debugImg",DebugN,0,DebugN);

    static pangolin::Var<bool> showObsSdf("ui.showObsSdf",false,true);
    static pangolin::Var<bool> showPredictedPoints("ui.showPredictedPoints",false,true);
    static pangolin::Var<bool> showCollisionClouds("ui.showCollisionClouds",false,true);

    static pangolin::Var<bool> record("ui.Record Start/Stop",false,false);
    static pangolin::Var<float> fps("ui.fps",0);

    // optimization options
    pangolin::Var<bool> iterateButton("opt.iterate",false,false);
    pangolin::Var<int> itersPerFrame("opt.itersPerFrame",3,0,30);
    pangolin::Var<float> normalThreshold("opt.normalThreshold",-1.01,-1.01,1.0);
    pangolin::Var<float> distanceThreshold("opt.distanceThreshold",0.035,0.0,0.1);
    pangolin::Var<float> handRegularization("opt.handRegularization",0.1,0,10); // 1.0
    pangolin::Var<float> objectRegularization("opt.objectRegularization",0.1,0,10); // 1.0
    pangolin::Var<float> resetInfoThreshold("opt.resetInfoThreshold",1.0e-5,1e-5,2e-5);
    pangolin::Var<float> stabilityThreshold("opt.stabilityThreshold",7.5e-6,5e-6,1e-5);
    pangolin::Var<float> lambdaModToObs("opt.lambdaModToObs",0.5,0,1);
    pangolin::Var<float> lambdaObsToMod("opt.lambdaObsToMod",1,0,1);
#ifdef USE_CONTACT_PRIOR
    pangolin::Var<float> lambdaIntersection("opt.lambdaIntersection",1.f,0,40);
    //pangolin::Var<float> selfIntersectWeight("opt.selfIntersectWeight",atof(argv[2]),0,40);
    pangolin::Var<float> lambdaContact("opt.lambdaContact",1.f,0,200);
#endif


    pangolin::Var<float> infoAccumulationRate("opt.infoAccumulationRate",0.1,0.0,1.0); // 0.8
    pangolin::Var<float> maxRotationDamping("opt.maxRotationalDamping",50,0,200);
    pangolin::Var<float> maxTranslationDamping("opt.maxTranslationDamping",5,0,10);

#ifdef JUSTIN
    pangolin::Var<float> tableNormX("opt.tableNormX",initialTableNorm.x,-1,1);
    pangolin::Var<float> tableNormY("opt.tableNormY",initialTableNorm.y,-1,1);
    pangolin::Var<float> tableNormZ("opt.tableNormZ",initialTableNorm.z,-1,1);
    pangolin::Var<float> tableIntercept("opt.tableIntercept",initialTableIntercept,-1,1);
    static pangolin::Var<bool> fitTable("opt.fitTable",true,true);
    static pangolin::Var<bool> subtractTable("opt.subtractTable",true,true);
#endif

#ifdef USE_CONTACT_PRIOR
    static pangolin::Var<bool> * contactVars[10];
    contactVars[0] = new pangolin::Var<bool>("lim.contactThumbR",false,true);
    contactVars[1] = new pangolin::Var<bool>("lim.contactIndexR",false,true);
    contactVars[2] = new pangolin::Var<bool>("lim.contactMiddleR",false,true);
    contactVars[3] = new pangolin::Var<bool>("lim.contactRingR",false,true);
    contactVars[4] = new pangolin::Var<bool>("lim.contactLittleR",false,true);
    contactVars[5] = new pangolin::Var<bool>("lim.contactThumbL",false,true);
    contactVars[6] = new pangolin::Var<bool>("lim.contactIndexL",false,true);
    contactVars[7] = new pangolin::Var<bool>("lim.contactMiddleL",false,true);
    contactVars[8] = new pangolin::Var<bool>("lim.contactRingL",false,true);
    contactVars[9] = new pangolin::Var<bool>("lim.contactLittleL",false,true);
    bool anyContact = false;
#endif

    int fpsWindow = 10;
    pangolin::basetime lastTime = pangolin::TimeNow();

    const int depthWidth = depthSource->getDepthWidth();
    const int depthHeight = depthSource->getDepthHeight();

    const int predWidth = tracker.getPredictionWidth();
    const int predHeight = tracker.getPredictionHeight();

    dart::MirroredVector<uchar3> imgDepthSize(depthWidth*depthHeight);
    dart::MirroredVector<uchar3> imgPredSize(predWidth*predHeight);
    dart::MirroredVector<const uchar3 *> allSdfColors(tracker.getNumModels());
    for (int m=0; m<tracker.getNumModels(); ++m) {
        allSdfColors.hostPtr()[m] = tracker.getModel(m).getDeviceSdfColors();
    }
    allSdfColors.syncHostToDevice();

    // set up VBO to display point cloud
    GLuint pointCloudVbo,pointCloudColorVbo,pointCloudNormVbo;
    glGenBuffersARB(1,&pointCloudVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);
    glGenBuffersARB(1,&pointCloudColorVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
    glGenBuffersARB(1,&pointCloudNormVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);


    dart::OptimizationOptions & opts = tracker.getOptions();
    opts.lambdaObsToMod = 1;
    memset(opts.lambdaIntersection.data(),0,tracker.getNumModels()*tracker.getNumModels()*sizeof(float));
#ifdef USE_CONTACT_PRIOR
    opts.contactThreshold = 0.02;
#endif
#ifdef JUSTIN
    opts.planeNormal[0] =  make_float3(0,0,1);
    opts.planeNormal[2] = make_float3(0,0,1);
    opts.planeNormal[1] = make_float3(0,0,0);
#endif
    opts.regularization[0] = opts.regularization[1] = opts.regularization[2] = 0.01;

#ifdef USE_CONTACT_PRIOR
    float3 initialContact = make_float3(0,0.02,0);

    std::vector<dart::ContactPrior *> contactPriors;
    for (int i=0; i<5; ++i) {
        dart::ContactPrior * prior = new dart::ContactPrior(0, 1, 3*(1+i), 0, 0.0, initialContact, 100);
        contactPriors.push_back(prior);
        tracker.addPrior(prior);
    }
    for (int i=0; i<5; ++i) {

        dart::ContactPrior * prior = new dart::ContactPrior(2, 1, 3*(1+i), 0, 0.0, initialContact, 100);
        contactPriors.push_back(prior);
        tracker.addPrior(prior);
    }
#endif

#ifdef ENABLE_JUSTIN
    // set up potential intersections
    {

        int * selfIntersectionMatrix = dart::loadSelfIntersectionMatrix("../models/spaceJustin/justinIntersection.txt",tracker.getModel(0).getNumSdfs());

        tracker.setIntersectionPotentialMatrix(0,selfIntersectionMatrix);
        tracker.setIntersectionPotentialMatrix(2,selfIntersectionMatrix);

        delete [] selfIntersectionMatrix;

    }
#endif

#ifdef ENABLE_JUSTIN
    dart::MirroredModel & rightHand = tracker.getModel(0);
    dart::MirroredModel & leftHand = tracker.getModel(2);
    dart::MirroredModel & object = tracker.getModel(1);

    dart::Pose & rightHandPose = tracker.getPose(0);
    dart::Pose & leftHandPose = tracker.getPose(2);
    dart::Pose & objectPose = tracker.getPose(1);
#endif

#ifdef ENABLE_URDF
    // get references to model and its pose for tracking
    dart::MirroredModel & val_torso_mm = tracker.getModel(tracker.getModelIDbyName("valkyrie"));
    dart::Pose & val_torso_pose = tracker.getPose("valkyrie");
#ifdef WITH_BOTTLE
    dart::Pose & bottle_pose = tracker.getPose("bottle");
#endif
#ifdef WITH_BOX
    dart::Pose & box_pose = tracker.getPose("box");
#endif
#ifdef WITH_RECT
    dart::Pose &object_pose = tracker.getPose("rect");
#endif
#endif


#ifdef ENABLE_JUSTIN
    // set up reported pose offsets
    std::vector<float *> reportedJointAngles;
    loadReportedJointAngles(videoLoc+"/reportedJointAngles.txt", reportedJointAngles);
#endif

#ifdef USE_CONTACT_PRIOR
    std::vector<int *> reportedContacts;
    loadReportedContacts(videoLoc+"/reportedContacts.txt", reportedContacts);
#endif

#ifdef ENABLE_JUSTIN
    std::cout << "loaded " << reportedJointAngles.size() << " frames" << std::endl;
#endif

#ifdef ENABLE_LCM_JOINTS
    // measures joint values for reported robot configuration
    dart::LCM_JointsProvider lcm_joints;
    lcm_joints.setJointNames(val);
    // listen on channel "EST_ROBOT_STATE" in a separate thread
    lcm_joints.subscribe_robot_state(LCM_CHANNEL_ROBOT_STATE);

    dart::LCM_StatePublish lcm_robot_state(LCM_CHANNEL_ROBOT_STATE, LCM_CHANNEL_DART_PREFIX, val_torso_pose);
    dart::LCM_FramePosePublish lcm_frame_pub("DART", val, val_torso_mm);

#ifdef WITH_BOTTLE
    dart::MirroredModel & bottle_mm = tracker.getModel(tracker.getModelIDbyName("bottle"));
    dart::LCM_FramePosePublish lcm_object_frame_pub("DART", bottle, bottle_mm);
#endif
#endif

    // -=-=-=-=- set up initial poses -=-=-=-=-
#ifdef ENABLE_JUSTIN
    spaceJustinPose.setTransformModelToCamera(initialT_cj);
    memcpy(spaceJustinPose.getReducedArticulation(),reportedJointAngles[depthSource->getFrame()],spaceJustinPose.getReducedArticulatedDimensions()*sizeof(float));
    spaceJustinPose.projectReducedToFull();
    spaceJustin.setPose(spaceJustinPose);

    rightHandPose.setTransformModelToCamera(spaceJustin.getTransformFrameToCamera(rightPalmFrame)*T_wh);
    memcpy(rightHandPose.getReducedArticulation(),spaceJustinPose.getReducedArticulation() + 7,rightHandPose.getReducedArticulatedDimensions()*sizeof(float));
    rightHand.setPose(rightHandPose);

    leftHandPose.setTransformModelToCamera(spaceJustin.getTransformFrameToCamera(leftPalmFrame)*T_wh);
    memcpy(leftHandPose.getReducedArticulation(),spaceJustinPose.getReducedArticulation() + 7 + 15 + 7,leftHandPose.getReducedArticulatedDimensions()*sizeof(float));
    leftHand.setPose(leftHandPose);
#endif

#ifdef ENABLE_URDF
    // wait to get initial configuration of robot from LCM thread
    usleep(100000);
    // set initial state of tracked model
    val_torso_pose.setReducedArticulation(lcm_joints.getJointsNameValue());
    val_torso_mm.setPose(val_torso_pose);
    dart::SE3 Tmc = val_torso_mm.getTransformModelToFrame(val_torso_cam_frame_id);
    val_torso_pose.setTransformModelToCamera(Tmc);
#ifdef WITH_BOTTLE
    bottle_pose.setTransformModelToCamera(T_cb);
    bottle.setPose(bottle_pose);
#endif
#ifdef WITH_BOX
    box_pose.setTransformModelToCamera(T_cb);
    box.setPose(box_pose);
#endif
#ifdef WITH_RECT
    object_pose.setTransformModelToCamera(T_cb);
    object.setPose(object_pose);
#endif
#endif

#ifdef ENABLE_JUSTIN
    const dart::SE3 T_camera_head = spaceJustin.getTransformFrameToCamera(headFrame);

    objectPose.setTransformModelToCamera(initialT_co);
    object.setPose(objectPose);

    if ( isnan(spaceJustinPose.getArticulation()[0])) {
        std::cerr << "???" << std::endl;
        spaceJustinPose.projectReducedToFull();
    }

    TrackingMode trackingMode = ModeObjOnTable;
#endif

    // ------------------- main loop ---------------------
    for (int pangolinFrame=1; !pangolin::ShouldQuit(); ++pangolinFrame) {

        if (pangolin::HasResized()) {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

#ifdef ENABLE_URDF
        tracker.stepForward();
#endif

#ifdef ENABLE_LCM_JOINTS
#ifdef ENABLE_URDF
        // get reported Valkyrie configuration
        val_pose.setReducedArticulation(lcm_joints.getJointsNameValue());
        // transform coordinate origin to camera image centre
        dart::SE3 Tmc = val.getTransformModelToFrame(val_cam_frame_id);
        val_pose.setTransformModelToCamera(Tmc);
#endif
#endif

#ifdef ENABLE_URDF
        if(pangolin::Pushed(resetRobotPose) || useReportedPose) {
#ifdef ENABLE_LCM_JOINTS
            val_torso_pose.setReducedArticulation(lcm_joints.getJointsNameValue());
#endif
            val_torso_mm.setPose(val_torso_pose);
            dart::SE3 Tmc = val_torso_mm.getTransformModelToFrame(val_torso_cam_frame_id);
            val_torso_pose.setTransformModelToCamera(Tmc);
        }
#endif

#ifdef ENABLE_JUSTIN
        static pangolin::Var<std::string> trackingModeStr("ui.mode");
        trackingModeStr = getTrackingModeString(trackingMode);

        opts.lambdaIntersection[0 + 3*0] = lambdaIntersection; // right
        opts.lambdaIntersection[2 + 3*2] = lambdaIntersection; // left

        opts.lambdaIntersection[1 + 3*0] = lambdaIntersection; // object->right
        opts.lambdaIntersection[0 + 3*1] = lambdaIntersection; // right->object

        opts.lambdaIntersection[1 + 3*2] = lambdaIntersection; // object->left
        opts.lambdaIntersection[2 + 3*1] = lambdaIntersection; // left->object
#endif

        opts.focalLength = focalLength;
        opts.normThreshold = normalThreshold;
        for (int m=0; m<tracker.getNumModels(); ++m) {
            opts.distThreshold[m] = distanceThreshold;
        }
        opts.regularization[0] = opts.regularization[1] = opts.regularization[2] = 0.01;
#ifdef ENABLE_JUSTIN
        opts.regularizationScaled[0] = handRegularization;
        opts.regularizationScaled[1] = objectRegularization;
        opts.regularizationScaled[2] = handRegularization;
#endif
#ifdef JUSTIN
        opts.planeOffset[2] = planeOffset;
#endif
        opts.lambdaObsToMod = lambdaObsToMod;
        opts.lambdaModToObs = lambdaModToObs;
#ifdef JUSTIN
        opts.planeOffset[0] = planeOffset;
#endif
        opts.debugObsToModDA = pointColoringObs == PointColoringDA || (debugImg == DebugObsToModDA);
        opts.debugModToObsDA = debugImg == DebugModToObsDA;
        opts.debugObsToModErr = ((pointColoringObs == PointColoringErr) || (debugImg == DebugObsToModErr));
        opts.debugModToObsErr = ((pointColoringPred == PointColoringErr) || (debugImg == DebugModToObsErr));
        opts.debugJTJ = (debugImg == DebugJTJ);
        opts.numIterations = itersPerFrame;

        if (pangolin::Pushed(stepVideoBack)) {
            tracker.stepBackward();
        }

        bool iteratePushed = Pushed(iterateButton);

        if (pangolinFrame % fpsWindow == 0) {
            pangolin::basetime time = pangolin::TimeNow();
            if (trackFromVideo) {
                static int totalFrames = 0;
                static double totalTime = 0;
                totalFrames += fpsWindow;
                totalTime += pangolin::TimeDiff_s(lastTime,time);
                fps = totalFrames / totalTime;
            } else {
                fps = fpsWindow / pangolin::TimeDiff_s(lastTime,time);
            }
            lastTime = time;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //
        //
        // Process this frame                                                                                   //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        {

            static pangolin::Var<bool> filteredNorms("ui.filteredNorms",false,true);
            static pangolin::Var<bool> filteredVerts("ui.filteredVerts",false,true);

            if (filteredNorms.GuiChanged()) {
                tracker.setFilteredNorms(filteredNorms);
            } else if (filteredVerts.GuiChanged()) {
                tracker.setFilteredVerts(filteredVerts);
            } else if (sigmaDepth.GuiChanged()) {
                tracker.setSigmaDepth(sigmaDepth);
            } else if (sigmaPixels.GuiChanged()) {
                tracker.setSigmaPixels(sigmaPixels);
            }

            // update pose based on sliders
            if (sliderControlled) {
                for (int m=0; m<tracker.getNumModels(); ++m) {
                    for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
                        tracker.getPose(m).getReducedArticulation()[i] = *poseVars[m][i+6];
                    }
                    tracker.getPose(m).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(*poseVars[m][0],*poseVars[m][1],*poseVars[m][2],0,0,0))*
                            dart::SE3Fromse3(dart::se3(0,0,0,*poseVars[m][3],*poseVars[m][4],*poseVars[m][5])));
                    tracker.updatePose(m);
                }
            }

            // run optimization method
            if (trackFromVideo || iteratePushed ) {

                // workaround: we need to wait 1 frame before starting optimization
                // otherwise, the no movement prior produces a wrong update
                if(pangolinFrame>1)
                    tracker.optimizePoses();

                // update accumulated info
                for (int m=0; m<tracker.getNumModels(); ++m) {
#ifdef ENABLE_JUSTIN
                    if (m == 1 && trackingMode == ModeIntermediate) { continue; }
#endif
                    const Eigen::MatrixXf & JTJ = *tracker.getOptimizer()->getJTJ(m);
                    if (JTJ.rows() == 0) { continue; }
                    Eigen::MatrixXf & dampingMatrix = tracker.getDampingMatrix(m);
                    for (int i=0; i<3; ++i) {
                        dampingMatrix(i,i) = std::min((float)maxTranslationDamping,dampingMatrix(i,i) + infoAccumulationRate*JTJ(i,i));
                    }
                    for (int i=3; i<tracker.getPose(m).getReducedDimensions(); ++i) {
                        dampingMatrix(i,i) = std::min((float)maxRotationDamping,dampingMatrix(i,i) + infoAccumulationRate*JTJ(i,i));
                    }
                }

                float errPerObsPoint = optimizer.getErrPerObsPoint(1,0);
                float errPerModPoint = optimizer.getErrPerModPoint(1,0);

                infoLog.Log(errPerObsPoint,errPerObsPoint+errPerModPoint,stabilityThreshold,resetInfoThreshold);

                for (int m=0; m<tracker.getNumModels(); ++m) {
                    for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
                        *poseVars[m][i+6] = tracker.getPose(m).getReducedArticulation()[i];
                    }
                    dart::SE3 T_cm = tracker.getPose(m).getTransformModelToCamera();
                    *poseVars[m][0] = T_cm.r0.w; T_cm.r0.w = 0;
                    *poseVars[m][1] = T_cm.r1.w; T_cm.r1.w = 0;
                    *poseVars[m][2] = T_cm.r2.w; T_cm.r2.w = 0;
                    dart::se3 t_cm = dart::se3FromSE3(T_cm);
                    *poseVars[m][3] = t_cm.p[3];
                    *poseVars[m][4] = t_cm.p[4];
                    *poseVars[m][5] = t_cm.p[5];
                }

#ifdef ENABLE_LCM_JOINTS
                // publish optimized pose
                lcm_robot_state.publish_estimate();
                // publish frame poses of reported and estimated model
                lcm_frame_pub.publish_frame_pose("leftWristPitch");

                //lcm_object_frame_pub.publish_frame_pose("joint1");
#endif
            }

        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                      //
        // Render this frame                                                                                    //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        glClearColor (1.0, 1.0, 1.0, 1.0);
        glShadeModel (GL_SMOOTH);
        float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
        glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

        camDisp.ActivateScissorAndClear(camState);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);

        camDisp.ActivateAndScissor(camState);

        glPushMatrix();
#ifdef ENABLE_URDF
        //static pangolin::Var<bool> showAxes("ui.showAxes",true,true);
        static pangolin::Var<bool> showAxes("ui.showAxes",false,true);
        if(showAxes) {
            // draw coordinates axes, x: red, y: green, z: blue
            pangolin::glDrawAxis(0.5);
            glColor3f(0,0,1);
            pangolin::glDraw_z0(0.01, 10);
        }
#endif

        if (showCameraPose) {

            glColor3f(0,0,0);
            glPushMatrix();

//            glRotatef(180,0,1,0);
//            glutSolidCube(0.02);

//            glTranslatef(0,0,-0.02);
//            glutSolidCone(0.0125,0.02,10,1);

            glPopMatrix();

        }

        glColor4ub(0xff,0xff,0xff,0xff);
        if (showEstimatedPose) {

            glEnable(GL_COLOR_MATERIAL);

            glPushMatrix();

            if (showVoxelized) {
                glColor3f(0.2,0.3,1.0);

                for (int m=0; m<tracker.getNumModels(); ++m) {
//                for (int m=1; m<=1; m+=10) {
                    tracker.updatePose(m);
                    tracker.getModel(m).renderVoxels(levelSet);
                }
            }
            else{
                for (int m=0; m<tracker.getNumModels(); ++m) {
                    tracker.updatePose(m);
                    tracker.getModel(m).render();
                }
            }

            glPopMatrix();

        }

        if (showReported) {
            glColor3ub(0xfa,0x85,0x7c);
            glEnable(GL_COLOR_MATERIAL);

#ifdef ENABLE_JUSTIN
            memcpy(spaceJustinPose.getReducedArticulation(),reportedJointAngles[depthSource->getFrame()],spaceJustinPose.getReducedArticulatedDimensions()*sizeof(float));
            spaceJustinPose.projectReducedToFull();
            spaceJustin.setPose(spaceJustinPose);
            spaceJustin.renderWireframe();
#endif

#ifdef ENABLE_URDF
            // render Valkyrie reported state as wireframe model, origin is the camera centre
            val.setPose(val_pose);
            val.renderWireframe();
#endif

            // glColor3ub(0,0,0);
            // glutSolidSphere(0.02,10,10);
        }

        glPointSize(1.0f);

#ifdef JUSTIN
        if (showTablePlane) {

            float3 normal = normalize(make_float3(tableNormX,tableNormY,tableNormZ));
            tableNormX = normal.x;
            tableNormY = normal.y;
            tableNormZ = normal.z;

            float3 ipv1 = cross(normal,normal.x == 1 ? make_float3(0,1,0) : make_float3(1,0,0));
            float3 ipv2 = cross(normal,ipv1);

            float3 pts[4] = { operator+(operator *( 0.5,ipv1),operator *( 0.5,ipv2)),
                              operator+(operator *( 0.5,ipv1),operator *(-0.5,ipv2)),
                              operator+(operator *(-0.5,ipv1),operator *(-0.5,ipv2)),
                              operator+(operator *(-0.5,ipv1),operator *( 0.5,ipv2))};

            glColor3ub(120,100,100);
            glBegin(GL_QUADS);
            glNormal3f(-normal.x,-normal.y,-normal.z);
            for (int i=0; i<4; ++i) {
                glVertex3f(tableIntercept*normal.x + pts[i].x,
                           tableIntercept*normal.y + pts[i].y,
                           tableIntercept*normal.z + pts[i].z);
            }
            glEnd();

        }
#endif

        if (showObsSdf) {
            static pangolin::Var<float> levelSet("ui.levelSet",0,-10,10);

            for (int m=0; m<tracker.getNumModels(); ++m) {

                glPushMatrix();
                dart::glMultSE3(tracker.getModel(m).getTransformModelToCamera());
                tracker.getModel(m).syncObsSdfDeviceToHost();
                dart::Grid3D<float> * obsSdf = tracker.getModel(m).getObsSdf();
                tracker.getModel(m).renderSdf(*obsSdf,levelSet);
                glPopMatrix();
            }
        }

        if (showTrackedPoints) {

            glPointSize(4.0f);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
            glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);

            glEnableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_NORMAL_ARRAY);
            glVertexPointer(4, GL_FLOAT, 0, 0);

            if(showPointColour)
                pointColoringObs = PointColoringRGB;
            else if(pointColoringObs == PointColoringRGB)
                pointColoringObs = PointColoringNone;

            switch (pointColoringObs) {
            case PointColoringNone:
                glColor3f(0.25,0.25,0.25);
                glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
                glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);

                glNormalPointer(GL_FLOAT, 4*sizeof(float), 0);
                glEnableClientState(GL_NORMAL_ARRAY);
                break;
            case PointColoringRGB:
                glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
                glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),depthSource->getColor(),GL_DYNAMIC_DRAW_ARB);
                glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
                glEnableClientState(GL_COLOR_ARRAY);
                glDisable(GL_LIGHTING);
                break;
            case PointColoringErr:
                {
                    static float errorMin = 0.0;
                    static float errorMax = 0.1;
                    float * dErr;
                    cudaMalloc(&dErr,depthWidth*depthHeight*sizeof(float));
                    dart::imageSquare(dErr,tracker.getDeviceDebugErrorObsToMod(),depthWidth,depthHeight);
                    dart::colorRampHeatMapUnsat(imgDepthSize.devicePtr(),dErr,depthWidth,depthHeight,errorMin,errorMax);
                    cudaFree(dErr);
                    imgDepthSize.syncDeviceToHost();
                    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
                    glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
                    glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
                    glEnableClientState(GL_COLOR_ARRAY);
                    glDisable(GL_LIGHTING);
                }
                break;
            case PointColoringDA:
                {
                    const int * dDebugDA = tracker.getDeviceDebugDataAssociationObsToMod();
                    dart::colorDataAssociationMultiModel(imgDepthSize.devicePtr(),dDebugDA,allSdfColors.devicePtr(),depthWidth,depthHeight);
                    imgDepthSize.syncDeviceToHost();
                    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
                    glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
                    glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
                    glEnableClientState(GL_COLOR_ARRAY);
                    glDisable(GL_LIGHTING);
                }
                break;
            }

            glDrawArrays(GL_POINTS,0,depthWidth*depthHeight);
            glBindBuffer(GL_ARRAY_BUFFER_ARB,0);

            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_NORMAL_ARRAY);

            glPointSize(1.0f);

        }

#ifdef USE_CONTACT_PRIOR
        static pangolin::Var<bool> showFingerContacts("ui.showContacts",true,true);
        if (showFingerContacts) {

            glPointSize(10.f);
            glBegin(GL_POINTS);
            for (int i=0; i<contactPriors.size(); ++i) {

                if (*contactVars[i]) {

                    const int model = contactPriors[i]->getSourceModel();
                    const int sdfNum = contactPriors[i]->getSourceSdfNum();
                    const int frameNum = tracker.getModel(model).getSdfFrameNumber(sdfNum);
                    const float3 contactPoint = contactPriors[i]->getContactPoint();

                    float4 contact_c = tracker.getModel(model).getTransformFrameToCamera(frameNum)*make_float4(contactPoint,1);

                    glColor3f(1.0,0,0);
                    glVertex3fv(&contact_c.x);

                }
            }
            glEnd();

        }
#endif

        if (showPredictedPoints) {

            glPointSize(4.0f);

            const float4 * dPredictedVertMap = tracker.getDevicePredictedVertMap();
            const int nPoints = tracker.getPredictionWidth()*tracker.getPredictionHeight();
            float4 * hPredictedVertMap = new float4[nPoints];
            cudaMemcpy(hPredictedVertMap,dPredictedVertMap,nPoints*sizeof(float4),cudaMemcpyDeviceToHost);


            glDisable(GL_LIGHTING);
            glBegin(GL_POINTS);

            if (pointColoringPred == PointColoringErr) {
                static pangolin::Var<float> errMin("ui.errMin",0,0,0.05);
                static pangolin::Var<float> errMax("ui.errMax",0.01,0,0.05);
                dart::MirroredVector<uchar3> debugImg(tracker.getPredictionWidth()*tracker.getPredictionHeight());
                dart::colorRampHeatMapUnsat(debugImg.devicePtr(),
                                            tracker.getDeviceDebugErrorModToObs(),
                                            depthWidth,depthHeight,
                                            errMin,errMax);
                debugImg.syncDeviceToHost();

                for (int i=0; i<nPoints; ++i) {
                    if (hPredictedVertMap[i].z > 0) {
                        uchar3 color = debugImg.hostPtr()[i];
                        glColor3ubv((unsigned char*)&color);
                        glVertex3f(hPredictedVertMap[i].x,hPredictedVertMap[i].y,hPredictedVertMap[i].z);
                    }
                }

            } else {

                for (int i=0; i<nPoints; ++i) {
                    if (hPredictedVertMap[i].z > 0) {
                        int id = round(hPredictedVertMap[i].w);
                        int model = id >> 16;
                        int sdf = id & 65535;
                        uchar3 color = tracker.getModel(model).getSdfColor(sdf);
                        glColor3ubv((unsigned char*)&color);
                        glVertex3f(hPredictedVertMap[i].x,hPredictedVertMap[i].y,hPredictedVertMap[i].z);
                    }
                }
            }

            glEnd();
            delete [] hPredictedVertMap;

            glPointSize(1.0f);
        }

        if (showCollisionClouds) {
            glPointSize(10);
            glColor3f(0,0,1.0f);
            glDisable(GL_LIGHTING);
            glBegin(GL_POINTS);
            for (int m=0; m<tracker.getNumModels(); ++m) {
                const float4 * collisionCloud = tracker.getCollisionCloud(m);
                for (int i=0; i<tracker.getCollisionCloudSize(m); ++i) {
                    int grid = round(collisionCloud[i].w);
                    int frame = tracker.getModel(m).getSdfFrameNumber(grid);
                    float4 v = tracker.getModel(m).getTransformModelToCamera()*
                               tracker.getModel(m).getTransformFrameToModel(frame)*
                               make_float4(make_float3(collisionCloud[i]),1.0);
                    glVertex3fv((float*)&v);
                }
            }
            glEnd();
            glEnable(GL_LIGHTING);

            glPointSize(1);
            glColor3f(1,1,1);
        }

        glPopMatrix();

        imgDisp.ActivateScissorAndClear();
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

        switch (debugImg) {
            case DebugColor:
            {
                if (depthSource->hasColor()) {
                    imgTexDepthSize.Upload(depthSource->getColor(),GL_RGB,GL_UNSIGNED_BYTE);
                    imgTexDepthSize.RenderToViewport();
                }
            }
            break;
        case DebugObsDepth:
            {

                static const float depthMin = 0.3;
                static const float depthMax = 1.0;

                //const unsigned short * depth = depthSource->getDepth();
                const auto * depth = depthSource->getDepth();

                for (int i=0; i<depthSource->getDepthWidth()*depthSource->getDepthHeight(); ++i) {
                    if (depth[i] == 0) {
                        imgDepthSize[i] = make_uchar3(128,0,0);
                    } else {
                        unsigned char g = std::max(0,std::min((int)(255*(depth[i]*depthSource->getScaleToMeters()-depthMin)/(float)(depthMax - depthMin)),255));
                        imgDepthSize[i] = make_uchar3(g,g,g);
                    }

                }

                imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
                imgTexDepthSize.RenderToViewport();
            }
        case DebugPredictedDepth:
            {
                static const float depthMin = 0.3;
                static const float depthMax = 1.0;

                const float4 * dPredictedVertMap = tracker.getDevicePredictedVertMap();
                static std::vector<float4> hPredictedVertMap(predWidth*predHeight);

                cudaMemcpy(hPredictedVertMap.data(),dPredictedVertMap,predWidth*predHeight*sizeof(float4),cudaMemcpyDeviceToHost);

                for (int i=0; i<predHeight*predWidth; ++i) {
                    const float depth = hPredictedVertMap[i].z;
                    if (depth == 0) {
                        imgPredSize[i] = make_uchar3(128,0,0);
                    } else {
                        unsigned char g = std::max(0,std::min((int)(255*(depth-depthMin)/(float)(depthMax - depthMin)),255));
                        imgPredSize[i] = make_uchar3(g,g,g);
                    }
                }

                imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
                imgTexPredictionSize.RenderToViewport();
            }
            break;
        case DebugObsToModDA:
        {
            dart::colorDataAssociationMultiModel(imgDepthSize.devicePtr(),
                                                 tracker.getDeviceDebugDataAssociationObsToMod(),
                                                 allSdfColors.devicePtr(),depthWidth,depthHeight);\
            imgDepthSize.syncDeviceToHost();
            imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
            imgTexDepthSize.RenderToViewport();
            break;
        }
        case DebugModToObsDA:
        {
            dart::colorDataAssociationMultiModel(imgPredSize.devicePtr(),
                                                 tracker.getDeviceDebugDataAssociationModToObs(),
                                                 allSdfColors.devicePtr(),predWidth,predHeight);\
            imgPredSize.syncDeviceToHost();
            imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
            imgTexPredictionSize.RenderToViewport();
            break;
        }
        case DebugObsToModErr:
            {
                static const float errMax = 0.01;
                dart::colorRampHeatMapUnsat(imgDepthSize.devicePtr(),
                                            tracker.getDeviceDebugErrorObsToMod(),
                                            depthWidth,depthHeight,
                                            0.f,errMax);
                imgDepthSize.syncDeviceToHost();
                imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
                imgTexDepthSize.RenderToViewport();
            }
            break;
        case DebugModToObsErr:
            {
                static const float errMax = 0.01;
                dart::colorRampHeatMapUnsat(imgPredSize.devicePtr(),
                                            tracker.getDeviceDebugErrorModToObs(),
                                            depthWidth,depthHeight,
                                            0.f,errMax);
                imgPredSize.syncDeviceToHost();
                imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
                imgTexPredictionSize.RenderToViewport();
            }
            break;
        case DebugJTJ:
            imgTexDepthSize.Upload(tracker.getOptimizer()->getJTJimg(),GL_RGB,GL_UNSIGNED_BYTE);
            imgTexDepthSize.RenderToViewportFlipY();
            break;
        default:
            break;
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << cudaGetErrorString(err) << std::endl;
        }

        if(pangolin::Pushed(record)) {
                pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");
        }

        pangolin::FinishFrame();

        if (pangolin::Pushed(stepVideo) || trackFromVideo || pangolinFrame == 1) {
#ifdef ENABLE_JUSTIN
            tracker.stepForward();

            const float * currentReportedPose = reportedJointAngles[depthSource->getFrame()];
            const float * lastReportedPose = (depthSource->getFrame()==0) ? currentReportedPose :  reportedJointAngles[depthSource->getFrame()-1];

            memcpy(spaceJustinPose.getReducedArticulation(),lastReportedPose,spaceJustinPose.getReducedArticulatedDimensions()*sizeof(float));
            spaceJustinPose.projectReducedToFull();
            spaceJustin.setPose(spaceJustinPose);

            dart::SE3 lastT_ro = rightHand.getTransformCameraToModel()*object.getTransformModelToCamera();
            lastT_ro = dart::SE3Fromse3(dart::se3FromSE3(lastT_ro));

            dart::SE3 lastT_lo = leftHand.getTransformCameraToModel()*object.getTransformModelToCamera();
            lastT_lo = dart::SE3Fromse3(dart::se3FromSE3(lastT_lo));

            dart::SE3 lastT_head_r = spaceJustin.getTransformModelToFrame(headFrame)*spaceJustin.getTransformFrameToModel(rightPalmFrame)*T_wh;
            dart::SE3 lastT_head_l = spaceJustin.getTransformModelToFrame(headFrame)*spaceJustin.getTransformFrameToModel(leftPalmFrame)*T_wh;

            // update justin 6DoF transform to keep the camera constant
            const dart::SE3 T_oldc_base = spaceJustinPose.getTransformModelToCamera();
            const dart::SE3 T_head_base = spaceJustin.getTransformModelToFrame(headFrame);
            const dart::SE3 T_newc_base = T_camera_head*T_head_base;
            spaceJustinPose.setTransformModelToCamera(T_newc_base);
            spaceJustin.setPose(spaceJustinPose);
            dart::SE3 T_newc_oldc = T_newc_base*dart::SE3Invert(T_oldc_base);

            memcpy(spaceJustinPose.getReducedArticulation(),currentReportedPose,spaceJustinPose.getReducedArticulatedDimensions()*sizeof(float));
            spaceJustinPose.projectReducedToFull();
            spaceJustin.setPose(spaceJustinPose);

            dart::SE3 currentT_head_r = spaceJustin.getTransformModelToFrame(headFrame)*spaceJustin.getTransformFrameToModel(rightPalmFrame)*T_wh;
            dart::SE3 update_r = dart::SE3Invert(dart::SE3Invert(currentT_head_r)*lastT_head_r); // TODO: simplify

            dart::SE3 currentT_head_l = spaceJustin.getTransformModelToFrame(headFrame)*spaceJustin.getTransformFrameToModel(leftPalmFrame)*T_wh;
            dart::SE3 update_l = dart::SE3Invert(dart::SE3Invert(currentT_head_l)*lastT_head_l); // TODO: simplify

            bool trackReported = true;

            // apply 6DoF deltas
            if (trackReported) {
                rightHandPose.setTransformModelToCamera(rightHand.getTransformModelToCamera()*update_r);
                rightHand.setPose(rightHandPose);

                leftHandPose.setTransformModelToCamera(leftHand.getTransformModelToCamera()*update_l);
                leftHand.setPose(leftHandPose);
            }

            // apply finger joint deltas
            float * rightHandArticulation = tracker.getPose(0).getReducedArticulation();
            for (int i=0; i<tracker.getPose(0).getReducedArticulatedDimensions(); ++i) {
                const int j = 7 + i;
                float diff = currentReportedPose[j] - lastReportedPose[j];
                rightHandArticulation[i] += diff;
            }
            tracker.updatePose(0);
            float * leftHandArticulation = tracker.getPose(2).getReducedArticulation();
            for (int i=0; i<tracker.getPose(2).getReducedArticulatedDimensions(); ++i) {
                const int j = 7 + 15 + 7 + i;
                float diff = currentReportedPose[j] - lastReportedPose[j];
                leftHandArticulation[i] += diff;
            }
            tracker.updatePose(2);

            // apply object 6DoF delta
            if (trackReported) {

                if (trackingMode == ModeObjGrasped) {
                    objectPose.setTransformModelToCamera(rightHand.getTransformModelToCamera()*lastT_ro);
                } else if (trackingMode == ModeObjGraspedLeft) {
                    objectPose.setTransformModelToCamera(leftHand.getTransformModelToCamera()*lastT_lo);
                } else if (trackingMode == ModeObjOnTable){
                    objectPose.setTransformCameraToModel(object.getTransformCameraToModel()*dart::SE3Invert(T_newc_oldc));
                }
                object.setPose(objectPose);

            }
#endif

#ifdef USE_CONTACT_PRIOR
            // update contact vars
            const int * contact = reportedContacts[depthSource->getFrame()];
            for (int i=0; i<10; ++i) {
                bool inContact = (contact[i] > 0);
                anyContact |= inContact;
                *contactVars[i] = inContact;
                contactPriors[i]->setWeight(inContact ? lambdaContact : 0);
            }
#endif

#ifdef ENABLE_JUSTIN
            // update table based on head movement
            {
                float4 tableNorm = make_float4(normalize(make_float3(tableNormX,tableNormY,tableNormZ)),0.f);
                float4 tablePoint = make_float4(make_float3(tableIntercept*tableNorm),1.f); // + make_Float4(0,tableNorm.z,-tableNorm.Y,1.f);
                tableNorm = T_newc_oldc*tableNorm;
                tablePoint = T_newc_oldc*tablePoint;
                tableNormX = tableNorm.x;
                tableNormY = tableNorm.y;
                tableNormZ = tableNorm.z;
                tableIntercept = dot(make_float3(tablePoint),make_float3(tableNorm));
            }
#endif

#ifdef JUSTIN
            static pangolin::Var<float> planeFitNormThresh("opt.planeNormThresh",0.25,-1,1);
            static pangolin::Var<float> planeFitDistThresh("opt.planeDistThresh",0.005,0.0001,0.005);

            if (fitTable) {
                float3 normal = normalize(make_float3(tableNormX,tableNormY,tableNormZ));
                float intercept = tableIntercept;
                dart::fitPlane(normal,
                               intercept,
                               tracker.getPointCloudSource().getDeviceVertMap(),
                               tracker.getPointCloudSource().getDeviceNormMap(),
                               tracker.getPointCloudSource().getDepthWidth(),
                               tracker.getPointCloudSource().getDepthHeight(),
                               planeFitDistThresh,
                               planeFitNormThresh,
                               1,
                               500);

                tableNormX = normal.x;
                tableNormY = normal.y;
                tableNormZ = normal.z;
                tableIntercept = intercept;
            }

            if (subtractTable) {
                tracker.subtractPlane(make_float3(tableNormX,tableNormY,tableNormZ),
                                      tableIntercept,0.005,-1.01);
            }
#endif

#ifdef ENABLE_JUSTIN
            float totalPerPointError = optimizer.getErrPerObsPoint(1,0) + optimizer.getErrPerModPoint(1,0);

            switch (trackingMode) {
            case ModeObjOnTable:
#ifdef USE_CONTACT_PRIOR
                if (anyContact || totalPerPointError > resetInfoThreshold) {
#else
                if (totalPerPointError > resetInfoThreshold) {
#endif
                    trackingMode = ModeIntermediate;
                    tracker.getDampingMatrix(1) = Eigen::MatrixXf::Zero(6,6);
                }
                break;
            case ModeIntermediate:
                if (totalPerPointError < stabilityThreshold) {
#ifdef USE_CONTACT_PRIOR
                    if (anyContact) {
                        bool contactRight = false;
                        for (int i=0; i<5; ++i) { contactRight = contactRight || *contactVars[i]; }
                        trackingMode = (contactRight ? ModeObjGrasped : ModeObjGraspedLeft);
                    } else {
                        trackingMode = ModeObjOnTable;
                    }
#else
                    trackingMode = ModeObjOnTable;
#endif
                }
                break;
            case ModeObjGrasped:
            case ModeObjGraspedLeft:
#ifdef USE_CONTACT_PRIOR
                if (!anyContact || totalPerPointError > resetInfoThreshold) {
#else
                if (true) {
#endif
                    trackingMode = ModeIntermediate;
                    tracker.getDampingMatrix(1) = Eigen::MatrixXf::Zero(6,6);
                }
                break;
            }
#endif

        } else {
#ifdef USE_CONTACT_PRIOR
            for (int i=0; i<10; ++i) {
                bool inContact =  *contactVars[i];
                contactPriors[i]->setWeight(inContact ? lambdaContact : 0);
            }
#endif
        }

    }

    glDeleteBuffersARB(1,&pointCloudVbo);
    glDeleteBuffersARB(1,&pointCloudColorVbo);
    glDeleteBuffersARB(1,&pointCloudNormVbo);

    for (int m=0; m<tracker.getNumModels(); ++m) {
        for (int i=0; i<tracker.getPose(m).getReducedDimensions(); ++i) {
            delete poseVars[m][i];
        }
        delete [] poseVars[m];
    }

    for (uint i=0; i<sizeVars.size(); ++i) {
        delete sizeVars[i];
    }

#ifdef USE_CONTACT_PRIOR
    for (int i=0; i<10; ++i) {
        delete contactVars[i];
    }
#endif

    delete depthSource;

    return 0;
}
