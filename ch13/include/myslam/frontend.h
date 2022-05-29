#pragma once

#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

class Backend;
class Viewer;

enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
 * @brief Frontend
 * Estimate the current frame Pose, add keyframes to the map
 * and trigger optimization when the keyframe condition is met.
 */
class Frontend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    // External interface to add a frame and calculate its positioning result
    bool AddFrame(Frame::Ptr frame);

    // Set functions
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

    FrontendStatus GetStatus() const { return status_; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }

private:
    /**
     * @brief Track in normal mode
     * @return true if success
     */
    bool Track();

    /**
     * @brief Reset when lost
     * @return true if success
     */
    bool Reset();

    /**
     * @brief Track with last frame
     * @return num of tracked points
     */
    int TrackLastFrame();

    /**
     * @brief Estimate current frame's pose
     * @return num of inliners
     */
    int EstimateCurrentPose();

    /**
     * @brief Set current frame as a keyframe and insert it into backend
     * @return true if success
     */
    bool InsertKeyframe();

    /**
     * @brief Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     */
    bool StereoInit();

    /**
     * @brief Detect features in left image in current_Frame_
     * keypoints will be saved in current_frame_
     * @return 
     */
    int DetectFeatures();

    /**
     * @brief Find the corresponding features in right image of current_frame_
     * @return num of features found
     */
    int FindFeaturesInRight();

    /**
     * @brief Build the initial map with single image
     * @return true if success
     */
    bool BuildInitMap();

    /**
     * @brief Triangulate the 2D points in current frame
     * @return num of triangulated points
     */
    int TriangulateNewPoints();

    /**
     * @brief Set the features in keyframe as new observation of the map points
     * @return
     */
    void SetObservationsForKeyFrame();

    // data
    FrontendStatus status_ = FrontendStatus::INITING;

    Frame::Ptr current_frame_ = nullptr;  // current frame
    Frame::Ptr last_frame_ = nullptr;  // previous frame
    Camera::Ptr camera_left_ = nullptr;  // left camera
    Camera::Ptr camera_right_ = nullptr;  // right camera

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    SE3 relative_motion_;  // Relative motion of the current frame to the previous frame, used to estimate the initial value of the current frame pose

    int tracking_inliers_ = 0;  // inliers, used for testing new keyframes

    // params
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 20;
    int num_features_needed_for_keyframe_ = 80;

    // utilities
    cv::Ptr<cv::GFTTDetector> gftt_;  // feature detector in opencv
};

}  // namespace myslam


#endif  // MYSLAM_FRONTEND_H