#pragma once

#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {

/**
 * @brief Map
 * Interaction with the map:
 * front-end calls InsertKeyframe and InsertMapPoint to insert new frames and map points,
 * back-end maintains the structure of the map, determines outlier/reject, etc.
 */
class Map {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {}

    // Add a keyframe
    void InsertKeyFrame(Frame::Ptr frame);
    // Add a map vertex
    void InsertMapPoint(MapPoint::Ptr map_point);

    // Get all map points
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }

    // Get all keyframes
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    // Get active map points
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    // Get active keyframes
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    // Clear the points in map with zero number of observations
    void CleanMap();

private:
    // Turn old keyframes into inactive state
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    LandmarksType landmarks_;  // all landmarks
    LandmarksType active_landmarks_;  // active landmarks
    KeyframesType keyframes_;  // all keyframes
    KeyframesType active_keyframes_;  // active keyframes

    Frame::Ptr current_frame_ = nullptr;

    // settings
    int num_active_keyframes_ = 7;  // Number of active keyframes
};

}  // namespace myslam
#endif // MYSLAM_MAP_H