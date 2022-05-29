#pragma once

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

class Map;

/**
 * @brief Backend
 * There is a separate optimization thread that starts optimization when Map updates.
 * Map updates are triggered by the front-end
 */
class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    // Start the optimization thread in the constructor and hang it
    Backend();

    // Set up left and right cameras for obtaining internal and external parameters
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    // Set map
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    // Triggers map updates and initiates optimization
    void UpdateMap();

    // Close back-end threads
    void Stop();

private:
    // Backend threads
    void BackendLoop();

    // Optimization for a given keyframe and waypoint
    void Optimize(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H