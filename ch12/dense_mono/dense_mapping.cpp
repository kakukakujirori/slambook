#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// for opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/********************************************************************************************
* This program demonstrates the dense depth estimation of a monocular camera with known
* trajectories using epipolar line search + NCC matching, corresponding to section 12.2 of the book.
* Note that this program is not perfect and you can improve it entirely
* - I am actually exposing some of the problems on purpose (this is an excuse).
**********************************************************************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // edge width
const int width = 640;          // image width
const int height = 480;         // image height
const double fx = 481.2f;       // camera internal parameters
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC half-width of the window
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // convergence determination: minimum variance
const double max_cov = 10;      // diversion determination: maximum variance

// ------------------------------------------------------------------
// Important functions
/// Reads data from the REMODE dataset
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    Mat &ref_depth);

/**
 * @brief Update the depth estimate based on the new image
 * 
 * @param ref           reference image
 * @param curr          current image
 * @param T_C_R         pose from reference image to current image
 * @param depth         depth
 * @param depth_cov2    depth variance
 * @return
 */
void update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2);

/**
 * @brief Epipolar line search
 * 
 * @param ref           reference image
 * @param curr          current image
 * @param T_C_R         pose from reference image to current image
 * @param pt_ref        position of the point in the reference image
 * @param depth_mu      depth mean
 * @param depth_cov     depth variance
 * @param pt_curr       current point
 * @param epipolar_direction  direction of epipolar line
 * @return              whether successfull or not
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction);

/**
 * @brief Update depth filter
 * 
 * @param pt_ref        reference image point
 * @param pt_curr       current image point
 * @param T_C_R         pose
 * @param epipolar_direction epipolar line direction
 * @param depth         depth mean
 * @param depth_cov2    depth variance
 * @return              whether successful or not
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2);

/**
 * @brief Calculate NCC score
 * 
 * @param ref           reference image
 * @param curr          current image
 * @param pt_ref        reference point
 * @param pt_curr       current point
 * @return              NCC score
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// Bilinear grayscale interpolation
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1- yy) * double(d[0])) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1]) / 255.0;
}

// ------------------------------------------------------------------
// Some widgets
// Display estimated depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// Pixel to camera coordinate system
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// Camera coordinate system to pixel
inline Vector2d cam2px(const Vector3d &p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// Detect if a point is inside the image border
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
        && pt(0, 0) + boarder < width && pt(1, 0) + boarder < height;
}

// Show epipolar matching
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_Ref, const Vector2d &px_curr);

// Show epipolar lines
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref,
                      const Vector2d &px_min_curr, const Vector2d &px_max_curr);

// Evaluate depth estimation
void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate);

// ------------------------------------------------------------------


int main(int argc, char ** argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return 1;
    }

    // Read data from a dataset
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (!ret) {
        cout << "Reading image files failed!" << endl;
        return 1;
    }
    cout << "Read total " << color_image_files.size() << " files." << endl;

    // First image
    Mat ref = imread(color_image_files[0], 0);  // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;  // depth init value
    double init_cov2 = 3.0;  // variance init value
    Mat depth(height, width, CV_64F, init_depth);  // depth image
    Mat depth_cov2(height, width, CV_64F, init_cov2);  // depth variance image

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;  // Coordinate conversion: T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." <<  endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    Mat &ref_depth) {

    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: filename, tx, ty, tz, qx, qy, qz, qw , note that it is TWC and not TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d: data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }

    return true;
}

// Updates the entire depth map
void update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; x++) {
        for (int y = boarder; y < height - boarder; y++) {
            // Iterate through each pixel
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) continue;  // depth has converged or diverged
            // Search for (x,y) matches on the epipolar line
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(ref, curr, T_C_R, Vector2d(x, y), depth.ptr<double>(y)[x], sqrt(depth_cov2.ptr<double>(y)[x]), pt_curr, epipolar_direction);
            if (!ret) continue;  // Failed to match
            
            // Cancel this comment to show matches
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // Match successfully, update depth map
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }
}

// Epipolar line search
// See book 12.2, section 12.3 for the method
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction) {

    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;  // P-vector of the reference frame

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref);  // Pixels by depth-averaged projection
    double d_min = depth_mu - 3 * depth_cov;
    double d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));  // Pixels projected by minimum depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));  // Pixels projected by maximum depth

    Vector2d epipolar_line = px_max_curr - px_min_curr;  // Epipolar lines (in the form of line segments)
    epipolar_direction = epipolar_line;  // Direction of epipolar line
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();  // Half length of the epipolar line segment
    if (half_length > 100) half_length = 100;  // We don't want to search for too many things

    // Uncomment this sentence to show epipolar lines (line segments)
    // showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);

    // Search on the epipolar line, centered on the depth-average point, taking half-lengths on each side
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l < half_length; l += 0.7) {  // l += sqrt(2)/2
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // Points to be matched
        if (!inside(px_curr)) continue;
        // Calculate the NCC of the point to be matched and the reference frame
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f) return false;  // Only trust the NCC very high match
    pt_curr = best_px_curr;
    return true;
}

double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // Zero-mean-normalized intercorrelation
    // Calculate the mean value first
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr;  // Mean value of the reference frame and the current frame
    for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
        for (int x = -ncc_window_size; x <= ncc_window_size; x++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // Compute zero mean NCC
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10);  // Preventing zero in the denominator
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {

    // Calculate the depth with triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // The equations
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // are transformed into the following system of matrix equations
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;  // result from ref side
    Vector3d xn = t + ans[1] * f2;  // result from curr side
    Vector3d p_esti = (xm + xn) / 2.0;  // The position of P, taking the average of both
    double depth_estimation = p_esti.norm();  // depth value

    // Calculation uncertainty (with one pixel as error)
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // Gaussian Fusion
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0 ,0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// These latter are too simple so I will not comment (in fact, because lazy)
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;     // mean error
    double ave_depth_error_sq = 0;      // squared error
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref,
                      const Vector2d &px_min_curr, const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
