/* ----------------------------------------------------------------------------
 * Author(s): Sudeep Pillai (spillai@csail.mit.edu)
 * License: MIT
 * First commit: 03 Dec, 2014
 * Latest commit: 18 Jun, 2015
 
 * This code is mostly based on:
 * 
 *    Fisher and VLAD with FLAIR
 *    Koen van de Sande, Cees G. M. Snoek, Arnold W. M. Smeulders
 *    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014

 * Please refer to the implementation in the paper below (and the one above)
 * if you use this code in your publication:
 *
 *    Monocular SLAM Supported Object Recognition
 *    S. Pillai, J. Leonard
 *    Robotics: Science and Systems (RSS) 2015

 * -------------------------------------------------------------------------- */

#include <opencv2/opencv.hpp>

#define DEBUG 0
#define PROFILE 0
#define VIZ 0

namespace bot { namespace vision { 

class FLAIR_code {
  /* ----------------------------------------------------------------------------
   * This code is mostly based on:
   *    Fisher and VLAD with FLAIR
   *    Koen van de Sande, Cees G. M. Snoek, Arnold W. M. Smeulders
   *    koen.me/research/pub/vandesande-cvpr2014.pdf
   * -------------------------------------------------------------------------- */

 private:
  enum encoding_type { BOW=0, VLAD=1, FISHER=2 };

  // BOW Encoding - for each code (k):
  // store code bit if exists at point (x,y)
  std::vector<cv::Mat1f> _codes; 
  
  // store look up location at point (x,y)
  std::vector<cv::Mat1d> _integrals;
  
  // offsets for codes, integrals
  std::vector<int> _DW; 
  
  // Constants
  bool _initialized; 
  cv::Size _im_sz, _grid_sz;
  
  // _Denc: # of dimensions for output code
  int _step, _W, _H, _D, _K, _Denc;

  // BOW, VLAD or FISHER vector encoding options
  encoding_type _encoding;
  
  // Spatial pyramid/pooling
  int _nlevelbins;
  std::vector<int> _levels, _levelbins_off;
  
  // Feature Matcher
  cv::Ptr<DescriptorMatcher> _matcher;

  // Train BoW
  int _training_idx;
  cv::Mat1f _codebook;
  cv::Ptr<BOWKMeansTrainer> _bow_trainer;
  
#if VIZ
  // For visualization
  cv::Mat3b _vis, _vis2, _vis3; 
  std::vector<cv::Scalar> _colorTab;
#endif

  // =============================
  // Private methods
  // =============================
 private:

  // Initialize with image grid size
  void _initialize(const int& D) {
    // One-time init.
    _initialized = true;
    _D = D;
    
    // Encoding size
    if (_encoding == BOW)
      _Denc = 1;
    else if (_encoding == VLAD)
      _Denc = _D;
    else if (_encoding == FISHER)
      _Denc = 2 *_D; 

    // One time initialize offsets
    _DW = std::vector<int>(_Denc);
    for (int d=0; d<_Denc; d++)
      _DW[d] = _grid_sz.width * d;

    // Initialize codes, and integrals
    _codes.clear();
    _integrals.clear();
    for (int k=0; k<_K; k++) { 
      _codes.push_back(cv::Mat1f::zeros(_grid_sz.height, _grid_sz.width * _Denc));
      _integrals.push_back(cv::Mat1d::zeros(_grid_sz.height, _grid_sz.width * _Denc));
    }
  }

  void _reset() {

    // Zero-out codes for next iteration
    for (int k=0; k<_K; k++) { 
      _codes[k] = cv::Mat1f::zeros(_grid_sz.height, _grid_sz.width * _Denc);
      _integrals[k] = cv::Mat1d::zeros(_grid_sz.height, _grid_sz.width * _Denc);
    }
    
  }


  // =============================
  // Public methods
  // =============================
 public:
 
  /**
   * FLAIR encoding
   * @param W,H,K,step: Image (W)idth, Image (H)eight, Codebook size(K), Dense Sampling (step) size
   * @param levels: Spatial pyramid levels (1x1,2x2,4x4)=>(1,2,4)
   * @param encoding: BOW, or VLAD encoding
   */

  FLAIR_code(const int& W, const int& H, const int& K, const int& step,
             const cv::Mat_<int>& levels, const int& encoding)
      : _step(step), _W(W), _H(H), _K(K), _encoding(encoding), _initialized(false)
  {

    // std::cerr << "levels: " << levels << " " << levels.rows << " " << levels.cols << std::endl;
    assert(levels.rows && levels.cols);

    // Enable spatial pooling
    if (levels.rows) { 

      // Initialize levels and bin offsets
      // levels: [1,2,4]
      _levels = std::vector<int>(levels.rows);
      _levelbins_off = std::vector<int>(levels.rows+1, 0);    

      // levels: [1,2,4] => levelbins_off: [0,1,5,21]
      for (int j=0; j<levels.rows; j++) { 
        _levels[j] = levels(j,0);
        _levelbins_off[j+1] = _levelbins_off[j] + _levels[j] * _levels[j];
      }
      _nlevelbins = _levelbins_off.end()[-1];
#if DEBUG
      std::cerr << "nlevelbins: " << cv::Mat(_levelbins_off) << std::endl;
#endif
      
    }
    // No spatial pooling
    else
    {
      _nlevelbins = 1;
    }
    
    // Setup image and grid size
    _im_sz = cv::Size(_W, _H);
    _grid_sz = cv::Size(int(ceil((_im_sz.width * 1.0) / _step)),
                        int(ceil((_im_sz.height * 1.0) / _step))); 
    // std::cerr << "grid_sz: " << _grid_sz << std::endl;
#if DEBUG
    std::cerr << "_nlevel_bins:" << _nlevelbins << std::endl;
    std::cerr << "im_sz: " << _im_sz << std::endl;
    std::cerr << "grid_sz: " << _grid_sz << std::endl;
#endif
    // Initialize vocab matcher
    _matcher = DescriptorMatcher::create("FlannBased");
    
#if VIZ
    for( int i = 0; i < _K; i++ ) {
      int b = cv::theRNG().uniform(0, 255);
      int g = cv::theRNG().uniform(0, 255);
      int r = cv::theRNG().uniform(0, 255);
      _colorTab.push_back(cv::Scalar((uchar)r, (uchar)g, (uchar)b));
    }
#endif
    
  }
  
  ~FLAIR_code() {
  }

 
 public:

  /**
   * Set the FLAIR encoding codebook/vocabulary
   * @param codebook: [K x D] codebook computed from training data
   */

  void setVocabulary(const cv::Mat& codebook) {
    if (codebook.type() != CV_32F)
      throw std::runtime_error("Codebook type not CV_32F");
    if (codebook.rows != _K)
      throw std::runtime_error("Codebook length != K");
    
#if DEBUG
    std::cerr << "codebook: " << codebook.size() << std::endl;
#endif
    // Copy codebook for internal reference
    _codebook = codebook.clone();

    // Initialize dimensionality of descriptor space
    _initialize(_codebook.cols);
    assert(codebook.cols == _D);
    
    // Setup codebook lookup
    _matcher->clear();
    _matcher->add( std::vector<cv::Mat>(1, _codebook) );
  }

  /**
   * Constructs the integral histogram table from the sampled
   * points, and descriptions. This is a pre-processing step before
   * object proposal descriptions)
   * @param descriptors: [N x D] description from Dense-SIFT
   * @param pts: [N x 2] (x,y) densely sampled point locations 
   */

  void process_descriptions(const cv::Mat& pts, const cv::Mat& descriptors) {
    assert(pts.type() == CV_32S);
    assert(descriptors.type() == CV_32F);

    // Reinitialize
    _reset();
    
    assert(_initialized);
    assert(!_codebook.empty());
    assert(pts.rows == descriptors.rows);

    
    // ===============================================================
    // 1a. Match keypoint descriptors to cluster center (to vocabulary)
    std::vector<cv::DMatch> matches;
    _matcher->match( descriptors, matches );
    assert(matches.size() == descriptors.rows);
    
    // 1b. Nearest center lookup
    int queryIdx, trainIdx;
    std::vector<int> codes(matches.size());
    for( size_t i = 0; i < matches.size(); i++ ) {
      queryIdx = matches[i].queryIdx; // query index
      trainIdx = matches[i].trainIdx; // cluster index
      // std::cerr << "code: " << trainIdx << std::endl;

      CV_Assert( queryIdx == (int) i );
      codes[queryIdx] = trainIdx;
    }
    
    // ===============================================================
    // 2. Fill sparse grid [K x [H x WD] ] with codes
    int gx, gy, code;

    float* resf = 0; 
    cv::Mat1f residual; // residual_v only for FISHER vectors
        
    for (int j=0; j<pts.rows; j++) {
      code = codes[j];
      // assert(code <= 256);
      assert(code <= _codes.size());
      
      gx = pts.at<int>(j,0) / _step, gy = pts.at<int>(j,1) / _step;
      assert( gx >= 0 && gy >= 0 && gx < _grid_sz.width && gy < _grid_sz.height);

      // Compute residual for each datapoint, and dimension
      if (_encoding == BOW)
      {
        // Construct bit array for each code k=codes(j,0)
        // Add up multi-scale features into a single coded array
        _codes[code](gy, gx) += 1;
      }

      // TODO: accumulate residuals across multiple image scales
      else if (_encoding == VLAD)
      { 
        residual = descriptors.row(j) - _codebook.row(code);
        resf = (float*) residual.data;
        for (int d=0; d<_D; d++) {
          _codes[code](gy, _DW[d] + gx) += resf[d];
        }
      }
     
      else if (_encoding == FISHER)
      {
        throw std::runtime_error("FISHER encoding not implemented yet!");
      }
    }

#if VIZ
    // ===============================================================
    // Visualize
    _vis = cv::Mat3b::zeros(_H, _W);
    _vis2 = cv::Mat3b::zeros(_H, _W);
    if (_encoding == BOW) { 
      // BoW only: Visualize each code word independently
      for (int k=0; k<_K; k++) {
        for (int dy=0; dy<_grid_sz.height; dy++)
          for (int dx=0; dx<_grid_sz.width; dx++)
            if (_codes[k](dy,dx) > 0)
              cv::circle(_vis, cv::Point(dx * _step, dy * _step), 3, _colorTab[k], -1, CV_AA);
      }
    }
    
    // for (int j=0; j<rects.size(); j++)
    //   cv::rectangle( _vis, rects[j], cv::Scalar::all(128), 1, 8, 0 );
#endif
    
#if PROFILE
    double st = cv::getTickCount();
#endif
    
    // ===============================================================
    // 4. Compute the integral image
    for (int k=0; k<_integrals.size(); k++) {
      cv::integral(_codes[k], _integrals[k]);
      // std::cerr << k << " : " << std::endl << _integrals[k] << std::endl;
    }

#if PROFILE
    std::cerr << "Time taken for integral images: "
              << (cv::getTickCount() - st) / cv::getTickFrequency() * 1e3 << " ms" << std::endl;
#endif

    
  }

  /**
   * Compute histograms for each object proposal given the integral histogram table
   * @param rects: [B x 4] boxes (xmin, ymin, xmax, ymax) for each bounding box
   * @return [B x _K * _Denc * _nlevelbins] Histogram for each bounding box
   */

  cv::Mat1f process_boxes(const cv::Mat& in_rects) {
    assert(rects.type() == CV_32F);    

    // ===============================================================
    // Copy over from matrix
    std::vector<cv::Rect> rects(in_rects.rows);
    for (int j=0; j<rects.size(); j++) {

      // Copy rect coords
      rects[j].x = in_rects.at<float>(j,0), rects[j].y = in_rects.at<float>(j,1),
          
          rects[j].width = std::min(_W-1-in_rects.at<float>(j,0), in_rects.at<float>(j,2) - in_rects.at<float>(j,0)),
          rects[j].height = std::min(_H-1-in_rects.at<float>(j,1), in_rects.at<float>(j,3) - in_rects.at<float>(j,1));
      
      assert(rects[j].x >=0 && rects[j].x + rects[j].width < _W &&
             rects[j].y >=0 && rects[j].y + rects[j].height < _H);
    }

    // ===============================================================
    // Modify rectangles to fit step size, and scale down to grid size
    // For all boxes: extremes are inclusive
    // left/top edges are rounded up to closest step size (ceil)
    // right/bottom edges are rounded down to closest step size (floor)
    int bx, by;
    for (int j=0; j<rects.size(); j++) {
      bx = int(round(rects[j].x * 1.f / _step)), by = int(round(rects[j].y * 1.f / _step));

      // New width: (rounded up right edge - rounded down left edge)
      // Similarly for height
      rects[j].width = int(round((rects[j].x + rects[j].width) * 1.f / _step)) - bx; 
      rects[j].height = int(round((rects[j].y + rects[j].height) * 1.f / _step)) - by; 
      
      rects[j].x = bx, rects[j].y = by;
    }

#if VIZ
    _vis = cv::Mat3b::zeros(_H, _W);
    _vis2 = cv::Mat3b::zeros(_H, _W);
    // Draw rectangles shifted to grid
    for (int j=0; j<rects.size(); j++) {
      cv::Rect r = rects[j];
      r.x *= _step, r.y *= _step, r.width *= _step, r.height *= _step;
    }
    cv::imshow("rects", _vis);
#endif

#if PROFILE        
    double st = cv::getTickCount();
#endif
    
    // ===============================================================
    // 5. Compute the BoW histograms for each box [B x K]
    // std::vector<double> hvec(_K * _D);
    cv::Mat1f hvec(1, _K * _Denc * _nlevelbins);
    cv::Mat1f hist(rects.size(), _K * _Denc * _nlevelbins);
    
    int bin_idx;

    int x1, x2, y1, y2;
    float px_y1, px_y2, px_x1, px_x2;

    for (int j=0; j<rects.size(); j++) {
      const cv::Rect& r = rects[j];

      // Reset values for each rect
      hvec = 0.f; bin_idx = 0;
      float* hv = (float*) hvec.data; 
            
      // 5a. Vectorize descriptor for each rect [1 x (KD)]
      // Compute the histogram

      // For each codeword k:
      for (int k=0; k<_K; k++) {
          const cv::Mat1d& int_k = _integrals[k];
          
        // For each dimension d: 
        for (int d=0; d<_Denc; d++) {
          
          // For each spatial level/split l:
          int bidx = 0; 
          for (int l=0; l<_levels.size(); l++) {

            // for each bin b within a level l:
            for (int by=0; by<_levels[l]; by++) {
              px_y1 = by * 1.f / _levels[l];
              px_y2 = (by+1) * 1.f / _levels[l];
              for (int bx=0; bx<_levels[l]; bx++, bidx++) {
              px_x1 = bx * 1.f / _levels[l];
              px_x2 = (bx+1) * 1.f / _levels[l];
               x1 = (1-px_x1) * r.x + (px_x1) * (r.x + r.width),
                  x2 = std::min(_grid_sz.width-1, int((1-px_x2) * r.x + (px_x2) * (r.x + r.width)) + 1);
               y1 = int((1-px_y1) * r.y + (px_y1) * (r.y + r.height)),
                  y2 = std::min(_grid_sz.height-1, int((1-px_y2) * r.y + (px_y2) * (r.y + r.height)) + 1);

                assert(x2 < _grid_sz.width && y2 < _grid_sz.height);

                // Inclusive left and top, exclusive right, and bottom
                // _K*_Denc*bin_idx + k*_Denc + d, bin_idx++
                assert(_DW[d] + x2 <= int_k.cols && _DW[d] + x1 <= int_k.cols);
                hv[_K*_Denc*bidx + k*_Denc + d] =
                            int_k(y2,  _DW[d] + x2) - int_k(y2,  _DW[d] + x1) -
                            int_k(y1,  _DW[d] + x2) + int_k(y1,  _DW[d] + x1); 
                assert(_K*_Denc*bidx + k*_Denc + d <= hvec.cols);

#if VIZ
                // vis2 = cv::Mat3b::zeros(vis2.size());
                cv::rectangle( vis2, cv::Rect((x1) * _step, (y1) * _step,
                                              (x2-x1) * _step, (y2-y1) * _step),
                               cv::Scalar::all(200), 1, 8, 0 );
                cv::imshow("rects-scaled", vis2);
                cv::waitKey(0);
#endif
#if DEBUG
std::cerr << "k: " << k << " d: " << d
                          << " l: " << _levels[l] << " b:" << bin_idx-1
                          << " bx:" << bx << " by:" << by
                          << " x: " << x1 << "->" << x2 << " "
                          << " y: " << y1 << "->" << y2 << " "
                          << " idx: " << bin_idx-1 << " "
                          << hv[bin_idx-1] << std::endl;
#endif
                // end for each bin bx
              }

              // end for each bin by
            }

            // end for each level l
          }

          // end for each dim d
        } 
        
        // end for each code k
      }

      // 5b. Normalization: 
      // Signed-Square Rooting / Power Normalization with alpha = 0.5
      if (_encoding == VLAD || _encoding == FISHER) {
        for (int kd=0; kd<hvec.cols; kd++) {
          if (hv[kd] >= 0.f)
            hv[kd] = std::sqrt(hv[kd]);
          else
            hv[kd] = -std::sqrt(-hv[kd]);
        }
      }
            
      // Followed by L2 normalization (per level bin)
      double inv_norm; 
      for (int bin=0; bin<_nlevelbins; bin++) {

        double ssq = 0;
        for (int kd=bin *_K * _Denc; kd < (bin+1) *_K * _Denc; kd++)
          ssq += (hv[kd] * hv[kd]);
        inv_norm = 1.f / (1e-12 + std::sqrt(ssq));
        
        for (int kd=bin *_K * _Denc; kd < (bin+1) *_K * _Denc; kd++)
          hv[kd] *= inv_norm;
      }
      
      // Copy hvec
      hvec.copyTo(hist.row(j));

      // end for each rect
    }


#if PROFILE
    std::cerr << "Time taken for histogram: "
              << (cv::getTickCount() - st) / cv::getTickFrequency() * 1e3 << " ms" << std::endl;
#endif
    
#if VIZ
    // ===============================================================
    // Visualize the BoW and boxes
    // Draw dense grid
    for (int j=0; j<pts.rows; j++) {
      cv::circle(_vis2, cv::Point(pts.at<int>(j,0), pts.at<int>(j,1)), 3, _colorTab[codes[j]], 1, CV_AA);
    }
    
    // Draw rectangles shifted to grid
    for (int j=0; j<rects.size(); j++) {
      cv::Rect r = rects[j];
      cv::Rect r_ = r;
      r_.x *= _step, r_.y *= _step, r_.width *= _step, r_.height *= _step;
      r_.x -= _step/2, r_.y -= _step/2, r_.width += _step, r_.height += _step; // for viz purposes
      cv::rectangle( _vis2, r_, cv::Scalar::all(200), 2, 8, 0 );
      // ===============================================================
      // BoW only: Visualize the BoW histograms
      if (_encoding == BOW) {
        
        cv::Mat1f hvis = hist.row(j).clone();
        cv::Mat3b _vis3 = cv::Mat3b::zeros(_H, _W);
        cv::rectangle( _vis3, r_, cv::Scalar::all(200), 1, 8, 0 );
        for (int dy = r.y; dy <= r.y + r.height; dy++) { 
          for (int dx = r.x; dx <= r.x + r.width; dx++) {
            for (int k=0; k<hvis.cols; k++) { // not visualizing all levels
              if (hvis(0,k) > 0) {
                hvis(0,k) = hvis(0,k) - 1;
                cv::circle(_vis3, cv::Point(dx * _step, dy * _step), 3, _colorTab[k % _K], -1, CV_AA);
                break;
              }

              // std::cerr << dy << " " << dx << " " << hvis << std::endl;
            }
          }
        }
        cv::imshow("rects-bow", _vis3);
        cv::waitKey(0);
      }
    }
    cv::imshow("rects-scaled", _vis2);
    cv::waitKey(0);    
#endif

    return hist;
  }

  /**
 * Main method for FLAIR encoding given densely-sampled descriptions, and
 * bounding boxes via object proposal techniques
 
 * @param descriptors: [N x D] description from Dense-SIFT
 * @param pts: [N x 2] (x,y) densely sampled point locations 
 * @param rects: [B x 4] boxes (xmin, ymin, xmax, ymax) for each bounding box
 * @return [B x _K * _Denc * _nlevelbins] Histogram for each bounding box
 */
  cv::Mat1f process(const cv::Mat& pts,
                    const cv::Mat& descriptors, const cv::Mat& in_rects) {
    process_descriptions(pts, descriptors);
    return process_boxes(in_rects);
  }
};

/**
 * Simple function call for one-time FLAIR encoding
 * @param descriptors: [N x D] description from Dense-SIFT
 * @param pts: [N x 2] (x,y) densely sampled point locations 
 * @param rects: [B x 4] boxes (xmin, ymin, xmax, ymax) for each bounding box
 * @param codebook: [K x D] codebook computed from training data
 * @param W,H,K,step: Image (W)idth, Image (H)eight, Codebook size(K), Dense Sampling (step) size
 * @param levels: Spatial pyramid levels (1x1,2x2,4x4)=>(1,2,4)
 * @param encoding: BOW, or VLAD encoding
 * @return [B x _K * _Denc * _nlevelbins] Histogram for each bounding box
 */
cv::Mat1f flair_code(const cv::Mat& descriptors, const cv::Mat& pts, const cv::Mat& rects, const cv::Mat& codebook,
                     const int& W, const int& H, const int& K, const int& step,
                     const cv::Mat_<int>& levels, const int& encoding) {
  FLAIR_code fc(W, H, K, step, levels, encoding);
  fc.setVocabulary(codebook);
  return fc.process(pts, descriptors, rects);
}


} // namespace vision
} // namespace bot
