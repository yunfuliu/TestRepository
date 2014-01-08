#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

#ifndef _yfCVDIP
#define _yfCVDIP
/// [OpenCV] Image Processing related add-ons
namespace cvdip{

	//////////////////////////////////////////////////////////////////////////
	/// Filtering related
	namespace filtering{

		/**
		* @brief		filtering with median filter
		* @brief		paper: no
		*
		* @author		Yunfu Liu (yunfuliu@gmail.com)
		* @date			Sept. 6, 2013
		* @version		1.0
		*
		* @param		src: input image (grayscale only)
		* @param		dst: output image
		* @param		blocksize: the used block size
		*
		* @return		bool: true: successful, false: failure
		*/
		bool medianfilter(const cv::Mat &src,cv::Mat &dst,cv::Size blocksize);

	}


	//////////////////////////////////////////////////////////////////////////
	/// Edge detection related
	namespace edgedetection{

		/**
		* @brief		Sobel edge detection
		* @brief		paper: digital image processing textbook
		*
		* @author		Yunfu Liu (yunfuliu@gmail.com)
		* @date			Sept. 4, 2013
		* @version		1.0
		*
		* @param		src: input image (grayscale only)
		* @param		dst: output image
		*
		* @return		bool: true: successful, false: failure
		*/
		bool Sobel(const cv::Mat &src, cv::Mat &dst);
	}

	//////////////////////////////////////////////////////////////////////////
	/// Halftoning related
	namespace halftoning{

		/// Dot diffusion related
		namespace dotdiffusion{

			/**
			* @brief		filtering with median filter
			* @brief		paper: J. M. Guo and Y. F. Liu"Improved dot diffusion by diffused matrix and class matrix co-optimization," IEEE Trans. Image Processing, vol. 18, no. 8, pp. 1804-1816, 2009.
			*
			* @author		Yunfu Liu (yunfuliu@gmail.com)
			* @date			May 17, 2013
			* @version		1.0
			* 
			* @param		src: input image (grayscale only)
			* @param		dst: output image
			* @param		ClassMatrixSize: 只能允許8x8 and 16x16
			*
			* @return		bool: true: successful, false: failure
			*/ 
			bool GuoLiu2009(const cv::Mat &src, cv::Mat &dst,const int ClassMatrixSize);


		}

	}


	//////////////////////////////////////////////////////////////////////////
	/// Image enhancement related
	namespace enhancement{
		
		/// Local methods
		namespace local{

			/**
			* @brief		local contrast enhancement
			* @brief		paper: B. Liu, W. Jin, Y. Chen, C. Liu, and L. Li, "Contrast enhancement using non-overlapped sub-blocks and local histogram projection," TCE, vol. 57, no. 2, 2011.
			* @brief		nickname: non-overlapped sub-blocks and local histogram projection based contrast enhancement (NOSHP)
			*
			* @author		Yunfu Liu
			* @date			Sept. 3, 2013
			* @version		1.0
			*
			* @param		N: number of blocks
			*
			* @return		bool: true: successful, false: failure
			*/
			bool LiuJinChenLiuLi2011(const cv::Mat &src,cv::Mat &dst,const cv::Size N);

			/**
			* @brief		local contrast enhancement		
			* @brief		paper: L. Jiao, Z. Sun, and A. Sha, "Improvement of image contrast with local adaptation,," Intl. Conf. Multimedia and Information Technology, 2010.
			* @brief		nickname: Partially Overlapped Sub-block Logarithmic Trandformation (POSLT)
			*
			* @author		Yunfu Liu
			* @date			Sept. 3, 2013 - present
			* @bug			目前仍無法正確使用, 待完善2013/11/28
			* 
			* @param		BlockSize: block size (blocksize >= step size)
			* @param		StepSize: step size (the smaller step size, the better quality; when step size = block size: nonoverlapped strategy; when step size< block: overlapped strategy)
			* @param		mode: 1: use EME as cost function; 2: use TEN as cost function
			*
			* @return		bool: true: successful, false: failure
			*/
			bool JiaoSunSha2010(const cv::Mat &src,cv::Mat &dst,const cv::Size BlockSize,const cv::Size StepSize,const short mode);

			/**
			* @brief		local contrast enhancement, KimKimHwang2001's POSHE的改良版本
			* @brief		paper: F. Lamberti, B. Montrucchio, and A. Sanna, "CMBFHE: a novel contrast enhancement technique based on cascaded multistep binomial filtering histogram equalization," TCE, vol. 52, no. 3, 2006.
			* @brief		nickname: cascaded multistep binomial filtering histogram equalization (CMBFHE)
			*
			* @author		賴柏勳, Yunfu Liu
			* @date			May 15, 2013
			*
			* @param: B: number of blocks
			* @param: S: number of regions divided by the step size. S should >2xB. .This should be the "Bx2" or "Bx4" or "Bx8" etc multiple of 2. "S"需為B的2的次方倍數(2,4,8,16,32,64,128...). S越大速度慢品質高.
			*
			* @return: bool: true: successful, false: failure
			*/
			bool LambertiMontrucchioSanna2006(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S);
			
			
			/**
			* @brief		local contrast enhancement
			* @brief		paper: Z. Yu and C. Bajaj, "A fast and adaptive method for image contrast enhancement," ICIP, vol. 2, pp. 1001-1004, 2004.
			*
			* @author		Yunfu Liu
			* @date			May 13, 2013
			*
			* @param		C: within [0,1] - this is only for isotropic mode
			* @param		anisotropicMode: false-isotropic mode; true-anisotropic mode (iso needs C; ani needs R, respectively)
			* @param		R: within [0.01,0.1] - this is only for anisotropic mode
			*
			* @return		bool: true: successful, false: failure
			*/ 
			bool YuBajaj2004(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float C=0.85f,bool anisotropicMode=false,const float R=0.09f);

			/**
			* @brief		local contrast enhancement
			* @brief		paper: J. Y. Kim, L. S. Kim, and S. H. Hwang, "An advanced contrast enhancement using partially overlapped sub-block histogram equalization," TCSVT, vol. 11, no. 4, pp. 475-484, 2001. 
			* @brief		nickname: partially overlapped sub-block histogram equalization (POSHE)
			* 
			* @author		Yunfu Liu
			* @date			May 15, 2013
			*
			* @param		B: number of blocks
			* @param		S: number of regions divided by the step size. S should >2xB. .This should be the "Bx2" or "Bx4" or "Bx8" etc multiple of 2. "S"需為B的2的次方倍數(2,4,8,16,32,64,128...). S越大速度慢品質高.
			*
			* @return		bool: true: successful, false: failure
			*/
			bool KimKimHwang2001(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S);

			/**
			* @brief		local contrast enhancement
			* @brief		paper: J. A. Stark, "Adaptive image contrast enhancement using generalizations of histogram equalization," TIP, vol. 9, no. 5, pp. 889-896, 2000.
			* 
			* @author		賴柏勳, Yunfu Liu 
			* @date			May 14, 2013
			* 
			* @param		alpha: 0~1. 0: histogram equalization; 1: local-mean subtraction effect.
			* @param		beta: 0~1
			* 
			* @return		bool: true: successful, false: failure
			*/
			bool Stark2000(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float alpha=0.5f,const float beta=0.5f);


			/**
			* @brief		local contrast enhancement
			* @brief		paper: R. C. Gonzalez and R. E. Woods, Digital Image Processing, 2nd ed., Reading, MA: Addison-Wesley, 1992.
			* @brief		nickname: local histogram equalization (LHE)
			*
			* @author		Yunfu Liu 
			* @date			May 16, 2013
			* 
			* @param		blocksize: block sizes
			* 
			* @return		bool: true: successful, false: failure
			*/
			bool LocalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst,const cv::Size blocksize);
		}

		/// Global methods
		namespace global{

			/**
			* @brief		global contrast enhancement
			* @brief		paper: M. Abdullah-Al-Wadud, Md. Hasanul Kabir, M. Ali Akber Dewan, and O. Chae, "A dynamic histogram equalization for image contrast enhancement," Intl. Conf. Consumer Electronics, pp. 1-2, 2007.
			* @brief		nickname: dynamic histogram equalization (DHE)
			* 
			* @author		劉少雲
			* @date			May 15, 2013
			* @bug			目前看似還有bug, 待修正.
			* 
			* @param		x: ??????????????????????????????????
			* 
			* @return		bool: true: successful, false: failure
			*/
			bool WadudKabirDewanChae2007(const cv::Mat &src, cv::Mat &dst, const int x);

			/**
			* @brief		global contrast enhancement
			* @brief		paper: R. C. Gonzalez and R. E. Woods, Digital Image Processing, 2nd ed., Reading, MA: Addison-Wesley, 1992.
			* @brief		nickname: global histogram equalization (GHE)
			* 
			* @author		Yunfu Liu
			* @date			May 16, 2013
			* 
			* @return		bool: true: successful, false: failure
			*/
			bool GlobalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst);

		}

	}

	//////////////////////////////////////////////////////////////////////////
	/// IQA related 
	namespace qualityassessment{

		/**
		* @brief		usually used to estimate contrast of an image, and it can evaluate naturalness and uniform lighting 
		* @brief		paper: original: S. S. Agaian, K. Panetta, and A. M. Grigoryan, "A new measure of image enhancement," in Proc. Intl. Conf. Signal Processing Communication, 2000.
		* @brief		another representation: (this is the one used in this implementation) S. S. Agaian, B. Silver, and K. A. Panetta, "Transform coefficient histogram-based image enhancement algorithms using contrast entropy," TIP, 2007. 
		* @brief		nickname: measure of enhancement (EME) or measure of improvement
		* 
		* @author		Yunfu Liu
		* @date			Sept. 4, 2013
		*
		* @param		nBlocks: number of blocks at either x or y axis; this size should be odd since this is just like a filter as defined in the paper
		* @param		mode: 1: standard mode: use the local max and min to evaluate the eme; 2: ab mode: use BTC's a and b to represent a block's contrast
		*
		* @return		float: return the value of EME
		*/
		float EME(const cv::Mat &src,const cv::Size nBlocks,const short mode=1);

		/**
		* @brief		it is able to describe whether some artificial texture appear or not
		* @brief		paper: L. Jiao, Z. Sun, and A. Sha, "Improvement of image contrast with local adaptation," in Proc. Intl. Conf. Multimedia and Informatin Technology, 2010.
		* @brief		(this paper used this TEN in their paper, and the original one is published in 1970 as in their reference list)
		* @brief		nickname: TEN
		*
		* @author		Yunfu Liu 
		* @date			Sept. 4, 2013
		*
		* @return		float: return the value of TEN
		*/
		float TEN(const cv::Mat &src);

		/**
		* @brief		used to estimate the difference between two images (!!!the lower the better)
		* @brief		paper: N. Phanthuna, F. Cheevasuvit, and S. Chitwong, "Contrast enhancement for minimum mean brightness error from histogram partitioning," ASPRS Conf. 2009.
		* @brief		nickname: absolute mean brightness error (AMBE)
		*
		* @author		Yunfu Liu
		* @date			Sept. 4, 2013
		*
		* @return		float: return the value of AMBE
		*/
		float AMBE(const cv::Mat &src1,const cv::Mat &src2);

	}

}
#endif

#ifndef _yfCVML
#define _yfCVML
/// [OpenCV] Machine Learning related add-ons
namespace cvml{
	
	/**
	* @brief		給CvMLData使用之功能, 將class label轉換為原本的類別名稱
	* 
	* @author		Yunfu Liu 
	* @date			Sept. 4, 2013
	* 
	* @param		tmap: 為CvMLData.get_class_labels_map()的返回結果
	* @param		classLabel: 為類別標籤
	* 
	* @return		string: 返回該classLabel所對應之類別名稱
	*/
	std::string	classindextostring(std::map<std::string,int> &tmap,const int classLabel);


	/// Used to evaluate ROC or Pre-Recall curves
	namespace ROC_eval{

		/**
		* @brief		分類. 輸入confidence, 輸出兩分類評估結果eval_output, 配合threshold
		*
		* @param		confidence, 輸入. 需為CV_32F格式
		* @param		eval_output, 輸出. 需先建立, 且為CV_8U格式. 輸出為binary, 0: negative, 1: positive 
		* @param		thres: 門檻值, 越高則判定positive越嚴格.
		*/ 
		bool	classify(const cv::Mat &confidence,cv::Mat &output,const double thres);

		/**
		* @brief		針對單一reference進行, 評估取得tp, fp, tn, fn
		*
		* @param		output, 輸入. 需為CV_8U格式, 為分類結果
		* @param		reference, 輸入. 輸入且為CV_8U格式. 輸出為binary, 0: negative, 1: positive
		* 
		* @return		a matrix in CV_32S (int) form. Also output's [0]: tp; [1]: fp; [2]: tn; [3]: fn.
		*/
		cv::Mat	eval_single(const cv::Mat &output,const cv::Mat &reference);

		/**
		* @brief		評估roc or precision/recall curve, 輸出為所有點的資料, 提供roc or pre-recall兩種選擇, 改變type 0 or 1即可
		*
		* @author		Yunfu Liu (yunfuliu@gmail.com)
		* @todo			此有加快的機會, 例如將confidence進行sorting, 如此即可以累加的方式取得有所資料, 不需要重複運算for不同的thresholds
		*
		* @param		confidence, 輸入. 需為CV_32F格式
		* @param		reference, 輸入. 輸入且為CV_8U格式. 輸出為binary, 0: negative, 1: positive
		* @param		f1score: 計算f1score, 只有當type=CVML_ROCEVAL_PRERECALL才是正確的f1score; 反之, 當type=CVML_ROCEVAL_PRERECALL時, 公式有更改, 但意義相同
		* @param		nPoints, 參數. 表示在roc curve上面要有幾個點, 且為uniformly的去選擇點
		* @param		type,	CVML_ROCEVAL_ROC:		[0,1]=[fpr,tpr]; CVML_ROCEVAL_PRERECALL:	[0,1]=[recall,precision];
		* @param		save_output, 確認是否需要儲存roc等的curve
		* @param		filename, save的儲存檔案名稱
		* 
		* @return		nPoints x 2之matrix, 置放上述結果. 根據type改變而有所不同.
		*/
		#define CVML_ROCEVAL_ROC		0
		#define CVML_ROCEVAL_PRERECALL	1
		cv::Mat	eval(const cv::Mat &confidence,const cv::Mat &reference,float &f1score,const int nPoints=100,const int type=CVML_ROCEVAL_ROC,bool save_output=false,std::string filename = "output.xls");
		
	}

	/// CvMLDate related add-ons
	namespace MLData{

		/**
		* @brief		設定mldata
		*
		* @warning		關於ratioTrainData的設定, 假設設定1., 則最少還會有一個sample於test set; 反之設定為0., 最少還會有一個sample於training set
		*
		* @param		ratioTrainData: 設定有多少比例的Data將會作為訓練使用; range為0~1. 
		* @param		mix: 是否將train and test sample編號混和編入, or not, 預設為true
		*/
		bool	set(CvMLData &data,const float ratioTrainData,const bool mix=true);

		/**
		* @todo			可以判定一下裡面是否可以看外面有給需求才進行, 反之不需要, 則可以加快處理速度. 
		*/
		bool	split(CvMLData &data,cv::Mat &train_featurevector=cv::Mat(),cv::Mat &train_responses=cv::Mat(),cv::Mat &test_featurevector=cv::Mat(),cv::Mat &test_responses=cv::Mat(),cv::Mat &varIdx=cv::Mat(),cv::Mat &sampleIdx=cv::Mat(),cv::Mat &varType=cv::Mat(),cv::Mat &missingDataMask=cv::Mat());
		
		/**
		* @brief		取得一個reference, 其中指定的label為positive時, output=1; 反之negative時, output=0;
		* 
		* @param		train_mode: true: 為取出訓練集合的responses, 反之, 取出測試集合的responses
		* @param		pos_label_name: 指定的positive label
		* @param		neg_label_name: 指定的negative label
		* @param		responses: [input]
		* 
		* @reture		output reference
		*/
		cv::Mat	get_reference(CvMLData &data,const bool train_mode,const std::string pos_label_name,const std::string neg_label_name);
		cv::Mat	get_reference(cv::Mat &responses,std::map<std::string,int> &label_map,const std::string pos_label_name,const std::string neg_label_name);
	}

	/// Neural networks
	namespace ANN_MLP{
		#define CVML_ANN_BACKPROP	CvANN_MLP_TrainParams::BACKPROP
		#define	CVML_ANN_RPROP		CvANN_MLP_TrainParams::RPROP
		bool	setParams(CvANN_MLP_TrainParams &params,const int train_method=CVML_ANN_BACKPROP,const int max_iter=1000);

		/**
		* @param		hiddenLayerSize: Is a Nx1 matrix, to set the number of units in hidden layer. i.e., [7,10] denotes the second layer has 7 units and third layer has 10 units. 
		*/
		bool	train(CvANN_MLP &ann,CvMLData &data,CvANN_MLP_TrainParams &params,cv::Mat &hiddenLayerSize,int flag_code=0,int activateFunc=CvANN_MLP::SIGMOID_SYM);

	}

	/// SVM
	namespace SVM{
		bool	setParams(CvSVMParams &params,const int svm_type=CvSVM::C_SVC,const int kernel_type=CvSVM::RBF, CvMat* class_weights=0, const int max_iter_for_convergence=1000);
		bool	train(CvSVM &svm,CvMLData &data,CvSVMParams &params,const int k_fold=10);
	}

	/// Boosting
	namespace	Boost{

		/**
		* @param		same to CvBoostParams, and their definitions present in .cpp
		*/
		bool	setParams(CvBoostParams &params,int boost_type=CvBoost::REAL, int weak_count=100, double weight_trim_rate=0.95f, int max_depth=1, bool use_surrogates=true, const float* priors=0);
		bool	train(CvBoost &boost,CvMLData &data,CvBoostParams &params,bool update);
		
		/**
		* @brief		Train a cascade structure of classifiers
		*
		* @warning		注意, 此function會自動將分類器即門檻值進行儲存
		* @warning		注意, 此function會自動建議一個資料夾, 為cascade_temp, 用以儲存訓練過程之資料, 例如每一個strong classifier and 對應的門檻值. 
		* @todo			看一下當負樣本沒有了是否要停止
		* @todo			將評估那邊也改為當bebug開啟時才會跑
		* @todo			關於弱分類器數量的選擇可改用搜尋演算法來解, or 平行, 如此速度還可以再加快.
		* @todo			看是否需要還要將目前的training set再次分為train set and cv set, 以提供進行門檻值的調整
		* @todo			train的update參數待確認
		*
		* @param		data: [input] training data
		* @param		param: [input] param for all boost classifiers
		* @param		sc_tpr and sc_fpr: [input] 每一個stage的可接受的tpr and fpr, 此兩個數值皆需介於0~1之間. 
		* @param		label_names: [input] 用以判斷需要哪一個資料庫的哪一個label作為正樣本or負樣本. 此應該為std::string[2]的結構, 其中[0]為正樣本,[1]為負樣本
		* @param		nStrongClassifiers: [input] 預計達到的強分類器數量.
		* @param		upperlimitNWeakCount: [input] 極限的每一個強分類器的弱分類器數量.
		* @param		update: [input] a param for CvBoost.train()
		* @param		debugmode: [input] 是否為bebug mode, 會產生debug info之類. 0: no debug info, 1: rough info, 2: detailed info
		*/
		bool	cascade_train(CvMLData &data,CvBoostParams &param,float sc_tpr,float sc_fpr,const std::string *label_names,const int nStrongClassifiers=10,const int upperlimitNWeakCount=200,const bool update=true,const short debugmode=1);
		
		/**
		* @brief		predict by cascade
		* 
		* @param		boost_vec: [input] boost vector
		* @param		boost_thres: [input] boost's thresholds for each strong classifier
		* @param		featurevector: [input] feature vector
		*
		* @return		confidence, positive>0; negative<=0. (so user can use 0 as the basic threshold for classification)
		*/
		double	cascade_predict(const std::vector<CvBoost> &boost_vec,const std::vector<float> &boost_thres,cv::Mat &featurevector);
		
		/**
		* @brief		load cascade
		* 
		* @param		boost_vec: [output] boost vector
		* @param		boost_thres: [output] boost's thresholds for each strong classifier
		* @param		pth: [input] the default path for temporary files, should include the cascades and thresholds
		*/
		bool	cascade_load(std::vector<CvBoost> &boost_vec,std::vector<float> &boost_thres,const std::string pth="cascade_temp");
	}

}
#endif