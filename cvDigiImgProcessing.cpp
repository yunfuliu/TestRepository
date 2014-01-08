#include "cvDigiImgProcessing.h"

//////////////////////////////////////////////////////////////////////////
bool cvdip::filtering::medianfilter(const cv::Mat &src,cv::Mat &dst,cv::Size blocksize){

	//////////////////////////////////////////////////////////////////////////
	if(blocksize.width>src.cols||blocksize.height>src.rows){
		return false;
	}
	if(blocksize.width%2==0||blocksize.height%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	const	int	half_block_height	=	blocksize.height/2;
	const	int	half_block_width	=	blocksize.width/2;
	std::vector<uchar>	temp_img(blocksize.height*blocksize.width,0);
	// process
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			for(int m=-half_block_height;m<=half_block_height;m++){
				for(int n=-half_block_width;n<=half_block_width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=src.data[(i+m)*src.cols+(j+n)];
					}else{
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=0;
					}
					
				}
			}
			// ordering
			for(int m=0;m<blocksize.height*blocksize.width;m++){
				for(int n=0;n<blocksize.height*blocksize.width-1;n++){
					if(temp_img[n]>temp_img[n+1]){
						double temp=temp_img[n+1];
						temp_img[n+1]=temp_img[n];
						temp_img[n]=temp;
					}										//將value陣列內數值由大至小排列
				}
			}
			tdst.data[i*tdst.cols+j]	=	temp_img[(blocksize.height*blocksize.width-1)/2];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool cvdip::edgedetection::Sobel(const cv::Mat &src, cv::Mat &dst){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// get magnitude and angle by Sobel operator
	const	double	Sobel_V[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}},
					Sobel_H[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};

	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		dst.data[i*dst.cols]=0;
		dst.data[i*dst.cols+(src.cols-1)]=0;
	}
	for(int j=0;j<src.cols;j++){
		dst.data[j]=0;
		dst.data[(src.rows-1)*dst.cols+j]=0;
	}
	for(int i=1;i<src.rows-1;i++){
		for(int j=1;j<src.cols-1;j++){
			float	Sh=0.,Sv=0.;	// 水平值and垂直值
			for(int m=-1;m<=1;m++){
				for(int n=-1;n<=1;n++){
					Sh+=(double)Sobel_H[m+1][n+1]*src.data[(i+m)*src.cols+(j+n)];
					Sv+=(double)Sobel_V[m+1][n+1]*src.data[(i+m)*src.cols+(j+n)];
				}
			}
			// get mag
			float	tempv	=	sqrt((double)Sh*Sh+(double)Sv*Sv);
			if(tempv>255){
				dst.data[i*dst.cols+j]=255.;
			}else if(tempv<0){
				dst.data[i*dst.cols+j]=0.;
			}else{
				dst.data[i*dst.cols+j]	=	(int)(tempv	+0.5);
			}
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool cvdip::halftoning::dotdiffusion::GuoLiu2009(const cv::Mat &src, cv::Mat &dst,const int ClassMatrixSize){



	//////////////////////////////////////////////////////////////////////////
	// exception	
	if(ClassMatrixSize!=8&&ClassMatrixSize!=16){	// 檢查參數輸入是否正確
		return false;
	}
	if(src.type()!=CV_8UC1){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const int	DiffusionMaskSize=3;
	const int	nColors	=	256;
	// define class matrix and diffused weighting
	const	int		ClassMatrix8[8][8]={{22,5,57,8,45,30,36,19},{40,58,32,18,1,43,29,38},{34,4,62,42,20,16,48,37},{28,7,21,56,15,3,49,11},{6,23,35,17,55,51,50,44},{47,12,39,26,25,27,63,61},{14,46,41,31,2,33,60,13},{9,24,52,0,53,54,59,10}};
	const	int		ClassMatrix16[16][16]={	{204,0,5,33,51,59,23,118,54,69,40,160,169,110,168,188},{3,6,22,36,60,50,74,115,140,82,147,164,171,142,220,214},{14,7,42,16,63,52,94,56,133,152,158,177,179,208,222,1},{15,26,43,75,79,84,148,81,139,136,166,102,217,219,226,4},{17,39,72,92,103,108,150,135,157,193,190,100,223,225,227,13},{28,111,99,87,116,131,155,112,183,196,181,224,232,228,12,21},{47,120,91,105,125,132,172,180,184,205,175,233,245,8,20,41},{76,65,129,137,165,145,178,194,206,170,229,244,246,19,24,49},{80,73,106,138,176,182,174,197,218,235,242,249,247,18,48,68},{101,107,134,153,185,163,202,173,231,241,248,253,44,88,70,45},{123,141,149,61,195,200,221,234,240,243,254,38,46,77,104,109},{85,96,156,130,203,215,230,250,251,252,255,53,62,93,86,117},{151,167,189,207,201,216,236,239,25,31,34,113,83,95,124,114},{144,146,191,209,213,237,238,29,32,55,64,97,126,78,128,159},{187,192,198,212,9,10,30,35,58,67,90,71,122,127,154,161},{199,210,211,2,11,27,37,57,66,89,98,121,119,143,162,186}};
	const	double	coe8[3][3]={{0.47972,1,0.47972},{1,0,1},{0.47972,1,0.47972}},
					coe16[3][3]={{0.38459,1,0.38459},{1,0,1},{0.38459,1,0.38459}};

	// = = = = = get class matrix and diffused weighting = = = = = //
	std::vector<std::vector<int>>		CM(ClassMatrixSize,std::vector<int>(ClassMatrixSize,0));
	std::vector<std::vector<double>>	DW(DiffusionMaskSize,std::vector<double>(DiffusionMaskSize,0));
	if(ClassMatrixSize==8){
		for(int i=0;i<ClassMatrixSize;i++){
			for(int j=0;j<ClassMatrixSize;j++){
				CM[i][j]=ClassMatrix8[i][j];
			}
		}
		for(int i=0;i<DiffusionMaskSize;i++){
			for(int j=0;j<DiffusionMaskSize;j++){
				DW[i][j]=coe8[i][j];
			}
		}
	}else if(ClassMatrixSize==16){
		for(int i=0;i<ClassMatrixSize;i++){
			for(int j=0;j<ClassMatrixSize;j++){
				CM[i][j]=ClassMatrix16[i][j];
			}
		}
		for(int i=0;i<DiffusionMaskSize;i++){
			for(int j=0;j<DiffusionMaskSize;j++){
				DW[i][j]=coe16[i][j];
			}
		}
	}

	// = = = = = processing = = = = = //
	std::vector<std::vector<int>>	ProcOrder(src.rows,std::vector<int>(src.cols,0));
	std::vector<std::vector<double>>	tdst(src.rows,std::vector<double>(src.cols,0));
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			tdst[i][j]	=	src.data[i*src.cols+j];
		}
	}
	// 取得處理順序
	for(int i=0;i<src.rows;i+=ClassMatrixSize){
		for(int j=0;j<src.cols;j+=ClassMatrixSize){
			for(int m=0;m<ClassMatrixSize;m++){
				for(int n=0;n<ClassMatrixSize;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						ProcOrder[i+m][j+n]=CM[m][n];
					}
				}
			}
		}
	}
	// 進行dot_diffusion
	int		OSCW=DiffusionMaskSize/2;
	int		OSCL=DiffusionMaskSize/2;
	int		number=0;
	while(number!=ClassMatrixSize*ClassMatrixSize){
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){
				if(ProcOrder[i][j]==number){
					// 取得error
					double	error;
					if(tdst[i][j]<(float)(nColors-1.)/2.){
						error=tdst[i][j];
						tdst[i][j]=0.;
					}else{
						error=tdst[i][j]-(nColors-1.);
						tdst[i][j]=(nColors-1.);
					}
					// 取得分母
					double	fm=0.;
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCL;n<=OSCL;n++){
							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
								if(ProcOrder[i+m][j+n]>number){		// 可以擴散的區域
									fm+=DW[m+OSCW][n+OSCL];
								}
							}
						}
					}
					// 進行擴散
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCL;n<=OSCL;n++){
							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
								if(ProcOrder[i+m][j+n]>number){		// 可以擴散的區域								
									tdst[i+m][j+n]+=error*DW[m+OSCW][n+OSCL]/fm;
								}
							}
						}
					}
				}
			}
		}
		number++;
	}


	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	(uchar)tdst[i][j]+0.5;
			assert(dst.data[i*dst.cols+j]==0||dst.data[i*dst.cols+j]==(nColors-1));
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool cvdip::enhancement::global::WadudKabirDewanChae2007(const cv::Mat &src, cv::Mat &dst, const int x){

	//////////////////////////////////////////////////////////////////////////
	//	exception process
	if (src.type()!=CV_8U){
		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	dst=src.clone();
	//////////////////////////////////////////////////////////////////////////
	//	step	1	:	histogram partition
	//	get histogram
	int		hist[256]={0};
	for (int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			hist[(int)src.data[i*src.cols+j]]++;
		}
	}
	//	smooth histogram
	int smoothfilter[3]={1,1,1};
	for(int i=0;i<256;i++){
		int		tempnum=0;
		float	sum=0;
		for (int j=-1;j<2;j++){
			if ((i+j)>=0 && (i+j)<256){
				sum+=(float)hist[i+j]*(float)smoothfilter[j+1];
				tempnum++;
			}	
		}
		tempnum*=smoothfilter[0];
		hist[i]=(int)(sum/(float)tempnum+0.5);
	}
	//	get minima for sub-histogram
	int	count=0;				//	pointer of minima array.
	int	minima[256]={0};		//	儲存最小值
	bool PartitionFlag=false;	//	true:進行histogram分區, 並判斷是否符合高斯的68.3%分布
	bool SubHistFlag=false;		//	true:histogram分區後, low 換到 high histogram 的68.3%判斷
	bool SubHistFlag2=false;
	double sumFactor=0.;			//	sum of factor.
	double range[256]={0};
	int q=0;
	for (int i=0;i<256;i++){
		if ((i-1)>=0 && (i+1)<256 || i==0 || i==255){
			//	get first non-zero number
			if (hist[i-1]==0 && hist[i]!=0 || (i==0 && hist[0]!=0)){
				minima[count]=i;
				count++;
				PartitionFlag=true;
			}	
			//	get minima number
			if (hist[i]<hist[i-1] && hist[i]<hist[i+1]){
				minima[count]=i;
				count++;
				PartitionFlag=true;
			}
			//	get last non-zero number && i==0, hist[0]!=0
			if (hist[i]!=0 && hist[i+1]==0 || i==255){
				minima[count]=i;
				count++;
				PartitionFlag=true;
			}
			if (count==1){					//	第一個minima不進行分區
				PartitionFlag=false;
			}
			if (minima[0]==minima[1]){		//	修正上面判斷BUG
				count=1;
				PartitionFlag=false;
			}
			//	judge is (mean +- standard deviation) satisfy 68.3% of GL or not.
			int a=0;		
			while (PartitionFlag){
				double	sum=0, mean=0, sd=0, temp=0;
				//	get mean
				for (int k=minima[count-2];k<=minima[count-1];k++){
					mean+=(double)hist[k]*k;
					sum+=(double)hist[k];
				}
				mean/=sum;
				//	get standard deviation
				for (int k=minima[count-2];k<=minima[count-1];k++){
					sd+=(pow((double)k-mean,2)*(double)hist[k]);
				}
				sd=sqrt(sd/sum);
				//	judge 68.3% for (mean +- sd)
				for (int k=(int)(mean-sd+0.5);k<=(int)(mean+sd+0.5);k++){
					temp+=(double)hist[k];
				}
				temp/=sum;
				if (temp>=0.683){
					if (SubHistFlag){		//	(mean+sd) 至 high-minima的高斯分布判定
						if(SubHistFlag2){
							count+=3;
							SubHistFlag2=false;
						}else{
							count+=2;
						}
						SubHistFlag=false;
						a=0;
					}else{
						PartitionFlag=false;
					}					
				}else{						//	low-minima 至 (mean-sd)的高斯分布判定.
					if(a>0){
						for (int m=0;m<=a;m++){
							minima[count+m+2]=minima[count+m];
							SubHistFlag2=true;
						}
					}
					minima[count+1]=minima[count-1];
					minima[count]=(int)(mean+sd+0.5);
					minima[count-1]=(int)(mean-sd+0.5);
					SubHistFlag=true;
					a++;
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//	step	2	:	gray level allocation by cumulative frequencies (CF)
	//////////////////////////////////////////////////////////////////////////
	for (int i=1;i<count;i++){
		double	sumA=0.;
		for (int j=minima[i-1];j<minima[i];j++){
			sumA+=(double)hist[j];
		}
		if(sumA!=0){
			double a=log10(sumA);
			range[i]=(minima[i]-minima[i-1])*pow(a,x);
			sumFactor+=range[i];
		}
	}
	double	a=0.;
	for (int i=0;i<count;i++){
		range[i]=range[i]*255./sumFactor;
		a+=range[i];
		range[i]=a;
	}
	//////////////////////////////////////////////////////////////////////////
	//	step	3	:	histogram equalization
	//////////////////////////////////////////////////////////////////////////
	double	cdf[256]={0.};
	for(int i=1;i<count;i++){
		double	sumCdf=0.;
		double	sumGL=0.;
		for (int j=minima[i-1];j<minima[i];j++){
			sumGL+=(double)hist[j];
		}
		for (int j=minima[i-1];j<minima[i];j++){
			sumCdf+=(double)hist[j]/sumGL;
			cdf[j]=sumCdf;
		}
		for (int j=minima[i];j<256;j++){
			cdf[j]=1;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//	image output
	//////////////////////////////////////////////////////////////////////////
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int a=0;
			for(int k=1;k<count;k++){
				if (minima[k-1]<(int)src.data[i*src.cols+j] && minima[k]>=(int)src.data[i*src.cols+j]){
					a=k;
					break;
				}
			}
			dst.data[i*src.cols+j]=(uchar)(cdf[(int)src.data[i*src.cols+j]]*(range[a]-range[a-1])+range[a-1]);
		}
	}

	return true;
}

bool cvdip::enhancement::global::GlobalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst){

	const int nColors	=	256;

	std::vector<double>	Histogram(nColors,0);	// 256個灰階值

	// 進行統計
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			Histogram[(int)(src.data[i*src.cols+j])]++;
		}
	}
	for(int graylevel=0;graylevel<nColors;graylevel++){
		Histogram[graylevel]/=(double)(src.rows*src.cols);
	}

	// 將Histogram改為累積分配函數
	for(int graylevel=1;graylevel<nColors;graylevel++){
		Histogram[graylevel]+=Histogram[graylevel-1];
	}

	// 取得新的輸出值
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			double	tempv	=	Histogram[(int)(src.data[i*src.cols+j])];
			if(tempv>1){
				tempv=1.;
			}
			assert(tempv>=0.&&tempv<=1.);
			tdst.data[i*src.cols+j]=tempv*(nColors-1.);	// 最多延展到255			
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool cvdip::enhancement::local::LiuJinChenLiuLi2011(const cv::Mat &src,cv::Mat &dst,const cv::Size N){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}
	if(N.width>src.cols||N.height>src.rows){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	const int	nColors	=	256;	// how many colors in the input image
	dst.create(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	// define the stepsize
	float		tempv1	=	(float)src.cols/N.width,
				tempv2	=	(float)src.rows/N.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);
	
	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<std::vector<float>>>	Tinput(N.height,std::vector<std::vector<float>>(N.width,std::vector<float>(nColors,0)));
	// get cdf of each block (Step 1)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// compute pdf, then compute cdf to store in Tinput
			std::vector<float>	pdf(nColors,0);
			int	temp_count=0;
			for(int m=0;m<stepsize.height;m++){
				for(int n=0;n<stepsize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}					
				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(float)temp_count;
			}

			// get cdf
			Tinput[i/stepsize.height][j/stepsize.width][0]=pdf[0];
			for(int m=1;m<nColors;m++){
				Tinput[i/stepsize.height][j/stepsize.width][m] = Tinput[i/stepsize.height][j/stepsize.width][m-1] + pdf[m];
				if(Tinput[i/stepsize.height][j/stepsize.width][m]>1.){
					Tinput[i/stepsize.height][j/stepsize.width][m]=1.;
				}
				assert(Tinput[i/stepsize.height][j/stepsize.width][m]>=0.&&Tinput[i/stepsize.height][j/stepsize.width][m]<=1.);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// get enhanced result (Step 3)
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			// enhance each pixel (A: current; B: right; C: top; D: top-right)
			float	enh_A=-1,enh_B=-1,enh_C=-1,enh_D=-1;	// the reason why not use the 0 to instead of -1 is for the following decision (to check whether that block had been accessed or not)
			enh_A	=	Tinput[i/stepsize.height][j/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];	// enh_x here denotes only the enhanced result
			if((float)(j+stepsize.width)/stepsize.width<N.width){
				enh_B	=	Tinput[i/stepsize.height][(j+stepsize.width)/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			if((float)(i+stepsize.height)/stepsize.height<N.height){
				enh_C	=	Tinput[(i+stepsize.height)/stepsize.height][j/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			if((float)(i+stepsize.height)/stepsize.height<N.height&&(float)(j+stepsize.width)/stepsize.width<N.width){
				enh_D	=	Tinput[(i+stepsize.height)/stepsize.height][(j+stepsize.width)/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			
			// enhancement
			double	weight_A	=	(stepsize.height+1-(i%stepsize.height+1))	*	(stepsize.width+1-(j%stepsize.width+1)),	// this is to represent the weight for each block only
					weight_B	=	(stepsize.height+1-(i%stepsize.height+1))	*	(j%stepsize.width+1),
					weight_C	=	(i%stepsize.height+1)						*	(stepsize.width+1-(j%stepsize.width+1)),
					weight_D	=	(i%stepsize.height+1)						*	(j%stepsize.width+1);

			double	temp_dst		=	(double)(1./((enh_A==-1?0:weight_A)+(enh_B==-1?0:weight_B)+(enh_C==-1?0:weight_C)+(enh_D==-1?0:weight_D)))	*	// this equation is additional added since the paper did not give the process when meet the boundary of an image and the normalize term is bigger than the sum of all the weights. 
										((double)	(enh_A==-1?0:enh_A)		*	weight_A	+	// also, this strategy is to make sure that only the accessed parts are added in this calculation.									
										(double)	(enh_B==-1?0:enh_B)		*	weight_B	+		
										(double)	(enh_C==-1?0:enh_C)		*	weight_C	+			
										(double)	(enh_D==-1?0:enh_D)		*	weight_D);
			
			assert(temp_dst>=0&&temp_dst<=255.);
			dst.data[i*src.cols+j]	=	(int)((temp_dst	*	255.)+0.5);	// (Step 2)

		}
	}

	return true;
}
bool cvdip::enhancement::local::JiaoSunSha2010(const cv::Mat &src,cv::Mat &dst,const cv::Size BlockSize,const cv::Size StepSize,const short mode){



	std::cout	<<	"this function is not completed yet!"	<<	std::endl;
	return false;
	// TEN還沒有實現, 目前寄送mail過去詢問該怎麼處理看似bug的問題. Sept. 3, 2013. 



	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}
	if(BlockSize.width>src.cols||BlockSize.height>src.rows){
		return false;
	}
	if(BlockSize.width%2==1||BlockSize.height%2==1){	// should be in even size
		return false;
	}
	if(StepSize.width%2==1||StepSize.height%2==1){	// should be in even size
		return false;
	}
	if(StepSize.height>BlockSize.height||StepSize.width>StepSize.width){
		return false;
	}
	if(mode!=1&&mode!=2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	const int	nColors	=	256;	// how many colors in the input image
	dst.create(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<short int>>		SM(src.rows,std::vector<short int>(src.cols,0));
	std::vector<std::vector<float>>			temp_dst(src.rows,std::vector<float>(src.cols,0));
	// get temp result
	for(int i=0;i<src.rows;i+=StepSize.height){
		for(int j=0;j<src.cols;j+=StepSize.width){

			//////////////////////////////////////////////////////////////////////////
			// various parameters
			short int	opt_p	=	-1,
						opt_q	=	-1;
			float		opt_cost=	0.;
			std::vector<std::vector<float>>	temp_dstblock(BlockSize.height,std::vector<float>(BlockSize.width,0));
			for(short int p=1;p<=100;p++){
				for(float q=1;q<=3;q+=0.1){

					//////////////////////////////////////////////////////////////////////////
					// initialization
					for(int m=0;m<BlockSize.height;m++){
						for(int n=0;n<BlockSize.width;n++){
							temp_dstblock[m][n]	=	0.;
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// process for each block
					for(int m=0;m<BlockSize.height;m++){
						for(int n=0;n<BlockSize.width;n++){

							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){

								double	norm_src	=	(double)src.data[(i+m)*src.cols+(j+n)]/(nColors-1);
								double	tempv		=	pow((double)	log((double)(1.+norm_src))	/	log((double)(1.+p))	,1./q);
								assert(tempv>=0.&&tempv<=1.);

								// result
								temp_dstblock[m][n]	+=	tempv*255.;
							}
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// get opt p and q with either EME or TEN simulations
					float	current_cost;
					if(mode==1){	// use EME

						// calculate
						float	local_maxv	=	temp_dstblock[0][0],
								local_minv	=	temp_dstblock[0][0];
						for(int m=0;m<BlockSize.height;m++){
							for(int n=0;n<BlockSize.width;n++){
								if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// this's very important to make sure that do not access out of the scope 
									if(temp_dstblock[m][n]>local_maxv){
										local_maxv	=	temp_dstblock[m][n];
									}
									if(temp_dstblock[m][n]<local_minv){
										local_minv	=	temp_dstblock[m][n];
									}
								}
							}
						}
						// calc EME (Eq. 4)
						current_cost	=	local_maxv/local_minv;

					}else{	// use TEN	***********************************************************************************************************************

					}
					if(current_cost>opt_cost){	// the higher the better
						opt_cost	=	current_cost;
						opt_p		=	p;
						opt_q		=	q;
					}

				}
			}

			//////////////////////////////////////////////////////////////////////////
			// get final block each
			for(int m=0;m<BlockSize.height;m++){
				for(int n=0;n<BlockSize.width;n++){

					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){

						double	norm_src	=	(double)src.data[(i+m)*src.cols+(j+n)]/(nColors-1);
						double	tempv		=	pow((double)	log((double)(1.+norm_src))	/	log((double)(1.+opt_p))	,1./opt_q);
						assert(tempv>=0.&&tempv<=1.);

						// result
						temp_dst[i+m][j+n]	+=	tempv*255.;
						SM[i+m][j+n]		+=	1;
					}
				}
			}

		}
	}

	//////////////////////////////////////////////////////////////////////////
	// get final result
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	(int)((double)temp_dst[i][j]/SM[i][j]	+0.5);
			assert((float)temp_dst[i][j]/SM[i][j]>=0&&(float)temp_dst[i][j]/SM[i][j]<nColors);
		}
	}

	return true;
}
bool cvdip::enhancement::local::LambertiMontrucchioSanna2006(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S){

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const	short	nColors	=	256;	// 影像之顏色數量.

	//////////////////////////////////////////////////////////////////////////
	// transformation (block size)
	float	tempv1	=	(float)src.cols/B.width,
			tempv2	=	(float)src.rows/B.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	blocksize((int)tempv1,(int)tempv2);
	tempv1	=	(float)src.cols/S.width;
	tempv2	=	(float)src.rows/S.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(blocksize.height>src.rows||blocksize.width>src.cols||blocksize.height==1||blocksize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
	if(stepsize.height>blocksize.height/2||stepsize.width>blocksize.width/2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
// 	// output image
// 	outputImage=new double*[width];
// 	for (int i=0;i<width;i++)
// 		outputImage[i]=new double[length];
	// transformation functions (S4P3-step a)
	std::vector<std::vector<std::vector<float>>>	Tinput(S.height,std::vector<std::vector<float>>(S.width,std::vector<float>(nColors,0)));
	std::vector<std::vector<std::vector<float>>>	Toutput(S.height,std::vector<std::vector<float>>(S.width,std::vector<float>(nColors,0)));
// 	Tinput = new double**[S+LPF_SIZE-1];
// 	Toutput = new double**[S+LPF_SIZE-1];
// 	for (int i=0;i<S+LPF_SIZE-1;i++){
// 		Tinput[i] = new double*[S+LPF_SIZE-1];
// 		Toutput[i] = new double*[S+LPF_SIZE-1];
// 		for (int j=0;j<S+LPF_SIZE-1;j++){
// 			Tinput[i][j] = new double[256];
// 			Toutput[i][j] = new double[256];
// 		}
// 	}

	//////////////////////////////////////////////////////////////////////////
	// get transformation functions (S4P3-step b)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// computing PDF
			std::vector<float>	pdf(nColors,0);
			int	temp_count=0;
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}					
				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(float)temp_count;
			}

			// computing CDF that is stored in Tinput
			Tinput[i/stepsize.height][j/stepsize.width][0]=pdf[0];
			for(int m=1;m<nColors;m++){
				Tinput[i/stepsize.height][j/stepsize.width][m] = Tinput[i/stepsize.height][j/stepsize.width][m-1] + pdf[m];
				if(Tinput[i/stepsize.height][j/stepsize.width][m]>1.){
					Tinput[i/stepsize.height][j/stepsize.width][m]=1.;
				}
				assert(Tinput[i/stepsize.height][j/stepsize.width][m]>=0.&&Tinput[i/stepsize.height][j/stepsize.width][m]<=1.);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// copy
	for(int i=0;i<S.height;i++){
		for(int j=0;j<S.width;j++){
			for(int m=0;m<nColors;m++){
				Toutput[i][j][m]	=	Tinput[i][j][m];
			}
		}
	}
	// refine the transformation functions
	int delta = 1;
	const	double	sx	=	log((double)S.width/B.width)/log(2.0);
	const	double	sy	=	log((double)S.height/B.height)/log(2.0);
	double	s	=	sx>sy?sy:sx;
	for(int times=0;times<s;times++){

		// horizontal direction (S4P3-step c)
		for(int i=0;i<S.height;i++){
			for(int j=delta;j<S.width-delta;j++){		
				for(int m=0;m<nColors;m++){
					Toutput[i][j][m] = 0;
					Toutput[i][j][m] += Tinput[i][j-delta][m]/4.;
					Toutput[i][j][m] += Tinput[i][j][m]/2;
					Toutput[i][j][m] += Tinput[i][j+delta][m]/4.;
					assert(Toutput[i][j][m]>=0&&Toutput[i][j][m]<=1);
				}
			}
		}

		// vertical direction (S4P3-step d)
		for(int i=delta;i<S.height-delta;i++){
			for(int j=0;j<S.width;j++){				
				for(int m=0;m<nColors;m++){
					Tinput[i][j][m] = 0;
					Tinput[i][j][m] += Toutput[i-delta][j][m]/4.;
					Tinput[i][j][m] += Toutput[i][j][m]/2.;
					Tinput[i][j][m] += Toutput[i+delta][j][m]/4.;
					assert(Tinput[i][j][m]>=0&&Tinput[i][j][m]<=1);
				}
			}
		}

		delta *= 2;
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<float>>	tdst(src.rows,std::vector<float>(src.cols,0));
	std::vector<std::vector<short>>	accu_count(src.rows,std::vector<short>(src.cols,0));	// the add number of each pixel
	// enhancement
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						tdst[i+m][j+n]	+=	Tinput[i/stepsize.height][j/stepsize.width][(int)src.data[(i+m)*src.cols+(j+n)]]	*	((float)nColors-1);
						accu_count[i+m][j+n]	++;
					}					
				}
			}
		}
	}
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			tdst[i][j]	/=	(float)accu_count[i][j];
			assert(tdst[i][j]>=0.&&tdst[i][j]<=nColors-1.);
		}
	}



	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<dst.rows;i++){
		for(int j=0;j<dst.cols;j++){
			dst.data[i*dst.cols+j]	=	(uchar)(tdst[i][j]	+0.5);
		}
	}

	return true;
}
bool cvdip::enhancement::local::YuBajaj2004(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float C,bool anisotropicMode,const float R){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(blockheight>src.rows||blockheight%2==0){
		return false;
	}
	if(blockwidth>src.cols||blockwidth%2==0){
		return false;
	}
	if(anisotropicMode){
		if(R<0.01||R>0.1){
			return false;
		}
	}else{
		if(C>1||C<0){
			return false;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// initialization
	cv::Mat	tdst;
	tdst.create(src.size(),src.type());
	const	float	w	=	255.;

	//////////////////////////////////////////////////////////////////////////
	// get max, min, and avg
	cv::Mat	maxmap(src.size(),src.type()),
			minmap(src.size(),src.type()),
			avgmap(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			
			float	maxv	=	src.data[i*src.cols+j];
			float	minv	=	src.data[i*src.cols+j];
			float	avgv	=	0.;
			int		avgv_count	=	0;
			for(int m=-blockheight/2;m<=blockheight/2;m++){
				for(int n=-blockwidth/2;n<=blockwidth/2;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						uchar	&currv	=	src.data[(i+m)*src.cols+(j+n)];
						if(currv>maxv){
							maxv=currv;
						}
						if(currv<minv){
							minv=currv;
						}
						avgv+=currv;
						avgv_count++;
					}
				}
			}
			avgv	/=	(float)	avgv_count;

			maxmap.data[i*maxmap.cols+j]	=	maxv;
			minmap.data[i*minmap.cols+j]	=	minv;
			avgmap.data[i*avgmap.cols+j]	=	avgv;		

		}
	}

	//////////////////////////////////////////////////////////////////////////
	// smoothing
	if(anisotropicMode){

		for(int i=0;i<src.rows;i++){
			for(int j=1;j<src.cols;j++){

				// avg
				avgmap.data[i*avgmap.cols+j]	+=	(avgmap.data[i*avgmap.cols+(j-1)] - avgmap.data[i*avgmap.cols+j]) * exp(-R * fabs((float)(avgmap.data[i*avgmap.cols+(j-1)] - avgmap.data[i*avgmap.cols+j])));
				// min
				if(minmap.data[i*minmap.cols+(j-1)] < minmap.data[i*minmap.cols+j]){
					minmap.data[i*minmap.cols+j]	+=	(minmap.data[i*minmap.cols+(j-1)] - minmap.data[i*minmap.cols+j]) * exp(-R * fabs((float)(minmap.data[i*minmap.cols+(j-1)] - minmap.data[i*minmap.cols+j])));	
				}
				// max
				if(maxmap.data[i*maxmap.cols+(j-1)] > maxmap.data[i*maxmap.cols+j]){
					maxmap.data[i*maxmap.cols+j]	+=	(maxmap.data[i*maxmap.cols+(j-1)] - maxmap.data[i*maxmap.cols+j]) * exp(-R * fabs((float)(maxmap.data[i*maxmap.cols+(j-1)] - maxmap.data[i*maxmap.cols+j])));
				}

			}
		}

	}else{

		for(int i=0;i<src.rows;i++){
			for(int j=1;j<src.cols;j++){

				// avg
				avgmap.data[i*avgmap.cols+j]	=	(uchar)((float)(1.-C)*avgmap.data[i*avgmap.cols+j]	+	(float)C*avgmap.data[i*avgmap.cols+(j-1)]	+0.5);
				// min
				if(minmap.data[i*minmap.cols+(j-1)] < minmap.data[i*minmap.cols+j]){
					minmap.data[i*minmap.cols+j]	=	(uchar)((float)(1.-C)*minmap.data[i*minmap.cols+j]	+	(float)C*minmap.data[i*minmap.cols+(j-1)]	+0.5);
				}
				// max
				if(maxmap.data[i*maxmap.cols+(j-1)] > maxmap.data[i*maxmap.cols+j]){
					maxmap.data[i*maxmap.cols+j]	=	(uchar)((float)(1.-C)*maxmap.data[i*maxmap.cols+j]	+	(float)C*maxmap.data[i*maxmap.cols+(j-1)]	+0.5);
				}

			}
		}

	}

	//////////////////////////////////////////////////////////////////////////
	// enhancement
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			// get Inew and Anew
			float	Inew	=	w	*	(float)(src.data[i*src.cols+j] - minmap.data[i*src.cols+j] )/(float)(maxmap.data[i*src.cols+j]  - minmap.data[i*src.cols+j] );
			float	Anew	=	w	*	(float)(avgmap.data[i*src.cols+j] - minmap.data[i*src.cols+j] )/(float)(maxmap.data[i*src.cols+j]  - minmap.data[i*src.cols+j] );

			// get afa
			float	afa	=	(Anew-Inew)/128.;

			// get, afa, beta, gamma
			float	a,b,c;
			a	=	afa	/	(2.	*	w);
			b	=	(float)afa / w * src.data[i*src.cols+j]	-	afa	-	1.;
			c	=	(float)afa / (2.*w) * src.data[i*src.cols+j] * src.data[i*src.cols+j]	-	(float)afa	* src.data[i*src.cols+j] + (float)src.data[i*src.cols+j];			

			// get result
			float	tempv;
			if(afa<-0.000001||afa>0.000001){
				tempv	=	(-b-sqrt((float)b*b-(float)4.*a*c))/(2.*a);
			}else{
				tempv	=	src.data[i*src.cols+j];
			}
			if(tempv>255.){
				tempv=255.;
			}
			if(tempv<0.){
				tempv=0.;
			}
			tdst.data[i*tdst.cols+j]	=	(uchar)	tempv+0.5;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
bool cvdip::enhancement::local::KimKimHwang2001(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S){

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const	short	nColors	=	256;	// 影像之顏色數量.
	// transformation
	cv::Size	blocksize;
	blocksize	=	cv::Size(src.cols/B.width,src.rows/B.height);
	cv::Size	stepsize;
	stepsize	=	cv::Size(src.cols/S.width,src.rows/S.height);

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(blocksize.height>src.rows||blocksize.width>src.cols||blocksize.height==1||blocksize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
	if(stepsize.height>blocksize.height/2||stepsize.width>blocksize.width/2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<float>>	tdst(src.rows,std::vector<float>(src.cols,0));
	std::vector<std::vector<short>>	accu_count(src.rows,std::vector<short>(src.cols,0));	// the add number of each pixel
	// process (S3P5-Steps 3 and 4)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){
			
			// get pdf
			std::vector<float>	pdf(nColors,0.);
			int	temp_count	=	0;
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}

				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(double)temp_count;
			}

			// get cdf
			std::vector<float>	cdf(nColors,0.);
			cdf[0]=pdf[0];
			for(int m=1;m<nColors;m++){
				cdf[m]=cdf[m-1]+pdf[m];
				if(cdf[m]>1.){
					cdf[m]=1;
				}
				assert(cdf[m]>=0.&&cdf[m]<=1.);
			}

			// get enhanced result and accumulate 
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						tdst[i+m][j+n]	+=	(float)cdf[(int)src.data[(i+m)*src.cols+(j+n)]]*(nColors-1);
						accu_count[i+m][j+n]++;
					}
				}
			}			
		}
	}
	// process (S3P5-Step5)
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			tdst[i][j]	/=	(float)accu_count[i][j];
			assert(tdst[i][j]>=0&&tdst[i][j]<=255.);
			
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// BERF (blocking effect reduction filter)
	// for vertical
	for(int i=stepsize.height;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j++){

			// S3P5-Step6 and S3P7
			if (fabs(tdst[i][j]-tdst[i-1][j])>=3){
 				double avg=(tdst[i][j]+tdst[i-1][j])/2.;
 				tdst[i][j]=avg;
 				tdst[i-1][j]=avg;
			}
		}
	}
	// for horizontal
	for(int i=0;i<src.rows;i++){
		for(int j=stepsize.width;j<src.cols;j+=stepsize.width){

			// S3P5-Step6 and S3P7
			if (fabs(tdst[i][j]-tdst[i][j-1])>=3){
				double avg=(tdst[i][j]+tdst[i][j-1])/2.;
				tdst[i][j]=avg;
				tdst[i][j-1]=avg;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	tdst[i][j];
		}
	}

	return true;
}
bool cvdip::enhancement::local::Stark2000(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float alpha,const float beta){

	//////////////////////////////////////////////////////////////////////////
	if(blockheight%2==0){
		return false;
	}
	if(blockwidth%2==0){
		return false;
	}
	if(alpha<0||alpha>1){
		return false;
	}
	if(beta<0||beta>1){
		return false;
	}
	if(src.type()!=CV_8UC1){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	// processing
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			//////////////////////////////////////////////////////////////////////////
			// calc histogram for each pixel
			int numCount=0;
			double hist[256]={0};
			for(int m=-blockheight/2;m<=blockheight/2;m++){
				for(int n=-blockwidth/2;n<=blockwidth/2;n++){
					if( (i+m>=0)&&(j+n>=0)&&(i+m<src.rows)&&(j+n<src.cols) ){
						numCount++;
						hist[(int)src.data[(i+m)*src.cols+(j+n)]]++;
					}
				}
			}
			// change to pdf
			for(int m=0;m<256;m++){
				hist[m] /= numCount;
			}

			//////////////////////////////////////////////////////////////////////////
			// 計算輸出值  ps. 需正規化 -1/2~1/2
			double normalizedinput=((float)src.data[i*src.cols+j]/255.)-0.5;
			assert(normalizedinput>=-0.5&&normalizedinput<=0.5);	
			double output	=	0.;
			for(int c=0;c<256;c++){

				// calc q
				double	q1=0., 
					q2=0., 
					d=normalizedinput-(((double)c/255.)-0.5);
				// for q1 (Eq. (13))
				if(d>0){
					q1	=	0.5*pow((double)2.*d,(double)alpha);
				}else if (d<0){
					q1	=	-0.5*pow((double)fabs(2.*d),(double)alpha);
				}
				// for q2 (Eq. (13))
				if(d>0){
					q2	=	0.5*2.*d;
				}else if (d<0){
					q2	=	-0.5*fabs(2.*d);
				}
				// Eq. (16)
				double	q	=	q1-beta*q2+beta*normalizedinput;

				// Eq. (5)
				output += hist[c]*q;
			}
			// normalize output
			output	=	255.*(output+0.5);
			if(output>255){
				output=255;
			}
			if(output<0){
				output=0;
			}
			tdst.data[i*tdst.cols+j]=(uchar)(output+0.5);
		}
	}

	dst	=	tdst.clone();

	return true;
}
bool cvdip::enhancement::local::LocalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst,const cv::Size blocksize){

	//////////////////////////////////////////////////////////////////////////
	if(blocksize.height%2==0){
		return false;
	}
	if(blocksize.width%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());
	const int nColors	=	256;

	//////////////////////////////////////////////////////////////////////////
	// processing
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			std::vector<double>	Histogram(nColors,0);	// 256個灰階值

			// 進行統計
			int temp_count=0;
			for(int m=-blocksize.height/2;m<=blocksize.height/2;m++){
				for(int n=-blocksize.width/2;n<=blocksize.width/2;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						Histogram[(int)(src.data[(i+m)*src.cols+(j+n)])]++;
						temp_count++;
					}					
				}
			}
			for(int graylevel=0;graylevel<nColors;graylevel++){
				Histogram[graylevel]/=(double)(temp_count);
			}

			// 將Histogram改為累積分配函數
			for(int graylevel=1;graylevel<nColors;graylevel++){
				Histogram[graylevel]+=Histogram[graylevel-1];
			}

			// 取得新的輸出值
			double	tempv	=	Histogram[(int)(src.data[i*src.cols+j])];
			if(tempv>1){
				tempv=1.;
			}
			assert(tempv>=0.&&tempv<=1.);
			tdst.data[i*src.cols+j]=tempv*(nColors-1.);	// 最多延展到255		

		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
//////////////////////////////////////////////////////////////////////////
float cvdip::qualityassessment::EME(const cv::Mat &src,const cv::Size nBlocks,const short mode){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}
	if(nBlocks.width>src.cols||nBlocks.height>src.rows){
		return false;
	}
	if(mode!=1&&mode!=2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// param
	const	float	c	=	0.0001;

	//////////////////////////////////////////////////////////////////////////
	// define the stepsize
	float	tempv1	=	(float)src.cols/nBlocks.width,
			tempv2	=	(float)src.rows/nBlocks.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);

	//////////////////////////////////////////////////////////////////////////
	// estimate
	int		count	=	0;
	float	eme		=	0.;
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// get local max and min
			float	local_maxv	=	src.data[i*src.cols+j],
					local_minv	=	src.data[i*src.cols+j];		
			if(mode==1){	// standard mode

				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){

						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>local_maxv){
								local_maxv	=	src.data[(i+m)*src.cols+(j+n)];
							}
							if(src.data[(i+m)*src.cols+(j+n)]<local_minv){
								local_minv	=	src.data[(i+m)*src.cols+(j+n)];
							}
						}
					}
				}

			}else if(mode==2){	// BTC's mode

				// find first moment and second moment
				double	moment1=0.,moment2=0.;
				int		count_mom=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							moment1+=src.data[(i+m)*src.cols+(j+n)];
							moment2+=src.data[(i+m)*src.cols+(j+n)]*src.data[(i+m)*src.cols+(j+n)];
							count_mom++;
						}
						
					}
				}
				moment1/=(double)count_mom;
				moment2/=(double)count_mom;

				// find variance
				double	sd=sqrt(moment2-moment1*moment1);

				// find num of higher than moment1
				int	q=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>=moment1){
								q++;
							}
						}
					}
				}
				int		m_q=count_mom-q;
				local_minv=moment1-sd*sqrt((double)q/m_q),
				local_maxv=moment1+sd*sqrt((double)m_q/q);
				if(local_minv>255){
					local_minv=255;
				}
				if(local_minv<0){
					local_minv=0;
				}
				if(local_maxv>255){
					local_maxv=255;
				}
				if(local_maxv<0){
					local_maxv=0;
				}
			}else{
				assert(false);
			}

			// calc EME (Eq. 2) -totally same
			if(local_maxv!=local_minv){
				eme	+=	log((double)local_maxv/(local_minv+c));
			}
			count++;

		}
	}

	return (float)20.*eme/count;
}
float cvdip::qualityassessment::TEN(const cv::Mat &src){

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	est;	
	// process
	edgedetection::Sobel(src,est);

	//////////////////////////////////////////////////////////////////////////
	// estimation
	double	ten	=	0.;
	for(int i=0;i<est.rows;i++){
		for(int j=0;j<est.cols;j++){
			ten	+=	est.data[i*est.cols+j]	*	est.data[i*est.cols+j];	// eq. 6
		}
	}

	return (double)ten/(est.rows*est.cols);
}
float cvdip::qualityassessment::AMBE(const cv::Mat &src1,const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	if((src1.rows!=src2.rows)||(src2.cols!=src2.cols)){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	double	mean1=0.,mean2=0.;
	for(int i=0;i<src1.rows;i++){
		for(int j=0;j<src1.cols;j++){
			mean1	+=	(double)src1.data[i*src1.cols+j];
			mean2	+=	(double)src2.data[i*src1.cols+j];
		}
	}
	mean1	/=	(double)(src1.cols*src1.rows);
	mean2	/=	(double)(src2.cols*src2.rows);

	return abs((double)(mean1-mean2));
}
//////////////////////////////////////////////////////////////////////////
std::string	cvml::classindextostring(std::map<std::string,int> &tmap,const int classLabel){
	
	for (std::map<std::string,int>::iterator it = tmap.begin(); it != tmap.end();it++){
		if(it->second==classLabel){
			return	it->first;
		}
	}

	std::cout	<<	"[classIdx]:"	<<	classLabel	<<	" is out of scope"	<<	std::endl;
	assert(false);
	return "";
}
//////////////////////////////////////////////////////////////////////////
bool	cvml::ROC_eval::classify(const cv::Mat &confidence,cv::Mat &output,const double thres){

	// exception
	if(confidence.type()!=CV_32F){
		return false;
	}
	if(output.type()!=CV_8U){	// range 0 ~ 255
		return false;
	}
	if(confidence.size()!=output.size()){
		return false;
	}

	// evaluation
	for(int i=0;i<confidence.rows;i++){	// number of samples
		if(((float*)confidence.data)[i]<thres){	
			output.data[i]=0;	// negative
		}else{
			output.data[i]=1;	// positive
		}		
	}
	return true;
}
cv::Mat	cvml::ROC_eval::eval_single(const cv::Mat &output,const cv::Mat &reference){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(output.type()!=CV_8U){
		return cv::Mat();
	}
	if(reference.type()!=CV_8U){
		std::cout	<<	"[cvml::ROC_eval::eval_single] reference type should be CV_8U. Please use the [cvml::MLData::get_reference] for it."	<<	std::endl;
		return cv::Mat();
	}
	if(output.size()!=reference.size()){
		return cv::Mat();
	}
	double	minv,maxv;
	cv::minMaxLoc(output,&minv,&maxv);
	if((maxv!=0&&maxv!=1)||(minv!=0&&minv!=1)){	// 不為 0 or 1的數值
		return cv::Mat();
	}
	cv::minMaxLoc(reference,&minv,&maxv);
	if((maxv!=0&&maxv!=1)||(minv!=0&&minv!=1)){	// 不為 0 or 1的數值
		return cv::Mat();
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	eval_output(4,1,CV_32S);
	///// evaluation
	int	tp=0,fp=0,tn=0,fn=0;
	for(int i=0;i<output.rows;i++){	// number of samples
		if(reference.data[i]==1){	// if positive appear
			if(output.data[i]	== reference.data[i]){	// output = positive
				tp++;
			}else{
				fn++;
			}
		}else if(reference.data[i]==0){	// if negative appear
			if(output.data[i]	== reference.data[i]){	// output = negative
				tn++;
			}else{
				fp++;
			}
		}else{
			std::cout	<<	"[cvml::ROC_eval::eval]\t"
						<<	"unexpected reference value ["	<<	reference.data[i]
						<<	"] appear."	<<	std::endl;
			assert(false);
		}
	}
	// pass results
	((int*)eval_output.data)[0]	=	tp;
	((int*)eval_output.data)[1]	=	fp;
	((int*)eval_output.data)[2]	=	tn;
	((int*)eval_output.data)[3]	=	fn;

	//////////////////////////////////////////////////////////////////////////
	return eval_output.clone();
}
cv::Mat	cvml::ROC_eval::eval(const cv::Mat &confidence,const cv::Mat &reference,float &f1score,const int nPoints,const int type,bool save_output,std::string filename){

	//////////////////////////////////////////////////////////////////////////
	double	minv,maxv;
	// exceptions
	cv::minMaxLoc(reference,&minv,&maxv);
	if(maxv!=1||minv!=0){
		return cv::Mat();
	}
	if(nPoints<2){
		return cv::Mat();
	}
	if(type!=0&&type!=1){
		return cv::Mat();
	}

	//////////////////////////////////////////////////////////////////////////
	// max and min
	cv::minMaxLoc(confidence,&minv,&maxv);
	double	step	=	(maxv-minv)/((double)(nPoints-1));
	// eval	
	cv::Mat	output(confidence.rows,1,CV_8U);
	cv::Mat	eval_output,
			roc_output(nPoints,2,CV_32F);	// 置放tpr fpr recall precision等資料
	// evaluation
	int	k=0;
	double	tp=0.,fp=0.,tn=0.,fn=0.;
	for(double thres=minv;thres<=maxv+step/2.;thres+=step){	// maxv+step/2.使得判斷不受精度影響, 否則有時候最後一點算不到. 
		
		if(k==nPoints-1){	// 使最後一個門檻值可使所有sample判斷為negative
			thres+=step/2.;
		}

		if(cvml::ROC_eval::classify(confidence,output,thres)){

			eval_output	=	cvml::ROC_eval::eval_single(output,reference);

			if(!eval_output.empty()){

				tp	=	((int*)eval_output.data)[0];
				fp	=	((int*)eval_output.data)[1];
				tn	=	((int*)eval_output.data)[2];
				fn	=	((int*)eval_output.data)[3];

				if(type==0){	// calc roc
					if(tp==0){
						((float*)roc_output.data)[k*roc_output.cols+1]	=	0;			// tpr
					}else{
						((float*)roc_output.data)[k*roc_output.cols+1]	=	tp/(tp+fn);	// tpr
					}
					if(fp==0){
						((float*)roc_output.data)[k*roc_output.cols]	=	0;			// fpr
					}else{
						((float*)roc_output.data)[k*roc_output.cols]	=	fp/(fp+tn);	// fpr
					}
					
				}else if(type==1){	// calc pre-recall
					if(tp==0){	// 避免數值錯誤
						if(fp==0){
							((float*)roc_output.data)[k*roc_output.cols+1]	=	1.;	// precision
						}else{
							((float*)roc_output.data)[k*roc_output.cols+1]	=	0.;	// precision
						}						
						((float*)roc_output.data)[k*roc_output.cols]	=	0.;	// recall
					}else{
						((float*)roc_output.data)[k*roc_output.cols+1]	=	tp/(tp+fp);	// precision
						((float*)roc_output.data)[k*roc_output.cols]	=	tp/(tp+fn);	// recall
					}

				}else{
					assert(false);
				}
				k++;
			}else{
				std::cout	<<"[cvml::ROC_eval::eval] no eval output."	<<	std::endl;
				return cv::Mat();
			}

		}else{
			return cv::Mat();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// calc f1score
	f1score	=	0.;	// initialization
	if(type==1){	// precision/recall
		for(int i=0;i<roc_output.rows;i++){
			float	pre		=	((float*)roc_output.data)[i*roc_output.cols+1];		// precision
			float	rec		=	((float*)roc_output.data)[i*roc_output.cols];	// recall
			assert(pre>=0.&&pre<=1.&&rec>=0.&&rec<=1.);

			// calc
			float	temp_f1score	=	2.*	(pre*rec)/(pre+rec);
			if(temp_f1score>f1score){
				f1score	=	temp_f1score;
			}
		}
	}else if(type==0){	// tpr/fpr, 正常而言f1score並非算此, 這邊更改公式以提供相同意義的供參考
		for(int i=0;i<roc_output.rows;i++){
			float	tpr		=		((float*)roc_output.data)[i*roc_output.cols+1];		// tpr
			float	nfpr	=	1.-	((float*)roc_output.data)[i*roc_output.cols];		// fpr
			assert(tpr>=0.&&tpr<=1.&&nfpr>=0.&&nfpr<=1.);

			// calc
			float	temp_f1score	=	2.*	(tpr*nfpr)/(tpr+nfpr);
			if(temp_f1score>f1score){
				f1score	=	temp_f1score;
			}
		}
	}else{
		assert(false);
	}
	assert(f1score>=0.&&f1score<=1.);

	//////////////////////////////////////////////////////////////////////////
	///// save result
	if(save_output){
		std::ofstream	ost(filename);
		ost	<<	"Output of [cvml::ROC_eval::eval]"	<<	std::endl;
		if(type==0){
			ost	<<	"FPR"	<<	"\t"	<<	"TPR"	<<	std::endl;
		}else if(type==1){
			ost	<<	"Recall"	<<	"\t"	<<	"Precision"	<<	std::endl;
		}
		for(int i=0;i<roc_output.rows;i++){
			for(int j=0;j<roc_output.cols;j++){
				ost	<<	(float)((float*)roc_output.data)[i*roc_output.cols+j]	<<	"\t";
			}
			ost	<<	std::endl;
		}
		ost.close();
	}

	return roc_output.clone();
}
//////////////////////////////////////////////////////////////////////////
bool	cvml::MLData::set(CvMLData &data,const float ratioTrainData,const bool mix){

	/// get value
	cv::Mat	mValue;
	mValue	=	data.get_values();				///< 僅用於取得變數數量

	/// set response index
	const	int	responseIdx	=	mValue.cols-1;	///< 設定最後一個column為class	
	data.set_response_idx(responseIdx);

	// set var idx
	data.change_var_idx(responseIdx,false);		///< 使得該variable將不會作為變數讀入至values

	// set train and test data ratio
	if(ratioTrainData<0.||ratioTrainData>1.){
		std::cout	<<	"[cvml::MLData::set] ratioTrainData is wrong."	<<	std::endl;
		assert(false);
	}
	CvTrainTestSplit	spl(ratioTrainData,mix);	
	data.set_train_test_split(&spl);

	// 測試上述設定是否可用
	cv::Mat	train_sample_idx	=	data.get_train_sample_idx(),
			test_sample_idx		=	data.get_test_sample_idx();
	if(train_sample_idx.empty()&&test_sample_idx.empty()){	///<	表示當初設定的ratioTrainData可能為1. or 0.
		float	diff0	=	ratioTrainData-0.;
		float	diff1	=	1.-ratioTrainData;
		if(diff1>diff0){	///< 表示比例靠近0
			CvTrainTestSplit	spl0((int)1,mix);				// 則設定一個樣本於訓練集合
			data.set_train_test_split(&spl0);
		}else{				///< 表示比例靠近1
			CvTrainTestSplit	spl1((int)mValue.rows-1,mix);	// 則設定一個樣本於測試集合
			data.set_train_test_split(&spl1);
		}
	}

	return true;
}
bool	cvml::MLData::split(CvMLData &data,cv::Mat &train_featurevector,cv::Mat &train_responses,cv::Mat &test_featurevector,cv::Mat &test_responses,cv::Mat &varIdx,cv::Mat &sampleIdx,cv::Mat &varType,cv::Mat &missingDataMask){
	

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	int	responseIdx	=	data.get_response_idx();
	if(responseIdx==-1){	// 尚未設定responseIdx
		std::cout	<<	"[cvml::MLData::split] The CvMLData's response hasn't been set."
					<<	"Need to use the [cvml::MLData::set] first."	<<	std::endl;
		assert(false);
	}

	// pretest train and test feature vectors
	cv::Mat	train_sample_idx,test_sample_idx;
	train_sample_idx	=	data.get_train_sample_idx();
	test_sample_idx		=	data.get_test_sample_idx();
	if(train_sample_idx.empty()&&test_sample_idx.empty()){
		std::cout	<<	"[cvml::MLData::split] train_sample_idx and test_sample_idx are both empty."	<<	std::endl;
		assert(false);
	}

	//////////////////////////////////////////////////////////////////////////	
	///// get train and test feature vectors
	cv::Mat		mValue	=	data.get_values();
	// get train vectors	
	train_featurevector.create(train_sample_idx.cols,responseIdx,CV_32F);
	train_responses.create(train_sample_idx.cols,1,CV_32F);
	for(int i=0;i<train_featurevector.rows;i++){

		// get sample index
		int	sample_index	=	((int*)train_sample_idx.data)[i];

		// copy
		for(int j=0;j<train_featurevector.cols;j++){
			((float*)train_featurevector.data)[i*train_featurevector.cols+j]	=	((float*)mValue.data)[sample_index*mValue.cols+j];
		}
		((float*)train_responses.data)[i*train_responses.cols]	=	((float*)mValue.data)[sample_index*mValue.cols+responseIdx];

	}
	// get test feature vectors
	test_featurevector.create(test_sample_idx.cols,responseIdx,CV_32F);
	test_responses.create(test_sample_idx.cols,1,CV_32F);
	for(int i=0;i<test_featurevector.rows;i++){

		// get sample index
		int	sample_index	=	((int*)test_sample_idx.data)[i];

		// copy
		for(int j=0;j<test_featurevector.cols;j++){
			((float*)test_featurevector.data)[i*test_featurevector.cols+j]	=	((float*)mValue.data)[sample_index*mValue.cols+j];
		}
		((float*)test_responses.data)[i*test_responses.cols]	=	((float*)mValue.data)[sample_index*mValue.cols+responseIdx];

	}

	//////////////////////////////////////////////////////////////////////////
	// get varIdx
	varIdx	=	data.get_var_idx();

	// get sampleIdx
	sampleIdx	=	data.get_train_sample_idx();

	// get varType
	varType	=	data.get_var_types();

	// get missingDataMsk
	missingDataMask	=	data.get_missing();

	return true;
}
cv::Mat	cvml_MLData_get_reference(cv::Mat &responses,std::map<std::string,int> &label_map,const std::string &pos_label_name,const std::string &neg_label_name){

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	//	cv::Mat						mValue		=	data.get_values();
	const	int					nSample		=	responses.rows;

	//////////////////////////////////////////////////////////////////////////
	///// process
	cv::Mat	refs(nSample,1,CV_8U);
	for(int i=0;i<nSample;i++){	// number of samples

		// 取得預測的標籤名稱 in train_map
		std::string	true_label_string	=	cvml::classindextostring(label_map,responses.at<float>(i,0));

		if(true_label_string==pos_label_name){			// if positive's label (classname_ann[0]) appear
			refs.data[i]	=	1;
		}else if(true_label_string==neg_label_name){	// if negative's label (classname_ann[1]) appear
			refs.data[i]	=	0;
		}else{
			std::cout	<<	"[cvml::MLData::get_reference] unexpected label appears."	<<	std::endl;
			return cv::Mat();
		}
	}

	return refs.clone();
}
cv::Mat	cvml::MLData::get_reference(CvMLData &data,const bool train_mode,const std::string pos_label_name,const std::string neg_label_name){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	cv::Mat	responses;
	if(train_mode){	// get train responses
		cvml::MLData::split(data,cv::Mat(),responses);
	}else{	// get test responses
		cvml::MLData::split(data,cv::Mat(),cv::Mat(),cv::Mat(),responses);
	}
	if(responses.empty()){
		std::cout	<<	"[cvml::MLData::get_reference] data.responses is empty."	<<	std::endl;
		return	cv::Mat();
	}

	//////////////////////////////////////////////////////////////////////////
	///// get map
	std::map<std::string,int>	label_map	=	data.get_class_labels_map();

	return cvml_MLData_get_reference(responses,label_map,pos_label_name,neg_label_name);
}
cv::Mat	cvml::MLData::get_reference(cv::Mat &responses,std::map<std::string,int> &label_map,const std::string pos_label_name,const std::string neg_label_name){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(responses.empty()){
		std::cout	<<	"[cvml::MLData::get_reference] data.responses is empty."	<<	std::endl;
		return	cv::Mat();
	}

	return cvml_MLData_get_reference(responses,label_map,pos_label_name,neg_label_name);
}
//////////////////////////////////////////////////////////////////////////
bool	cvml::ANN_MLP::setParams(CvANN_MLP_TrainParams &params,const int train_method,const int max_iter){
	
	params.term_crit	=	cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iter, 0.01);
	
	// set train method
	if(train_method==0){
		params.train_method	=	CvANN_MLP_TrainParams::BACKPROP;
	}else if(train_method==1){
		params.train_method	=	CvANN_MLP_TrainParams::RPROP;
	}else{
		assert(false);
	}
	
	// default
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;
	params.rp_dw0 = 0.1; 
	params.rp_dw_plus = 1.2; 
	params.rp_dw_minus = 0.5;
	params.rp_dw_min = FLT_EPSILON; 
	params.rp_dw_max = 50.;

	return true;
}
bool	cvml::ANN_MLP::train(CvANN_MLP &ann,CvMLData &data,CvANN_MLP_TrainParams &params,cv::Mat &hiddenLayerSize,int flag_code,int activateFunc){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(hiddenLayerSize.type()!=CV_32S){
		assert(false);
	}
	if(hiddenLayerSize.empty()){
		assert(false);
	}
	if(hiddenLayerSize.cols!=1){
		assert(false);
	}
	if(hiddenLayerSize.rows<1){
		assert(false);
	}


	//////////////////////////////////////////////////////////////////////////
	// set flag
	if(flag_code==0){
		flag_code	=	0;
	}else if(flag_code==1){
		flag_code	=	CvANN_MLP::UPDATE_WEIGHTS;
	}else if (flag_code==2){
		flag_code	=	CvANN_MLP::NO_INPUT_SCALE;
	}else if (flag_code==3){
		flag_code	=	CvANN_MLP::UPDATE_WEIGHTS + CvANN_MLP::NO_INPUT_SCALE;
	}else if (flag_code==4){
		flag_code	=	CvANN_MLP::NO_OUTPUT_SCALE;
	}else if (flag_code==5){
		flag_code	=	CvANN_MLP::UPDATE_WEIGHTS + CvANN_MLP::NO_OUTPUT_SCALE;
	}else if (flag_code==6){
		flag_code	=	CvANN_MLP::NO_INPUT_SCALE + CvANN_MLP::NO_OUTPUT_SCALE;
	}else if (flag_code==7){
		flag_code	=	CvANN_MLP::UPDATE_WEIGHTS + CvANN_MLP::NO_INPUT_SCALE + CvANN_MLP::NO_OUTPUT_SCALE;
	}else{
		assert(false);
	}
	// set activateFunc
	if(activateFunc==0){
		activateFunc	=	CvANN_MLP::IDENTITY;	// 測試沒成功過
	}else if (activateFunc==1){
		activateFunc	=	CvANN_MLP::SIGMOID_SYM;
	}else if (activateFunc==2){
		activateFunc	=	CvANN_MLP::GAUSSIAN;
	}else{
		assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	// train
	cv::Mat trainData, responses, varIdx, sampleIdx;
	if(cvml::MLData::split(data,trainData, responses, cv::Mat(), cv::Mat(), varIdx)){
		// create
		cv::Mat	layerSizes	=	(cv::Mat_<int>(1,1)<<trainData.cols);
		cv::Mat	outputLayer	=	(cv::Mat_<int>(1,1)<<1);
		layerSizes.push_back(hiddenLayerSize);
		layerSizes.push_back(outputLayer);

		//////////////////////////////////////////////////////////////////////////


		ann.create(layerSizes,activateFunc,0.,0.);
		// train
		if(ann.train(trainData,responses,cv::Mat(),cv::Mat(),params,flag_code)){
			return true;
		}else{
			return false;
		}
	}else{
		return false;
	}

	return true;
}
//////////////////////////////////////////////////////////////////////////
bool	cvml::SVM::setParams(CvSVMParams &params,const int svm_type,const int kernel_type, CvMat* class_weights, const int max_iter_for_convergence){

	// set svm_type
	if(svm_type==100){
		params.svm_type	=	CvSVM::C_SVC;	// C-Support Vector Classification. n-class classification (n \geq 2),
	}else if(svm_type==101){
		params.svm_type	=	CvSVM::NU_SVC;	// Support Vector Classification. n-class classification with possible imperfect separation.
	}/*else if(svm_type==102){
		params.svm_type	=	CvSVM::ONE_CLASS; // support only one class
	}else if(svm_type==103){
		params.svm_type	=	CvSVM::EPS_SVR;
	}else if(svm_type==104){
		params.svm_type	=	CvSVM::NU_SVR;
	}*/else{
		assert(false);
	}

	// set kernel_type
	if(kernel_type==0){
		params.kernel_type	=	CvSVM::LINEAR;
	}else if(kernel_type==1){
		params.kernel_type	=	CvSVM::POLY;
	}else if(kernel_type==2){
		params.kernel_type	=	CvSVM::RBF;	// Radial basis function (RBF), a good choice in most cases.
	}else if(kernel_type==3){
		if(params.svm_type==CvSVM::C_SVC){	// 配CvSVM::C_SVC無法work
			assert(false);
		}
		params.kernel_type	=	CvSVM::SIGMOID;
	}else{
		assert(false);
	}

	// set class_weights
	if(params.svm_type==CvSVM::C_SVC){
		params.class_weights	=	class_weights;
	}else{
		params.class_weights	=	0;
	}

	// set convergence
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, max_iter_for_convergence, FLT_EPSILON );
	

	// set other parameters (此些設定類似初始設定)
	params.degree	=	0.000001; 
	params.gamma	=	1.;
	params.coef0	=	0.;
	params.C		=	1.;
	params.nu		=	0.000001; 
	params.p		=	0.;

	return true;
}
bool	cvml::SVM::train(CvSVM &svm,CvMLData &data,CvSVMParams &params,const int k_fold){

	cv::Mat trainData, responses, varIdx, sampleIdx;

	if(cvml::MLData::split(data,trainData, responses, cv::Mat(), cv::Mat(), varIdx)){
		if(svm.train_auto(trainData,responses,varIdx,cv::Mat(),params,k_fold)){	// Parameters are considered optimal when the cross-validation estimate of the test set error is minimal.
			return true;
		}else{
			return false;
		}
	}else{
		return false;
	}
	return true;
}
//////////////////////////////////////////////////////////////////////////
bool	cvml::Boost::setParams(CvBoostParams &params,int boost_type, int weak_count, double weight_trim_rate, int max_depth, bool use_surrogates, const float* priors){
	
	// Type of boosting algorithm.
	if(boost_type==0){
		params.boost_type	=	CvBoost::DISCRETE;
	}else if(boost_type==1){
		params.boost_type	=	CvBoost::REAL;
	}else if(boost_type==2){
		params.boost_type	=	CvBoost::LOGIT;
	}else if(boost_type==3){
		params.boost_type	=	CvBoost::GENTLE;
	}else{
		assert(false);
	}

	// The number of weak classifiers.
	assert(weak_count>0);
	params.weak_count	=	weak_count;	

	// A threshold between 0 and 1 used to save computational time.
	assert(weight_trim_rate>=0.&&weight_trim_rate<=1.);
	params.weight_trim_rate	=	weight_trim_rate;
	
	// The maximum possible depth of the tree. That is the training algorithms attempts to split a node while its depth is less than max_depth.
	assert(max_depth>0);
	params.max_depth	=	max_depth;

	// If true then surrogate splits will be built. These splits allow to work with missing data and compute variable importance correctly.
	params.use_surrogates	=	use_surrogates;

	// The array of a priori class probabilities, sorted by the class label value. The parameter can be used to tune the decision tree preferences toward a certain class.
	params.priors	=	priors;

	return true;
}
bool	cvml::Boost::train(CvBoost &boost,CvMLData &data,CvBoostParams &params,bool update){

	cv::Mat trainData, responses, varIdx, varType, missingDataMask;

	if(cvml::MLData::split(data,trainData, responses, cv::Mat(), cv::Mat(), varIdx, cv::Mat(), varType, missingDataMask)){
		if(boost.train(trainData,CV_ROW_SAMPLE,responses,varIdx,cv::Mat(),varType,missingDataMask,params,update)){	
			return true;
		}else{
			return false;
		}
	}else{
		return false;
	}
	return true;
}
void	cvml_Boost_cascade_train_getBestThreshold(cv::Mat &outputs,cv::Mat &responses,float *best_buffer,const int &nPositives,const int &nNegatives,const int nSamples,const float sc_tpr,const float sc_fpr,const bool showInfo=true){

	//////////////////////////////////////////////////////////////////////////
	///// exception
	if(responses.type()!=CV_8U){
		std::cout	<<	"[cvml_Boost_cascade_train_getBestThreshold] responses.type() error."	<<	std::endl;
	}
	if(outputs.type()!=CV_32F){
		std::cout	<<	"[cvml_Boost_cascade_train_getBestThreshold] outputs.type() error."	<<	std::endl;
	}

	//////////////////////////////////////////////////////////////////////////
	/// 進行sorting, 方便以下計算
	cv::Mat	sample_outputsIdx;
	cv::sortIdx(outputs,sample_outputsIdx,CV_SORT_EVERY_COLUMN+CV_SORT_ASCENDING);

	//////////////////////////////////////////////////////////////////////////
	///// 進行評估
	int		tp=nPositives,fp=nNegatives,tn=0,fn=0;	///< used to collect the numbers of tp, fp, tn, fn
	float	curr_buffer[3]={0.,1.,1.};				// [0]: best_threshold, [1]: tpr, [2]: fpr
	float	prior_best_buffer[3]={0.,1.,1.};		// [0]: best_threshold, [1]: tpr, [2]: fpr
	float	prior_buffer[3]={0.,1.,1.};				// [0]: best_threshold, [1]: tpr, [2]: fpr
	int		best_updated_count=0;					// 描述prior_best_buffer被更新之次數
	for(int i=0;i<nSamples;i++){	// # of samples

		// 更新prior_buffer
		for(int j=0;j<3;j++){
			prior_buffer[j]	=	curr_buffer[j];											// curr_buffer -> prior_buffer
		}

		//////////////////////////////////////////////////////////////////////////
		// 取得目前門檻值的成果
		int		temp_index	=	((int*)sample_outputsIdx.data)[i];
		int		true_output	=	(int)responses.data[temp_index];
		curr_buffer[0]		=	((float*)outputs.data)[temp_index];	// current threshold
		if(true_output==0){	// negative, and classify to negative, tn
			tn++;
			fp--;
		}else if(true_output==1){	// positive, and classify to negative, fn
			fn++;
			tp--;
		}else{
			assert(false);
		}
		curr_buffer[1]	=	(float)tp/((float)(tp+fn));	// tpr
		curr_buffer[2]	=	(float)fp/((float)(fp+tn));	// fpr

		//////////////////////////////////////////////////////////////////////////
		// 更新best_buffer
		if(curr_buffer[0]!=prior_buffer[0]){	// 如果門檻值改變, 則需要更動best_threshold and related performances

			// 更新prior_best_buffer
			for(int j=0;j<3;j++){
				prior_best_buffer[j]	=	best_buffer[j];
				best_buffer[j]			=	prior_buffer[j];				
			}			
			best_buffer[0]	=	(prior_buffer[0]+curr_buffer[0])/2.;	// 更改為平均門檻值
			if(i!=0){	// 第一個iteration顯示沒有意義, 因為其為初始數值
				if(showInfo){
					std::cout	<<	"\t\t\tthres:"<<best_buffer[0]<<"\ttpr/fpr: "<<best_buffer[1]<<"/"<<best_buffer[2]<<std::endl;
				}
				best_updated_count++;	// 第一輪的計算無意義, 故放於此if中
			}

			// 判斷是否結束訓練
			if(best_buffer[1]>=sc_tpr&&best_buffer[2]<sc_fpr){	// 理想的tpr and fpr條件
				break;
			}else if(best_buffer[1]<sc_tpr){	// 正好目前的門檻值使得tpr低於門檻, 則前面一個為最佳值
				if(best_updated_count!=1){	// 若非第一次, 使用prior_best_buffer 
					for(int j=0;j<3;j++){
						best_buffer[j]	=	prior_best_buffer[j];
					}
				}else{	// 若是第一次, 因為prior_best_buffer沒有最佳值, 則可直接使用prior_buffer即可. 
					// do nothing
				}
				break;
			}else{
				// normal case
			}
		}else{		// 如果門檻值與之前相同.
			// normal case
		}

	}	 // end to # of samples
	
}
bool	cvml::Boost::cascade_train(CvMLData &data,CvBoostParams &param,float sc_tpr,float sc_fpr,const std::string *label_names,const int nStrongClassifiers,const int upperlimitNWeakCount,const bool update,const short debugmode){
	

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(sc_tpr<0.||sc_tpr>1.||sc_fpr<0.||sc_fpr>1.){
		std::cout	<<	"[cvml::Boost::trainCascade] error."	<<	std::endl;
		assert(false);
	}


	//////////////////////////////////////////////////////////////////////////
	///// 預備初始資料集合, outputs are : [train_set] and [train_reference]
	cv::Mat train_set, train_reference;
	if(!cvml::MLData::split(data,train_set, train_reference)){
		std::cout	<<	"[cvml::Boost::trainCascade] cvml::MLData::split error."	<<	std::endl;
		return false;
	}
	const	int	nFeatures	=	train_set.cols;	//	number of features
	// 整理references, 將positive改為1, negative改為0, 如此即可便於預測
	std::map<std::string,int>	class_map	=	data.get_class_labels_map();
	train_reference	=	cvml::MLData::get_reference(train_reference,class_map,label_names[0],label_names[1]);
	if(train_reference.empty()||train_reference.type()!=CV_8U){
		std::cout	<<	"[cvml::Boost::cascade_train] train_reference is empty or it is not CV_8U."	<<	std::endl;
		assert(false);
	}


	//////////////////////////////////////////////////////////////////////////
	///// create temp dictionary
	system("rd /S/Q cascade_temp");
	system("mkdir cascade_temp");
	
	
	//////////////////////////////////////////////////////////////////////////
	///// initialization, outputs are [target_tpr] and [target_fpr], 整體系統的目標值
	const	float		TargetTPR				=	cv::pow((float)sc_tpr,(float)nStrongClassifiers);	
	const	float		TargetFPR				=	cv::pow((float)sc_fpr,(float)nStrongClassifiers);
	/// 儲存best_threshold
	FILE	*fn	=	fopen("cascade_temp/threshold.txt","w");
	fclose(fn);
	
	
	//////////////////////////////////////////////////////////////////////////
	///// 設定參數
	CvBoostParams	temp_param	=	param;
	///// training
	float			curr_tpr=1.,curr_fpr=1.;	///< 目前整體系統的tpr and fpr
	cv::Mat			negative_and_true_mask;	// 用於樣本集合過濾使用.
	int				nStages=0;	// 強分類器數量
	while(true){	///< 只要有一個條件沒有達到則繼續訓練
		nStages++;


		//////////////////////////////////////////////////////////////////////////
		///// 調整樣本集合, 僅扣除negative_and_true_mask為true的那些. 剩下的僅為all positive and hard samples
		if(!negative_and_true_mask.empty()){
			cv::Mat	temp_trainDate,temp_responses;
			for(int i=0;i<train_set.rows;i++){	// # of samples				
				if(negative_and_true_mask.data[i]==false){	// false, 則留下, 故僅僅true negative從此樣本集合中移除
					temp_trainDate.push_back(train_set.row(i));
					temp_responses.push_back(train_reference.row(i));
				}
			}
			train_set.release();
			train_reference.release();
			train_set	=	temp_trainDate.clone();
			train_reference	=	temp_responses.clone();
		}
		negative_and_true_mask.release();
		const	int	nSamples	=	train_set.rows;	//	number of samples
		/// get the numbers of positive and negative
		int	nPositives=0, nNegatives=0;
		for(int i=0;i<train_reference.rows;i++){	// # of samples
			if(train_reference.data[i]==1){	// positive
				nPositives++;
			}else if(train_reference.data[i]==0){	// negative
				nNegatives++;
			}else{
				assert(false);
			}
		}
		if(debugmode>=1){
			std::cout	<<	"Strong #"	<<	nStages	<<	" with ["<<	nSamples	<<	"] samples (pos/neg): ("<<nPositives<<"/"<<nNegatives<<")."<<	std::endl;
		}		
		
		
		//////////////////////////////////////////////////////////////////////////
		/// 逐步增加弱分類器數量進行訓練
		CvBoost	curr_strongBoost;
		double	local_tpr=0.,local_fpr=0.;
		for(int nWeak=1;nWeak<=upperlimitNWeakCount;nWeak++){
			
			/// 設定參數, 僅更改弱分類器數量
			temp_param.weak_count	=	nWeak;

			//////////////////////////////////////////////////////////////////////////
			/// 訓練
			cv::Mat	f_train_response;	// 由於train僅收CV_32F, 故這裡進行此轉換
			train_reference.convertTo(f_train_response,CV_32F);
			if(!curr_strongBoost.train(train_set,CV_ROW_SAMPLE,f_train_response,cv::Mat(),cv::Mat(),cv::Mat(),cv::Mat(),temp_param,update)){		// 用於不同type的feature時候, 若是長時間沒有錯誤, 即可將那些vartype等取消以不使用, 另外, update用途待確認.
				std::cout	<<	"[cvml::Boost::trainCascade] boost.train error."	<<	std::endl;
				assert(false);
			}
			f_train_response.release();


			//////////////////////////////////////////////////////////////////////////
			///// 評估看看是否還需要增加弱分類器數量
			// 取得所有樣本的confidences, output: [sample_outputs]
			cv::Mat	sample_outputs(cv::Size(1,nSamples),CV_32F),sample_outputsIdx;
			cv::Mat	temp_featurevector(cv::Size(nFeatures,1),train_set.type());	// temp feature vector
			for(int i=0;i<nSamples;i++){	// # of samples
				/// copy feature of one sample				
				temp_featurevector.setTo(0);
				for(int j=0;j<nFeatures;j++){	// # of features
					temp_featurevector.at<float>(0,j)	=	train_set.at<float>(i,j);
				}
				/// prediction
				((float*)sample_outputs.data)[i]	=	(float)curr_strongBoost.predict((const cv::Mat)temp_featurevector,cv::Mat(),cv::Range::all(),false,true);
			}
			
			
			// 取得可達到需求的最佳門檻值, output: [best_buffer]
			float	best_buffer[3]={0.,1.,1.};			// [0]: best_threshold, [1]: tpr, [2]: fpr
			cvml_Boost_cascade_train_getBestThreshold(sample_outputs,train_reference,&best_buffer[0],nPositives,nNegatives,nSamples,sc_tpr,sc_fpr,(debugmode>=2?true:false));
	

			//////////////////////////////////////////////////////////////////////////
			/// do classification and evaluation to each sample, [output] negative_and_true_mask and local_tpr and local_fpr
			negative_and_true_mask.create(cv::Size(1,nSamples),CV_8U);
			negative_and_true_mask.setTo(false);
			int	tp=0;
			int	fp=0;
			int	tn=0;
			int	fn=0;
			for(int i=0;i<nSamples;i++){	// # of samples

				/// evaluation, by using the label name not just the index
				int	true_output	=	(int)train_reference.data[i];
				if(true_output==1){	// positive
					if(((float*)sample_outputs.data)[i]>best_buffer[0]){	// positive
						tp++;
						negative_and_true_mask.data[i]	=	false;
					}else if(((float*)sample_outputs.data)[i]<=best_buffer[0]){		// negative
						fn++;
						negative_and_true_mask.data[i]	=	false;
					}else{
						assert(false);
					}
				}else if(true_output==0){	// negative
					if(((float*)sample_outputs.data)[i]<=best_buffer[0]){		// negative
						tn++;
						negative_and_true_mask.data[i]	=	true;
					}else if(((float*)sample_outputs.data)[i]>best_buffer[0]){	// positive
						fp++;
						negative_and_true_mask.data[i]	=	false;
					}else{
						assert(false);
					}			
				}else{
					std::cout	<<	"[cvml::Boost::trainCascade] cvml::classindextostring error."	<<	std::endl;
					assert(false);
				}
			} // end to # of samples
			assert((tp+fp+tn+fn)==nSamples);
			/// calculate tpr and fpr
			local_tpr	=	(float)tp/((float)(tp+fn));
			local_fpr	=	(float)fp/((float)(fp+tn));
			if(debugmode>=1){
				std::cout	<<	"\t\tStrongs #"	<<nStages<<	"\tWeaks #"	<<	nWeak	<<	"\t(tpr/fpr): ("<<(float)local_tpr*100.<<"/"<<(float)local_fpr*100.<<")"<<std::endl;
			}
			assert(local_tpr==best_buffer[1]&&local_fpr==best_buffer[2]);	// 用於debug, 應該要相同. 


			//////////////////////////////////////////////////////////////////////////
			///// 判斷是否還需要增加弱分類器
			if((local_tpr>sc_tpr&&local_fpr<sc_fpr)||(nWeak==upperlimitNWeakCount)){	// 有兩個條件會跳出訓練, 一個為符合條件, 第二為目前為最後一個弱分類器, 最後一個不論如何直接輸出
				/// 儲存best_threshold
				FILE	*fn	=	fopen("cascade_temp/threshold.txt","a");
				fprintf(fn,"%lf\n",best_buffer[0]);
				fclose(fn);	
				break;	// 達到sc_tpr and sc_fpr的標準即停止此strong classifier的訓練
			}else{
				curr_strongBoost.clear();	// 反之清除目前的進行重新訓練
			}

		}	// end of nWeak
	

		//////////////////////////////////////////////////////////////////////////
		///// save cascade
		char	file_name[200];
		sprintf(file_name,"cascade_temp/cascade_boost%.3d.xml",nStages);
		curr_strongBoost.save(file_name);
		

		//////////////////////////////////////////////////////////////////////////
		///// 計算目前的tpr and fpr
		curr_tpr	*=	local_tpr;
		curr_fpr	*=	local_fpr;
		if(debugmode>=1){
			std::cout	<<	"\tCurrent entire (tpr/fpr): ("<<(float)curr_tpr*100.<<"/"<<(float)curr_fpr*100.<<")"<<std::endl;
		}
		

		//////////////////////////////////////////////////////////////////////////
		///// 判斷是否需要進行下一個strong classifier的訓練
		if(curr_tpr>=TargetTPR&&curr_fpr<=TargetFPR){
			break;
		}else{
			if(nStages==nStrongClassifiers){	///< the last round
				break;
			}
		}

	}	// end of training of strong classifier


	//////////////////////////////////////////////////////////////////////////
	if(debugmode>=1){
		std::cout	<<	"Training complete."	<<	std::endl;
	}	
	return true;
}
double	cvml::Boost::cascade_predict(const std::vector<CvBoost> &boost_vec,const std::vector<float> &boost_thres,cv::Mat &featurevector){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(featurevector.rows!=1){
		std::cout	<<	"[cvml::Boost::cascade_predict] featurevector.rows error."	<<	std::endl;
		return false;
	}
	
	//////////////////////////////////////////////////////////////////////////
	/// prediction, a cascade structure
	for(int nStages=1;nStages<=boost_vec.size();nStages++){

		const	CvBoost	&curr_boost	=	boost_vec[nStages-1]; 
		const	float	&curr_thres	=	boost_thres[nStages-1];

		// get confidence
		float	confidence	=	(float)curr_boost.predict((const cv::Mat)featurevector,cv::Mat(),cv::Range::all(),false,true);

		// classification
		if(confidence<=curr_thres){	// negative. NOTE: should involve the "=" for negatives
			return	confidence-curr_thres;	// exclude it, and the confidence should be confidence-curr_thres since lower than it is negative
		}else{	// positive
			if(nStages==boost_vec.size()){	// whether it is the last stage
				return	confidence-curr_thres;	// same as above
			}else{	
				// go to the next stage
			}
		}
	}
	
	std::cout	<<	"[cvml::Boost::cascade_predict] boost_vec is empty."	<<	std::endl;
	assert(false);
	return 0.;
}
bool	cvml::Boost::cascade_load(std::vector<CvBoost> &boost_vec,std::vector<float> &boost_thres,const std::string pth){
	
	/// initialization
	boost_vec.clear();
	boost_thres.clear();

	/// load thresholds
	std::string	path_threshold	=	pth	+	"/threshold.txt";
	CvMLData	data;
	if(data.read_csv(path_threshold.c_str())==-1){	///< 表示未讀取到檔案, 反之讀取到為0
		std::cout	<<	"[cvml::Boost::cascade_load] threshold file: "<<	path_threshold.c_str()<<	" load error."	<<	std::endl;
		return false;
	}
	cv::Mat	thresholds	=	data.get_values();
	for(int i=0;i<thresholds.rows;i++){
		boost_thres.push_back(((float*)thresholds.data)[i]);
	}

	/// load cascades
	char	path_cascade[200];
	boost_vec.resize(thresholds.rows);	// 直接定義size
	for(int i=1;i<=thresholds.rows;i++){
		// get name
		sprintf(path_cascade,"%s%.3d.xml",((std::string)pth	+	"/cascade_boost").c_str(),i);
		std::cout	<<	"Load "<< path_cascade	<<	std::endl;
		boost_vec[i-1].load(path_cascade);	// 若是沒有讀取到, 此會直接crash, 故若是可以run, 則表示正常. 
	}

	return true;
}