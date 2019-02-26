#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ATen/ATen.h>
#include <torch/torch.h>

int main(){
 // Deserialize the ScriptModule from a file using torch::jit::load().
 std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../eco_finetune_ucf101.pt"); 
 
 assert(module != nullptr);   
 std::cout << "ok\n";
 double time1 = static_cast<double>( cv::getTickCount());
// Create a vector of inputs.
 const int numSegments = 8;
 std::string videoName = "v_ApplyEyeMakeup_g01_c01.avi";
 int frameNum = 0;
 cv::VideoCapture cap;
 cap.open(videoName);
 cv::Mat frame;

 std::vector<cv::Mat> images;
 while (true)
 {
 	cap >> frame;
 	if (!frame.empty()) {
 		frameNum++;
 		cv::resize(frame, frame, cv::Size(frame.cols*0.5, frame.rows*0.5));
 		images.push_back(frame);
 	}
 	else
 	{
 		break;
 	}
 
 }
 int step = frameNum / numSegments;
 std::vector<torch::jit::IValue> inputs;
 float *test =  new float[8*3*224*224]();
 auto tensor = torch::CPU(torch::kFloat32).tensorFromBlob(test,{8,3,224,224});
 for (int i = 0; i < numSegments; i++)
  {
 	cv::Mat image = images[i*step].clone();
 	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
 	cv::Mat img_float;
 	image.convertTo(img_float, CV_32F, 1);
        cv::resize(img_float, img_float, cv::Size(224, 224));
 	auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, { 224, 224, 3 });
 	img_tensor = img_tensor.permute({ 2,0,1 });
 	img_tensor[0] = img_tensor[0].sub_(128).div_(1);
 	img_tensor[1] = img_tensor[1].sub_(117).div_(1);
 	img_tensor[2] = img_tensor[2].sub_(104).div_(1);
        tensor[i] = img_tensor;
 }
 auto img_var = torch::autograd::make_variable(tensor, false);
 inputs.push_back(img_var);
 std::vector<cv::Mat>().swap(images);

 // Execute the model and turn its output into a tensor.
 module->forward(inputs);
 auto out_tensor = module->forward(inputs).toTensor();

 //std::cout << out_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/109) << '\n';
 //std::cout <<out_tensor<<std::endl;
 std::tuple<torch::Tensor,torch::Tensor> result = out_tensor.sort(-1, true);
 torch::Tensor top_scores = std::get<0>(result)[0];
 torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

  // Load labels
 std::string label_file = "../classInd.txt";
 std::ifstream rf(label_file.c_str());
 CHECK(rf) << "Unable to open labels file " << label_file;
 std::string line;
 std::vector<std::string> labels;
 while (std::getline(rf, line))
     labels.push_back(line);
 auto top_scores_a = top_scores.accessor<float,1>();
 auto top_idxs_a = top_idxs.accessor<int,1>();
 for (int i = 0; i < 5; ++i) {
    int idx = top_idxs_a[i];
    std::cout << "top-" << i+1 << " label: ";
    std::cout  <<idx<<"  "<<labels[idx] << ", score: " << top_scores_a[i] << std::endl;
 }
 double time2 = (static_cast<double>( cv::getTickCount()) - time1)/cv::getTickFrequency();
 std::cout<<"单次处理："<< time2 <<"秒"<<std::endl;//输出运行时间
 return 0;
 }
  
