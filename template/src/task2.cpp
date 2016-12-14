#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "io.h"
#include <cmath>

#include <string>
using std::string;

using std::tuple;
using std::make_tuple;
using std::tie;

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

int root[2][8];
int sobelX[3][3];
int sobelY[3][3];
Image load_image(BMP * input)
{
    BMP in = *input;
    Image res(in.TellHeight(), in.TellWidth());

    for (uint i = 0; i < res.n_rows; ++i) {
        for (uint j = 0; j < res.n_cols; ++j) {
            RGBApixel *p = in(j, i);
            res(i, j) = make_tuple(p->Red, p->Green, p->Blue);
        }
    }

    return res;
}

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<double>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}
double check_pi(double val) {
    if(val > 255) 
        return 255;
    if(val < 0)
        return 0;
    return val;
}
Image grayscale(Image src_image)
{
    int n = src_image.n_rows;
    int m = src_image.n_cols;
    Image tmp(n, m);
    tmp = src_image.deep_copy();
    
    for(int i = 0; i < n; i++) {

        for(int j = 0; j < m; j++) {
            double value = 0.299 * std::get<0>(src_image(i, j)) + 0.587 * std::get<1>(src_image(i, j)) + 0.114 * std::get<2>(src_image(i, j));
            tmp(i, j) = std::make_tuple((value),(value),(value));
        }
    }
    return tmp;
}


Image sobel_x(Image src_image)
{
    int n = src_image.n_rows;
    int m = src_image.n_cols;
    Image tmp(n, m);
    tmp = src_image.deep_copy();
    for(int i = 1; i < n - 1; i++) {
        for(int j = 1; j < m - 1; j++) {
            double value[3];
            value[0] = 0;
            value[1] = 0;
            value[2] = 0;
           
            for(int k = 0; k < 8; k++) {
                value[0] += std::get<0>(src_image(i + root[0][k], j + root[1][k])) * sobelX[1 + root[0][k]][1 + root[1][k]];
            }
            for(int k = 0; k < 8; k++) {
                value[1] += std::get<1>(src_image(i + root[0][k], j + root[1][k])) * sobelX[1 + root[0][k]][1 + root[1][k]];
            }
            for(int k = 0; k < 8; k++) {
                value[2] += std::get<2>(src_image(i + root[0][k], j + root[1][k])) * sobelX[1 + root[0][k]][1 + root[1][k]];
            }
            tmp(i, j) = std::make_tuple((value[0]),(value[1]),(value[2]));
        }
    }
    return tmp;
}
Image sobel_y(Image src_image)
{
    int n = src_image.n_rows;
    int m = src_image.n_cols;
    Image tmp(n, m);
    tmp = src_image.deep_copy();
    for(int i = 1; i < n - 1; i++) {
        for(int j = 1; j < m - 1; j++) {
            double value[3];
            value[0] = 0;
            value[1] = 0;
            value[2] = 0;
           
            for(int k = 0; k < 8; k++) {
                value[0] += std::get<0>(src_image(i + root[0][k], j + root[1][k])) * sobelY[1 + root[0][k]][1 + root[1][k]];
            }
            for(int k = 0; k < 8; k++) {
                value[1] += std::get<1>(src_image(i + root[0][k], j + root[1][k])) * sobelY[1 + root[0][k]][1 + root[1][k]];
            }
            for(int k = 0; k < 8; k++) {
                value[2] += std::get<2>(src_image(i + root[0][k], j + root[1][k])) * sobelY[1 + root[0][k]][1 + root[1][k]];
            }
            tmp(i, j) = std::make_tuple((value[0]),(value[1]),(value[2]));
        }
    }
    return tmp;
}
double** module(Image src_x, Image src_y)
{
    int n = src_x.n_rows;
    int m = src_x.n_cols;
    double** tmp = (double **)malloc(sizeof(double * ) * n);
    for(int i = 0; i < n; i++) {
        tmp[i] =  (double *)malloc(sizeof(double) * m);
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            double value = sqrt(pow(std::get<0>(src_x(i, j)), 2) + pow(std::get<0>(src_y(i, j)), 2));
           // cout << value;
            tmp[i][j] = value;
        }
    }
  
    return tmp;
}
int** direction(Image src_x, Image src_y)
{
    int n = src_x.n_rows;
    int m = src_x.n_cols;

    int** tmp = (int **)malloc(sizeof(int *) * n);
    for(int i = 0; i < n; i++) {
        tmp[i] = (int *) malloc(sizeof(int) * m);
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            int value = (M_PI + atan2(std::get<0>(src_y(i, j)), std::get<0>(src_x(i, j)))) * 4/ M_PI;
            tmp[i][j] = value;
           // cout << value << ' ';
        }
    }
    return tmp;
}
vector<double> get_f(int n, double **src_m, int ** src_d, int n_rows, int n_cols)
{
    vector<double> data[10][10];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            data[i][j] = vector<double>(8, 0);
        }
    }
    for(int t = 0; t < n_rows; t++) {
        for(int k = 0; k < n_cols; k++) {
            //cout << src_m[t][k] << ' ';
            (data[t * n/n_rows][k * n/n_cols])[src_d[t][k]] += src_m[t][k];
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            long double val = 0;
            for(int t = 0; t < 8; t++) {
                val += (data[i][j])[t] * (data[i][j])[t];
            }
            val = sqrt(val);
            if(val > 0.007)
            for(int t = 0; t < 8; t++) {
                (data[i][j])[t] /= val; 
            }
        }
    }
    vector<double>one_image_feature;
     for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < 8; t++) {
                
                one_image_feature.push_back(data[i][j][t]);
            }
        }
    }
    return one_image_feature;

}
vector<double> get_f2(int n, Image src_image, int n_rows, int n_cols)
{
    int values[n_rows][n_cols];
    vector<int> data[10][10];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            data[i][j] = vector<int>(256, 0);
        }
    }
    for(int t = 1; t < n_rows - 1; t++) {
        for(int k = 1; k < n_cols - 1; k++) {
            vector<int> r = vector<int>(8, 0);
            for(int i = 0; i < 8; i++) {
                if(src_image(t + root[0][i], k + root[1][i]) >  src_image(t, k) ) {
                    r[i] = 1;
                }
            }
            int val = 0;
            long int pos = 1;
            for(int i = 0; i < 8; i++) {
                val += r[i] * pos;
                pos *= 2;
            }
            data[n* t/n_rows][n *k/n_cols][val]++;
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            long double val = 0;
            for(int t = 0; t < 8; t++) {
                val += (data[i][j])[t] * (data[i][j])[t];
            }
            val = sqrt(val);
            if(val > 0.007)
            for(int t = 0; t < 8; t++) {
                (data[i][j])[t] /= val; 
            }
        }
    }
    vector<double>one_image_feature;
     for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < 8; t++) {
               
                one_image_feature.push_back(data[i][j][t]);
            }
        }
    }
    return one_image_feature;

}
vector<double> get_f3(int n, Image src_image)
{
    vector<int> data[8][8];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            data[i][j] = vector<int>(4, 0);
        }
    }
    for(int t = 0; t < src_image.n_rows; t++) {
        for(int k = 0; k < src_image.n_cols; k++) {
            data[n* t/src_image.n_rows][n *k/src_image.n_cols][0] += std::get<0>(src_image(t, k));
            data[n* t/src_image.n_rows][n *k/src_image.n_cols][1] += std::get<1>(src_image(t, k));
            data[n* t/src_image.n_rows][n *k/src_image.n_cols][2] += std::get<2>(src_image(t, k));
            data[n* t/src_image.n_rows][n *k/src_image.n_cols][3]++;
        }
    }
    
    vector<double>one_image_feature;
     for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < 3; t++) {
                one_image_feature.push_back((1.0 * data[i][j][t]/data[i][j][3])/255);
            }
        }
    }
    return one_image_feature;

}
// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        Image src1 =  load_image(data_set[image_idx].first);
        //3.1 grayscale
        //cout << src.n_rows << ' ' << src.n_cols << endl;
        Image src = grayscale(src1);
        
        Image src_x = sobel_x(src);
        Image src_y = sobel_y(src);
        int n = src_x.n_rows;
        int m = src_x.n_cols;

        double** src_m = module(src_x, src_y);
        int** src_d = direction(src_x, src_y);
        vector<double> data1 = get_f(10, src_m, src_d, n, m);
        vector<double> data2 = get_f2(10, src, n, m);
        vector<double> data3 = get_f3(8, src1);
        // PLACE YOUR CODE HERE
        vector<double> data;
        for(int i = 0; i < data1.size(); i++) {
            data.push_back(data1[i]);
        }
        for(int i = 0; i < data2.size(); i++) {
            data.push_back(data2[i]);
        }
        for(int i = 0; i < data3.size(); i++) {
            data.push_back(data3[i]);
        }

        
        
        features->push_back(make_pair(data, data_set[image_idx].second));
        for(int i = 0; i < n; i++) {
            free(src_m[i]);
            free(src_d[i]);
        }
        free(src_m);
        free(src_d);
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    root[0][0] = -1;
    root[1][0] = -1;

    root[0][1] = -1;
    root[1][1] = 0;
    root[0][2] = -1;
    root[1][2] = 1;

    root[0][3] = 0;
    root[1][3] = -1;

    root[0][7] = 0;
    root[1][7] = +1;

    root[0][4] = +1;
    root[1][4] = +1;
    root[0][5] = +1;
    root[1][5] = 0;
    root[0][6] = +1;
    root[1][6] = -1;

    sobelX[0][0] = -1;
    sobelX[0][1] = 0;
    sobelX[0][2] = 1;

    sobelX[1][0] = -2;
    sobelX[1][1] = 0;
    sobelX[1][2] = 2;

    sobelX[2][0] = -1;
    sobelX[2][1] = 0;
    sobelX[2][2] = 1;

    sobelY[0][0] = -1;
    sobelY[0][1] = -2;
    sobelY[0][2] = -1;

    sobelY[1][0] = 0;
    sobelY[1][1] = 0;
    sobelY[1][2] = 0;

    sobelY[2][0] = 1;
    sobelY[2][1] = 2;
    sobelY[2][2] = 1;


    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}