#include <stdlib.h>
#include <string>
#include <vector>
#include <stdio.h>
#include "mfcc.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>

using namespace std;
mfcc_t mfcc;

int dma_audio_buffer[AUDIO_FRAME_LEN]; //512
int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN*1.5)]; // an easy method for 50% overlapping
int audio_sample_i = 0;
int count_number_of_mfccs = 0;


float mfcc_features_f[MFCC_COEFFS];             // output of mfcc
float mfcc_features[49][MFCC_COEFFS];     // ring buffer
uint32_t mfcc_feat_index = 0;


// Function to read the CSV file and store the data (short) and labels (string)
bool read_CSV(const std::string& filename, std::vector<std::vector<short> >& data, std::vector<std::string>& labels) {
    // Open the CSV file
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<short> row;
        std::string value_str;
        if (count % 1000 == 0)
            printf("read count: %d\n", count);
        count++;
        // Read the first 16000 values as short data
        for (int i = 0; i < 16000; ++i) {
            
            if (std::getline(ss, value_str, ',')) {
                try {
                    short value = std::stoi(value_str);  // Convert string to short
                    row.push_back(value);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid data at row " << data.size() + 1 << ": " << value_str << std::endl;
                    return false;
                }
            }

        }
        

        // Read the label (the last value in the line, a string)
        std::string label;
        if (std::getline(ss, label, ',')) {
            // Add the data row to the 2D vector
            data.push_back(row);
            
            // Add the label to the labels vector
            labels.push_back(label);
        }
        //if (count == 1000)
        //    break;
    }

    // Close the file
    file.close();

    return true;  // Successfully read the file
}
std::vector<float> process_mfcc_audio_data(const std::vector<short>& sig, int rate) {
    int p, i, j;
    audio_sample_i = 0;  // Initialize audio sample index (make sure it's declared)
    int count = 0;  

    // Ensure that dma_audio_buffer and thread_kws_serv() are defined
    while (1) {
        // Read audio samples from file

        for ( p = audio_sample_i, j = 0; j < 480; audio_buffer_16bit[j++] = sig[p++]);
        //----
        audio_sample_i+=320;
        count++;


        for (int k = 480; k < 512; k++)
            audio_buffer_16bit[k] = 0;

        //cout << "audio_buffer_16bit: " << endl;
        //// debugging 
        //for (int k = 0; k < 512; k++) {
        //    cout << audio_buffer_16bit[k] << " ";
        //}
        //cout << endl;


        mfcc_compute(&mfcc, audio_buffer_16bit, mfcc_features_f);
        int8_t mfcc_features_f_int8[MFCC_COEFFS];

        quantize_data(mfcc_features_f, mfcc_features_f_int8, MFCC_COEFFS, 0);
        for (int i = 0; i < MFCC_COEFFS; ++i) {
            mfcc_features[mfcc_feat_index][i] = mfcc_features_f_int8[i];
        }

        //memcpy(mfcc_features[mfcc_feat_index], mfcc_features_f, sizeof(mfcc_features_f));

        mfcc_feat_index++;
        if (count == 49) {
            // Reset audio sample index and frame size
            audio_sample_i = 0;
            count = 0;
            mfcc_feat_index = 0;

            // Flatten the 2D array into a 1D vector
            std::vector<float> ret;
            ret.reserve(49 * MFCC_COEFFS);
            for ( i = 0; i < 49; ++i) {
                for ( j = 0; j < MFCC_COEFFS; ++j) {
                    ret.push_back(mfcc_features[i][j]);
                }
            }



            //// debug print
            //cout << "mfcc_features: " << endl;
            //for (int i = 0; i < 49; ++i) {
            //    for (int j = 0; j < MFCC_COEFFS; ++j) {
            //        cout << mfcc_features[i][j] << " ";
            //    }
            //    cout << endl;
            //}
            //cout << endl;
            return ret;
        }
    }
}




// Function to generate MFCC features
std::vector<float> generate_mfcc_fix(const std::vector<short>& sig, int rate, int sig_len, 
                                     const std::vector<std::vector<short> >* noise_samples = nullptr, 
                                     float noise_weight = 0.1f, float winlen = 0.032f, 
                                     float winstep = 0.032f/2, int numcep = 13, int nfilt = 26, 
                                     int nfft = 512, int lowfreq = 20, int highfreq = 4000, 
                                     float ceplifter = 0.0f, float preemph = 0.97f) {
    std::vector<short> processed_sig = sig;

    // Handle padding or cropping
    if (processed_sig.size() < sig_len) {
        processed_sig.resize(sig_len, 0); // Pad with zeros
    } else if (processed_sig.size() > sig_len) {
        printf("Cropped audio to %d samples\n", sig_len);
        processed_sig.resize(sig_len); // Crop to the desired length
    }

    // Apply noise if provided
    if (noise_samples != nullptr && !noise_samples->empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> noise_dist(0, noise_samples->size() - 1);
        std::uniform_int_distribution<> start_dist(0, noise_samples->at(0).size() - sig_len);

        int noise_index = noise_dist(gen);
        int start = start_dist(gen);

        const auto& selected_noise = noise_samples->at(noise_index);
        std::vector<short> noise_segment(selected_noise.begin() + start, selected_noise.begin() + start + sig_len);

        for (size_t i = 0; i < processed_sig.size(); ++i) {
            processed_sig[i] = static_cast<short>(processed_sig[i] * (1 - noise_weight) + noise_segment[i] * noise_weight);
        }
    }

    // Compute MFCC features
    std::vector<float> mfcc_features_flaten = process_mfcc_audio_data(processed_sig, rate);

    // Debug print

    ++count_number_of_mfccs;
    //std::cout << count_number_of_mfccs << std::endl;

    return mfcc_features_flaten;
}

int main() {
    mfcc_create(&mfcc, MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, MFCC_TOTAL_NUM_BANK, 480, 0.97f, true);
    // 2D vector to store the data field
    std::vector<std::vector<short> > data_2d;
    // 1D vector to store the labels
    std::vector<std::vector<float> > final_data;
    std::vector<std::string> labels;

    // Read the CSV file
    //read_CSV("/Users/youssef/Desktop/transformed_data.csv", data_2d, labels);
    read_CSV("/Users/youssef/Desktop/organic_aug_data.csv", data_2d, labels);
    //read_CSV("/Users/youssef/Desktop/transformed_raw_data.csv", data_2d, labels);
    //read_CSV("/Users/youssef/Documents/Work/Efabless/Hey_kira_training/audio_data.csv", data_2d, labels);
    //read_CSV("/Users/youssef/Documents/Work/Efabless/Hey_kira_training/hey_kira_test.csv", data_2d, labels);
    

    std::cout << "Data size: " << data_2d.size() << "\n";
    vector<float>  data;
    // Process the signal to generate MFCC
    
    printf("-----------------------------\n");
    printf("Start processing data\n");
    printf("-----------------------------\n");
    for (size_t i = 0; i < data_2d.size(); ++i) {
        if (i % 1000 == 0)
            printf("processed count: %d\n", i);
        vector <short> sig = data_2d[i];
        data = generate_mfcc_fix(sig, 16000, 16000);
        final_data.push_back(data);
    }

    printf("-----------------------------\n");
    printf("End processing data\n");
    printf("-----------------------------\n\n\n");


    printf("-----------------------------\n");
    printf("Start writing data\n");
    printf("-----------------------------\n");

    // Write the processed data to an output CSV file
    //string output_file_path = "/Users/youssef/Desktop/processed_raw_data.csv";
    //string output_file_path = "/Users/youssef/Desktop/processed_data.csv";
    string output_file_path = "/Users/youssef/Desktop/processed_organic_data.csv";
    std::ofstream output_file(output_file_path);

    if (output_file.is_open()) {

        // Write the header
        for (int i = 1; i <= 49 * MFCC_COEFFS; ++i) {
            output_file << "MFCC" << i;
            output_file << ",";
            
        }
        output_file << "Label\n";

        for (size_t i = 0; i < data_2d.size(); ++i) {
            const auto& row = final_data[i];
            for (size_t j = 0; j < row.size(); ++j) {
                output_file << row[j];
                if (j != row.size() - 1) {
                    output_file << ",";
                }
            }
            output_file << "," << labels[i] << "\n";
        }
        output_file.close();
        std::cout << "Processed data written to file: " << output_file_path << std::endl;
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    printf("-----------------------------\n");
    printf("End writing data\n");
    printf("-----------------------------\n\n\n");
    return 0;
}
