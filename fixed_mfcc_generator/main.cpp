
#include <stdlib.h>
#include <string>
#include <vector>
#include <stdio.h>
#include "mfcc.h"
#include <filesystem>
#include <sndfile.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>

using namespace std;
namespace fs = std::filesystem;
mfcc_t mfcc;
#define AUDIO_FRAME_LEN (512) //31.25ms * 16000hz = 512, // FFT (windows size must be 2 power n)
int dma_audio_buffer[AUDIO_FRAME_LEN]; //512
int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN*1.5)]; // an easy method for 50% overlapping
int audio_sample_i = 0;
int count_number_of_mfccs = 0;

//the mfcc feature for kws
#define MFCC_LEN            (62)
#define MFCC_COEFFS_FIRST   (1)     // ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN     (13)    // the total coefficient to calculate
#define MFCC_TOTAL_NUM_BANK (26)    // total number of filter bands
#define MFCC_COEFFS         (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)


#define MFCC_FEAT_SIZE  (MFCC_LEN * MFCC_COEFFS)
float mfcc_features_f[MFCC_COEFFS];             // output of mfcc
float mfcc_features[MFCC_LEN][MFCC_COEFFS];     // ring buffer
uint32_t mfcc_feat_index = 0;


void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit);
void thread_kws_serv(void);
void process_audio();
vector<vector<short>> load_noise(const string& path = "dat/_background_noise_/") {
    std::vector<std::vector<short>> noise;
    namespace fs = std::filesystem;

    for (const auto& entry : fs::directory_iterator(path)) {
        std::string filename = entry.path().string();
        if (filename.find(".wav") == std::string::npos) {
            continue;
        }

        SF_INFO sfInfo;
        SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue;
        }

        std::vector<short> buffer(sfInfo.frames * sfInfo.channels);
        sf_readf_short(file, buffer.data(), sfInfo.frames);
        sf_close(file);

        noise.push_back(std::move(buffer));
    }
    cout << "Loaded " << noise.size() << " noise files." << endl;
    return noise;
}

vector<string> read_file_lines(const string& filepath) {
    vector<string> lines;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return lines;
    }

    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}

void reset_directory(const std::string& output_base_path) {
    try {
        // Check if the directory exists
        if (fs::exists(output_base_path)) {
            // Remove the directory and its contents
            fs::remove_all(output_base_path);
        }
        
        // Create the directory
        fs::create_directory(output_base_path);
        std::cout << "Directory has been reset and created: " << output_base_path << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

std::vector<float> process_mfcc_audio_data(const std::vector<short>& sig, int rate) {
    int p, i, j;
    int frame_size = 512;
    audio_sample_i = 0;  // Initialize audio sample index (make sure it's declared)

    // Ensure that dma_audio_buffer and thread_kws_serv() are defined
    while (1) {
        // Read audio samples from file

        for ( p = audio_sample_i, j = 0; j < frame_size; dma_audio_buffer[j++] = sig[p++]);
            
        //printf("audio_sample_i in process_mfcc_audio_data: %d\n", audio_sample_i);
        // Process audio samples and compute MFCC features
        thread_kws_serv();

        // Update audio sample index and adjust frame size for overlap
        audio_sample_i += frame_size;
        if (audio_sample_i == 15872) // 31*512
            frame_size = 128; // 0.25*512 // Adjust frame size for 50% overlap 

        if (audio_sample_i == 16000) {
            // Reset audio sample index and frame size
            audio_sample_i = 0;

            // Flatten the 2D array into a 1D vector
            std::vector<float> ret;
            ret.reserve(MFCC_LEN * MFCC_COEFFS);
            for ( i = 0; i < MFCC_LEN; ++i) {
                for ( j = 0; j < MFCC_COEFFS; ++j) {
                    ret.push_back(mfcc_features[i][j]);
                }
            }
            return ret;
        }
    }
}



// Function to generate MFCC features
std::vector<float> generate_mfcc_fix(const std::vector<short>& sig, int rate, int sig_len, 
                                     const std::vector<std::vector<short>>* noise_samples = nullptr, 
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
// Example function to write float values as int8_t
void write_floats_as_int8(const std::string& output_file_path, const std::vector<float>& data) {
    std::ofstream output_file(output_file_path, std::ios::binary);
    if (output_file.is_open()) {
        for (float value : data) {
            // Convert float to int8_t. You might want to adjust the scaling or clipping.
            int8_t int_value = static_cast<int8_t>(value);  // Simple casting. Consider clamping if needed.

            // Write the int8_t value to the file
            output_file.write(reinterpret_cast<const char*>(&int_value), sizeof(int8_t));
        }
        output_file.close();
    } else {
        std::cerr << "Error creating output file: " << output_file_path << std::endl;
    }
}


void write_int8_as_csv(const std::string& output_file_path, const std::vector<int8_t>& data) {
    std::ofstream output_file(output_file_path);
    if (output_file.is_open()) {
        bool first = true;
        for (int8_t value : data) {
            // Write the int8_t value as text to the file
            if (!first) {
                output_file << ",";  // Add a comma before the value if it's not the first one
            }
            output_file << static_cast<int>(value);  // Write as integer
            
            first = false;
        }
        output_file.close();
    } else {
        std::cerr << "Error creating output file: " << output_file_path << std::endl;
    }
}


void write_float_as_csv(const std::string& output_file_path, const std::vector<float>& data) {
    std::ofstream output_file(output_file_path);
    if (output_file.is_open()) {
        for (size_t i = 0; i < data.size(); ++i) {
            output_file << data[i];
            if (i != data.size() - 1) {
                output_file << ",";  // Add a comma between elements
            }
        }
        output_file.close();  // Close the file after writing
    } else {
        std::cerr << "Error creating output file: " << output_file_path << std::endl;
    }
}


int main (void){

    mfcc_create(&mfcc, MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, MFCC_TOTAL_NUM_BANK, AUDIO_FRAME_LEN, 0.97f, true);
    string input_path = "dat";
    string output_base_path = "speech_commands_V2_mfcc_version/";
    string test_list_path = input_path + "/testing_list.txt";
    string validate_list_path = input_path + "/validation_list.txt";
    vector<vector<float>> train_data;
    vector<vector<float>> test_data;
    vector<vector<float>> validate_data;
    vector<string> train_label;
    vector<string> test_label;
    vector<string> validate_label;

    vector<vector<short>> noise_samples = load_noise("dat/_background_noise_/");

    vector<string> test_list = read_file_lines(test_list_path);
    vector<string> validate_list = read_file_lines(validate_list_path);

    reset_directory(output_base_path);
    string output_directory = output_base_path + '/' + input_path;

    // Create the output directory if it does not exist
    if (!fs::exists(output_directory)) {
        fs::create_directory(output_directory);
    }

    // Copy the test and validate lists to the output directory
    try {
        fs::copy_file(test_list_path, output_directory + "/testing_list.txt", fs::copy_options::overwrite_existing);
        fs::copy_file(validate_list_path, output_directory + "/validation_list.txt", fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error &e) {
        cerr << "Error copying files: " << e.what() << endl;
    }

    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (fs::is_directory(entry.status())) {
            std::string fi_d = entry.path().string(); // Full directory path
            std::string parent_path = entry.path().parent_path().string(); // Parent path
            std::string label = entry.path().filename().string(); // Label (current directory name)
            cout << label << endl;
            if (!fs::exists(output_base_path+fi_d+"/")) {
                fs::create_directory(output_base_path+fi_d+"/");
            }
            // Noise in training
            if (label.find("noise") != std::string::npos) {
                for (const auto& file : fs::directory_iterator(fi_d)) {
                    std::string filename = file.path().filename().string();
                    if (filename.find(".wav") == std::string::npos) {
                        continue;
                    }

                    std::string file_path = file.path().string();
                    SF_INFO sf_info;
                    SNDFILE* file_2 = sf_open(file_path.c_str(), SFM_READ, &sf_info);
                    if (!file_2) {
                        std::cerr << "Error reading WAV file: " << file_path << std::endl;
                        return -1;
                    }

                    // Read the WAV file data
                    std::vector<short> sig(sf_info.frames * sf_info.channels);
                    sf_readf_short(file_2, sig.data(), sf_info.frames);
                    sf_close(file_2);
                    // Process the signal in chunks of 16000 samples
                    size_t chunk_size = 16000;
                    size_t num_chunks = sig.size() / chunk_size;
                    for (size_t i = 0; i < num_chunks; ++i) {
                        // Extract the chunk
                        std::vector<short> chunk(sig.begin() + i * chunk_size, sig.begin() + (i + 1) * chunk_size);
                        
                        // Generate MFCC for the chunk
                        std::vector<float> data = generate_mfcc_fix(chunk, sf_info.samplerate, chunk_size);

                        // Append to training data
                        train_data.push_back(data);
                        train_label.push_back("noise");

                        // Create the output file path for each chunk
                        std::string output_file_path = output_base_path + fi_d + "/" + filename + "_chunk_" + std::to_string(i) + ".txt";
                        write_float_as_csv(output_file_path, data);
                    }
                }
                continue;
            }
            // Dataset
            for (const auto& file : fs::directory_iterator(fi_d)) {
                std::string filename = file.path().filename().string();
                if (filename.find(".wav") == std::string::npos) {
                    continue;
                }

                std::string file_path = file.path().string();
                SF_INFO sf_info;
                SNDFILE* file_2 = sf_open(file_path.c_str(), SFM_READ, &sf_info);
                if (!file_2) {
                    std::cerr << "Error reading WAV file: " << file_path << std::endl;
                    return -1;
                }

                // Read the WAV file data
                std::vector<short> sig(sf_info.frames * sf_info.channels);
                sf_readf_short(file_2, sig.data(), sf_info.frames);
                sf_close(file_2);
                vector<float>  data;
                // Process the signal to generate MFCC
                if (std::find(test_list.begin(), test_list.end(), filename) != test_list.end()) {
                    data = generate_mfcc_fix(sig, sf_info.samplerate, 16000);
                    test_data.push_back(data);
                    test_label.push_back(label);
                    // Create the output file path
                    std::string output_file_path = output_base_path + fi_d + "/" + filename + ".txt";
                    write_float_as_csv(output_file_path, data);
                } else if (std::find(validate_list.begin(), validate_list.end(), filename) != validate_list.end()) {
                    data = generate_mfcc_fix(sig, sf_info.samplerate, 16000);
                    validate_data.push_back(data);
                    validate_label.push_back(label);
                    // Create the output file path
                    std::string output_file_path = output_base_path+fi_d + "/" + filename + ".mfcc";
                    write_float_as_csv(output_file_path, data);
                } else {
                    data = generate_mfcc_fix(sig, sf_info.samplerate, 16000, &noise_samples);
                    train_data.push_back(data);
                    train_label.push_back(label);
                    // Create the output file path
                    std::string output_file_path = output_base_path+fi_d + "/" + filename + ".mfcc";
                    write_float_as_csv(output_file_path, data);
                }
            }
        }
        std::cout << count_number_of_mfccs << std::endl;
    }
    return 0;
}


void thread_kws_serv(void)
{
        //printf("audio_sample_i in thread_kws_serv: %d\n", audio_sample_i);

        if (audio_sample_i == 15872)
            memset(&dma_audio_buffer[128], 0, sizeof(int) * 384); //to fill the latest quarter in the latest frame

        // memory move
        // audio buffer = | 256 byte old data |   256 byte new data 1 | 256 byte new data 2 |
        //                         ^------------------------------------------|
        memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN], (AUDIO_FRAME_LEN/2)*sizeof(int16_t));

        // convert it to 16 bit.
        // volume*4
        for(int i = 0; i < AUDIO_FRAME_LEN; i++)
        {
            audio_buffer_16bit[AUDIO_FRAME_LEN/2+i] = dma_audio_buffer[i];
        }

        // MFCC
        // do the first mfcc with half old data(256) and half new data(256)
        // then do the second mfcc with all new data(512).
        // take mfcc buffer

        for(int i=0; i<2; i++)
        {
            if ((audio_sample_i != 0 || i==1) && (audio_sample_i != 15872 || i==0)) //to skip computing first mfcc block that's half empty
            {
                mfcc_compute(&mfcc, &audio_buffer_16bit[i*AUDIO_FRAME_LEN/2], mfcc_features_f);


                // quantise them using the same scale as training data (in keras), by 2^n.
                //quantize_data(mfcc_features_f, mfcc_features[mfcc_feat_index], MFCC_COEFFS, 3);
                memcpy(mfcc_features[mfcc_feat_index], mfcc_features_f, sizeof(mfcc_features_f));

                // debug only, to print mfcc data on console
                if(0)
                {
                    for(int q=0; q<MFCC_COEFFS; q++)
                        printf("%d ",  mfcc_features[mfcc_feat_index][q]);
                    printf("\n");
                }
                mfcc_feat_index++;
                if(mfcc_feat_index >= MFCC_LEN)
                    mfcc_feat_index = 0;
            }

        }
}

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
    #define _MAX(x, y) (((x) > (y)) ? (x) : (y))
    #define _MIN(x, y) (((x) < (y)) ? (x) : (y))
    float limit = (1 << int_bit);
    float d;
    for(uint32_t i=0; i<size; i++)
    {
        d = round(_MAX(_MIN(din[i], limit), -limit) / limit * 128);
        d = d/128.0f;
        dout[i] = round(d *127);
    }
}
