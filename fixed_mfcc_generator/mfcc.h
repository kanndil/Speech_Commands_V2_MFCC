/*
 * Code adapted from kanndil/mfcc_optimization
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified by Jianjia Ma for C implementation
 *
 * Modified by Youssef Kandil embedded optimization
 */

#ifndef MFCC_H
#define MFCC_H
#include <math.h>

#include <stdint.h>
#include <math.h>
#include <float.h>

#define SAMP_FREQ 16000
#define SAMP_FREQ_INV (1/1600)
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000

#define M_2PI 6.283185307179586476925286766559005
#define FFT_N 512
#define N_WAVE      1024    /* full length of Sinewave[] */
#define LOG2_N_WAVE 10      /* log2(N_WAVE) */


typedef struct _mfcc_t{ // MARK: - this can be static
    int num_mfcc_features;      // this is the number of 
	int num_features_offset;
	int num_fbank;
    int frame_len;
    int frame_len_padded;           // MARK: - this can be static // this can be fixed to make other variables static
	float preempha;
    float frame[512];                  
    float buffer[512];                 
    float mel_energies[26];           
    float window_func[512];                   
    float mel_fbins[28];
    float dct_matrix[338];             
} mfcc_t;

static inline float InverseMelScale(float mel_freq);
static inline float MelScale(float freq);
void rearrange(float data_re[], float data_im[], const unsigned int N);
void compute(float data_re[], float data_im[], const unsigned int N);
void fft(float data_re[], float data_im[], const int N);
void mfcc_create(mfcc_t *mfcc,int num_mfcc_features, int feature_offset, int num_fbank, int frame_len, float preempha, int is_append_energy);
void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc);
void create_mel_fbank(mfcc_t *mfcc);
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out);
void apply_filter_banks(mfcc_t *mfcc);


// hz --> mel
static inline float MelScale(float freq) {
  return 2595.0f * log10(1.0f + freq / 700.0f);
}

// mel --> hz
static inline float InverseMelScale(float mel_freq) {
  return 700.0f * (pow(10,(mel_freq / 2595.0f)) - 1.0f);
}

void rearrange(float data_re[], float data_im[], const unsigned int N)
{
    unsigned int target = 0;
    for (unsigned int position = 0; position < N; position++)
    {
        if (target > position) {
            const float temp_re = data_re[target];
            const float temp_im = data_im[target];
            data_re[target] = data_re[position];
            data_im[target] = data_im[position];
            data_re[position] = temp_re;
            data_im[position] = temp_im;
        }
        unsigned int mask = N;
        while (target & (mask >>= 1))
            target &= ~mask;
        target |= mask;
    }
}

void compute(float data_re[], float data_im[], const unsigned int N)
{
    const float pi = -3.14159265358979323846;
    for (unsigned int step = 1; step < N; step <<= 1) {
        const unsigned int jump = step << 1;
        const float step_d = (float)step;
        float twiddle_re = 1.0;
        float twiddle_im = 0.0;
        for (unsigned int group = 0; group < step; group++)
        {
            for (unsigned int pair = group; pair < N; pair += jump)
            {
                const unsigned int match = pair + step;
                const float product_re = twiddle_re * data_re[match] - twiddle_im * data_im[match];
                const float product_im = twiddle_im * data_re[match] + twiddle_re * data_im[match];
                data_re[match] = data_re[pair] - product_re;
                data_im[match] = data_im[pair] - product_im;
                data_re[pair] += product_re;
                data_im[pair] += product_im;
            }
            // we need the factors below for the next iteration
            // if we don't iterate then don't compute
            if (group + 1 == step)
            {
                continue;
            }
            float angle = pi * ((float)group + 1) / step_d;
            twiddle_re = cosf(angle);
            twiddle_im = sinf(angle);
        }
    }
}

void fft(float data_re[], float data_im[], const int N)
{
    rearrange(data_re, data_im, N);
    compute(data_re, data_im, N);
}


void mfcc_create(mfcc_t *mfcc,int num_mfcc_features, int feature_offset, int num_fbank, int frame_len, float preempha, int is_append_energy)
{
    /*  This is the methodology of processing the MFCC  */
    
    /*
       **************************************************
       *                    Waveform                    *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                   DFT OR FFT                   *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *             Log-Amplitude Spectrum             *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                  Mel-Scaling                   *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *         Discrete Cosine Transform (DCT)        *
       *                       |                        *
       *                       |                        *
       *                       V                        *
       *                     MFCCs                      *
       **************************************************
     */
    
    
    
    mfcc->num_mfcc_features = num_mfcc_features;
    mfcc->num_features_offset = feature_offset;
    mfcc->num_fbank = num_fbank;
    mfcc->frame_len = frame_len;
    mfcc->preempha = preempha;

    // Round-up to nearest power of 2.
    mfcc->frame_len_padded = 512;

    //create window function, hanning
    // By processing data through HANNING before applying FFT, more realistic results can be obtained.
    for (int i = 0; i < frame_len; i++)
        mfcc->window_func[i] = 0.5f - 0.5f*cosf((float)M_2PI * ((float)i) / (frame_len));

    //create mel filterbank
    create_mel_fbank(mfcc);

    //create DCT matrix
    create_dct_matrix(mfcc->num_fbank , num_mfcc_features, mfcc);

#ifdef MFCC_PLATFORM_ARM
    
    // MARK: - this can be static // depends on the Hardware
    //initialize FFT
    mfcc->rfft = mfcc_malloc(sizeof(arm_rfft_fast_instance_f32));
    arm_rfft_fast_init_f32(mfcc->rfft, mfcc->frame_len_padded);
#else

#endif
    return;
}


void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc)
{
    int32_t k, n;
    float normalizer;
#ifdef MFCC_PLATFORM_ARM
    arm_sqrt_f32(2.0f/(float)input_length, &normalizer);
#else
    normalizer = sqrtf(2.0f/(float)input_length);
#endif
    for (k = 0; k < coefficient_count; k++)
    {
        for (n = 0; n < input_length; n++)
        {
            mfcc->dct_matrix[k*input_length+n] = normalizer * cosf( ((float)M_PI)/input_length * (n + 0.5f) * k );
        }
    }
    return;
}

void create_mel_fbank(mfcc_t *mfcc) {

    // compute points evenly spaced in mels
    float mel_low_freq = MelScale(MEL_LOW_FREQ);                                    // MARK: - this can be fixed
    float mel_high_freq = MelScale(MEL_HIGH_FREQ);                                  // MARK: - this can be fixed
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (mfcc->num_fbank +1);   // MARK: - this can be fixed

    float bin[28];

    for (int i=0; i<mfcc->num_fbank+2; i++)
    {
        mfcc->mel_fbins[i] = mel_low_freq + mel_freq_delta*i;
        mfcc->mel_fbins[i] = floor((mfcc->frame_len_padded+1)*InverseMelScale(mfcc->mel_fbins[i])/SAMP_FREQ);
    }

    return;
}
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out)
{
    int32_t i, j, bin;
    //1. TensorFlow way of normalizing .wav data to (-1,1) and 2. do pre-emphasis.
    float last = (float)audio_data[0];
    mfcc->frame[0] = last / (1 << 15);  

    for (i = 1; i < mfcc->frame_len; i++) {
        mfcc->frame[i] = ((float)audio_data[i] - last * mfcc->preempha) / (1<<15); 
        last = (float)audio_data[i];

    }
    //Fill up remaining with zeros
    if(mfcc->frame_len_padded - mfcc->frame_len) 
    {
        for (i = mfcc->frame_len; i < mfcc->frame_len_padded - mfcc->frame_len; i++)
            mfcc->frame[i] = 0.0f;
    }


    // windows filter
    for (i = 0; i < mfcc->frame_len; i++) {
        mfcc->frame[i] *= mfcc->window_func[i];
    }


    float *data_re = mfcc->frame;
    float data_im[512];

    for (int i = 0; i < 512; ++i) {
        data_im[i] = 0.0f;
    }


    fft(data_re, data_im, mfcc->frame_len_padded);
    
    
   // FFT data structure
    // only need half (N/2+1)
    for (int i = 0; i <= mfcc->frame_len_padded/2; i++) {
        mfcc->frame[i] = (data_re[i] * data_re[i] + data_im[i]* data_im[i])/mfcc->frame_len_padded;
    }

    //Apply mel filterbanks
    apply_filter_banks(mfcc);

    //Take log
    float total_energy = 0;
    for (bin = 0; bin < mfcc->num_fbank; bin++)
    {
        total_energy += mfcc->mel_energies[bin];
        mfcc->mel_energies[bin] = logf(mfcc->mel_energies[bin]);
    }

    //Take DCT. Uses matrix mul.
    int out_index = 0;
    for (i = mfcc->num_features_offset; i < mfcc->num_mfcc_features; i++)
    {
        float sum = 0.0;
        for (j = 0; j < mfcc->num_fbank ; j++)
        {
            sum += mfcc->dct_matrix[i*mfcc->num_fbank +j] * mfcc->mel_energies[j];
        }
        mfcc_out[out_index] = sum;
        out_index ++;
    }

}


void apply_filter_banks(mfcc_t *mfcc){

    float right_this, center_this;
    
    float left_next = mfcc->mel_fbins[0];
    float center_next = mfcc->mel_fbins[1];
    
    float mel_energy_this = 0;
    float mel_energy_next = 0;
    
    // calc first left
    
    for (int i = left_next + 1; i < center_next; i++) {
        float mel_factor = (i - left_next) / (center_next - left_next);
        mel_energy_this += mfcc->frame[i] * mel_factor;
    }
    

    for (int j = 0; j < mfcc->num_fbank-1 ; j++){
        
        right_this  = mfcc->mel_fbins[j + 2];
        center_this = mfcc->mel_fbins[j + 1];
        
        left_next   = mfcc->mel_fbins[j + 1];
        center_next = mfcc->mel_fbins[j + 2];
       
        mel_energy_this += mfcc->frame[(int)center_this];
        
        float calc_base = (center_next - left_next);
        
        int i , k;
        for( i = left_next + 1,  k = right_this - 1; (i < center_next) && (k > center_this); i++, k--){
            float mel_factor = (i - left_next) / calc_base;
            mel_energy_this += mfcc->frame[i] * mel_factor;
            // calc next right
            mel_energy_next += mfcc->frame[k] * mel_factor;// change index
        }
        
        
        if (mel_energy_this == 0.0f)
            mel_energy_this = FLT_MIN;
        
        mfcc->mel_energies[j] = mel_energy_this;
        mel_energy_this=mel_energy_next;
        mel_energy_next=0;
    }
    
    // cal last right
    right_this  = mfcc->mel_fbins[mfcc->num_fbank - 1];
    center_this = mfcc->mel_fbins[mfcc->num_fbank - 2];
    
    for (int i = center_this+1; i < right_this; i++) {
        float mel_factor = (right_this - i) / (right_this - center_this);
        mel_energy_this += mfcc->frame[i] * mel_factor;
    }

    if (mel_energy_this == 0.0f)
        mel_energy_this = FLT_MIN;
    
    mfcc->mel_energies[mfcc->num_fbank-1] = mel_energy_this;
    mel_energy_this=mel_energy_next;
    mel_energy_next=0;
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
int math_errno;
int* __errno() { return &math_errno; }


#endif