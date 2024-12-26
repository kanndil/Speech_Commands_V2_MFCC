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

#ifndef __MFCC_H__
#define __MFCC_H__

#include <math.h>

#include <stdint.h>
#include <math.h>
#include <float.h>

#define AUDIO_FRAME_LEN 512
#define SAMP_FREQ 16000
#define SAMP_FREQ_INV (1/16000)
#define MEL_LOW_FREQ 20
#define NYQUIST_FREQ (SAMP_FREQ/2)
#define MEL_HIGH_FREQ NYQUIST_FREQ
#define M_2PI 6.283185307179586476925286766559005
#define FFT_N 512
#define N_WAVE      1024    /* full length of Sinewave[] */
#define LOG2_N_WAVE 10      /* log2(N_WAVE) */


//the mfcc feature for kws
#define MFCC_LEN            (62)
#define MFCC_COEFFS_FIRST   (1)     // ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN     (13)    // the total coefficient to calculate
#define MFCC_TOTAL_NUM_BANK (26)    // total number of filter bands
#define MFCC_COEFFS         (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)



typedef struct _mfcc_t{ // MARK: - this can be static
    int num_mfcc_features;      // this is the number of 
	int num_features_offset;
	int num_fbank;
    int frame_len;
    int frame_len_padded;           // MARK: - this can be static // this can be fixed to make other variables static
    
	short preempha;
    float frame[512];                                  
    float mel_energies[MFCC_TOTAL_NUM_BANK];            
    short window_func[512];            
    int mel_fbins[MFCC_TOTAL_NUM_BANK+2];
    float dct_matrix[MFCC_COEFFS_LEN*MFCC_TOTAL_NUM_BANK];             

} mfcc_t;


void mfcc_create(mfcc_t *mfcc,int num_mfcc_features, int feature_offset, int num_fbank, int frame_len, float preempha, int is_append_energy);
void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc);
void create_mel_fbank(mfcc_t *mfcc);
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out);
void apply_filter_banks2(mfcc_t *mfcc);
short FIX_MPY(short a, short b);
int fix_fft(short fr1[], short fi1[], short m, short inverse);
void quantize_data(float* din, int8_t* dout, uint32_t size, uint32_t int_bit);


#define FFT_N 512
#define N_WAVE      1024    /* full length of Sinewave[] */
#define LOG2_N_WAVE 10      /* log2(N_WAVE) */

#define inv_2595 (1.0f / 2595.0f)
#define inv_700 (1.0f / 700.0f)

static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (pow(10, mel_freq * inv_2595) - 1.0f);
}
static inline float MelScale(float freq) {
    return 2595.0f * log10(1.0f + freq * inv_700);
}

int32_t log2fix (uint32_t x, size_t precision);

int32_t logfix (uint32_t x, size_t precision);



#endif