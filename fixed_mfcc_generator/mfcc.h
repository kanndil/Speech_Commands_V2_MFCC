/*
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
 */

#ifndef __MFCC_H__
#define __MFCC_H__

#include <stdint.h>
#include "string.h"
#include <math.h>
#include <stdio.h>

#define SAMP_FREQ 16000
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000

#define M_2PI 6.283185307179586476925286766559005

typedef struct _mfcc_t{ 
    int num_mfcc_features;      
    int num_features_offset;
    int num_fbank;
    int frame_len;
    int frame_len_padded;           
    int is_append_energy;
    short preempha;
    double frame[512];                  
    double mel_energies[26];           
    short window_func[512];            
    float mel_fbins[28];
    float mel_fbins_mel[28];
    int mel_fbins_fix[28];
    float dct_matrix[338];             
} mfcc_t;

static inline float InverseMelScale(float mel_freq);
static inline float MelScale(float freq);
void mfcc_create(mfcc_t *mfcc,int num_mfcc_features, int feature_offset, int num_fbank, int frame_len, float preempha, int is_append_energy);
//void mfcc_delete(mfcc_t* mfcc);
void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc);
void create_mel_fbank(mfcc_t *mfcc);
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out);
void apply_filter_banks(mfcc_t *mfcc);
void apply_filter_banks2(mfcc_t *mfcc);
#endif
