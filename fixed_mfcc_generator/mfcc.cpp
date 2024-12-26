#include "mfcc.h"

#define INV_LOG2_E_Q1DOT31  UINT64_C(0x58b90bfc) // Inverse log base 2 of e
#define INV_LOG2_10_Q1DOT31 UINT64_C(0x268826a1) // Inverse log base 2 of 10

int32_t log2fix (uint32_t x, int precision)
{
    // This implementation is based on Clay. S. Turner's fast binary logarithm
    // algorithm[1].

    int32_t b = 1U << (precision - 1);
    int32_t y = 0;

    if (precision < 1 || precision > 31) {
        return INT32_MAX; // indicates an error
    }

    if (x == 0) {
        return INT32_MIN; // represents negative infinity
    }

    while (x < 1U << precision) {
        x <<= 1;
        y -= 1U << precision;
    }

    while (x >= 2U << precision) {
        x >>= 1;
        y += 1U << precision;
    }

    uint64_t z = x;

    for (int i = 0; i < precision; i++) {
        z = z * z >> precision;
        if (z >= 2U << precision) {
            z >>= 1;
            y += b;
        }
        b >>= 1;
    }

    return y;
}

int32_t logfix (uint32_t x, int precision)
{
    uint64_t t;

    t = log2fix(x, precision) * INV_LOG2_E_Q1DOT31;

    return t >> 31;
}



/*
  Henceforth "short" implies 16-bit word. If this is not
  the case in your architecture, please replace "short"
  with a type definition which *is* a 16-bit word.
*/


/*
  Since we only use 3/4 of N_WAVE, we define only
  this many samples, in order to conserve data space.
*/
short Sinewave[N_WAVE-N_WAVE/4] = {
      0,    201,    402,    603,    804,   1005,   1206,   1406,
   1607,   1808,   2009,   2209,   2410,   2610,   2811,   3011,
   3211,   3411,   3611,   3811,   4011,   4210,   4409,   4608,
   4807,   5006,   5205,   5403,   5601,   5799,   5997,   6195,
   6392,   6589,   6786,   6982,   7179,   7375,   7571,   7766,
   7961,   8156,   8351,   8545,   8739,   8932,   9126,   9319,
   9511,   9703,   9895,  10087,  10278,  10469,  10659,  10849,
  11038,  11227,  11416,  11604,  11792,  11980,  12166,  12353,
  12539,  12724,  12909,  13094,  13278,  13462,  13645,  13827,
  14009,  14191,  14372,  14552,  14732,  14911,  15090,  15268,
  15446,  15623,  15799,  15975,  16150,  16325,  16499,  16672,
  16845,  17017,  17189,  17360,  17530,  17699,  17868,  18036,
  18204,  18371,  18537,  18702,  18867,  19031,  19194,  19357,
  19519,  19680,  19840,  20000,  20159,  20317,  20474,  20631,
  20787,  20942,  21096,  21249,  21402,  21554,  21705,  21855,
  22004,  22153,  22301,  22448,  22594,  22739,  22883,  23027,
  23169,  23311,  23452,  23592,  23731,  23869,  24006,  24143,
  24278,  24413,  24546,  24679,  24811,  24942,  25072,  25201,
  25329,  25456,  25582,  25707,  25831,  25954,  26077,  26198,
  26318,  26437,  26556,  26673,  26789,  26905,  27019,  27132,
  27244,  27355,  27466,  27575,  27683,  27790,  27896,  28001,
  28105,  28208,  28309,  28410,  28510,  28608,  28706,  28802,
  28897,  28992,  29085,  29177,  29268,  29358,  29446,  29534,
  29621,  29706,  29790,  29873,  29955,  30036,  30116,  30195,
  30272,  30349,  30424,  30498,  30571,  30643,  30713,  30783,
  30851,  30918,  30984,  31049,  31113,  31175,  31236,  31297,
  31356,  31413,  31470,  31525,  31580,  31633,  31684,  31735,
  31785,  31833,  31880,  31926,  31970,  32014,  32056,  32097,
  32137,  32176,  32213,  32249,  32284,  32318,  32350,  32382,
  32412,  32441,  32468,  32495,  32520,  32544,  32567,  32588,
  32609,  32628,  32646,  32662,  32678,  32692,  32705,  32717,
  32727,  32736,  32744,  32751,  32757,  32761,  32764,  32766,
  32767,  32766,  32764,  32761,  32757,  32751,  32744,  32736,
  32727,  32717,  32705,  32692,  32678,  32662,  32646,  32628,
  32609,  32588,  32567,  32544,  32520,  32495,  32468,  32441,
  32412,  32382,  32350,  32318,  32284,  32249,  32213,  32176,
  32137,  32097,  32056,  32014,  31970,  31926,  31880,  31833,
  31785,  31735,  31684,  31633,  31580,  31525,  31470,  31413,
  31356,  31297,  31236,  31175,  31113,  31049,  30984,  30918,
  30851,  30783,  30713,  30643,  30571,  30498,  30424,  30349,
  30272,  30195,  30116,  30036,  29955,  29873,  29790,  29706,
  29621,  29534,  29446,  29358,  29268,  29177,  29085,  28992,
  28897,  28802,  28706,  28608,  28510,  28410,  28309,  28208,
  28105,  28001,  27896,  27790,  27683,  27575,  27466,  27355,
  27244,  27132,  27019,  26905,  26789,  26673,  26556,  26437,
  26318,  26198,  26077,  25954,  25831,  25707,  25582,  25456,
  25329,  25201,  25072,  24942,  24811,  24679,  24546,  24413,
  24278,  24143,  24006,  23869,  23731,  23592,  23452,  23311,
  23169,  23027,  22883,  22739,  22594,  22448,  22301,  22153,
  22004,  21855,  21705,  21554,  21402,  21249,  21096,  20942,
  20787,  20631,  20474,  20317,  20159,  20000,  19840,  19680,
  19519,  19357,  19194,  19031,  18867,  18702,  18537,  18371,
  18204,  18036,  17868,  17699,  17530,  17360,  17189,  17017,
  16845,  16672,  16499,  16325,  16150,  15975,  15799,  15623,
  15446,  15268,  15090,  14911,  14732,  14552,  14372,  14191,
  14009,  13827,  13645,  13462,  13278,  13094,  12909,  12724,
  12539,  12353,  12166,  11980,  11792,  11604,  11416,  11227,
  11038,  10849,  10659,  10469,  10278,  10087,   9895,   9703,
   9511,   9319,   9126,   8932,   8739,   8545,   8351,   8156,
   7961,   7766,   7571,   7375,   7179,   6982,   6786,   6589,
   6392,   6195,   5997,   5799,   5601,   5403,   5205,   5006,
   4807,   4608,   4409,   4210,   4011,   3811,   3611,   3411,
   3211,   3011,   2811,   2610,   2410,   2209,   2009,   1808,
   1607,   1406,   1206,   1005,    804,    603,    402,    201,
      0,   -201,   -402,   -603,   -804,  -1005,  -1206,  -1406,
  -1607,  -1808,  -2009,  -2209,  -2410,  -2610,  -2811,  -3011,
  -3211,  -3411,  -3611,  -3811,  -4011,  -4210,  -4409,  -4608,
  -4807,  -5006,  -5205,  -5403,  -5601,  -5799,  -5997,  -6195,
  -6392,  -6589,  -6786,  -6982,  -7179,  -7375,  -7571,  -7766,
  -7961,  -8156,  -8351,  -8545,  -8739,  -8932,  -9126,  -9319,
  -9511,  -9703,  -9895, -10087, -10278, -10469, -10659, -10849,
 -11038, -11227, -11416, -11604, -11792, -11980, -12166, -12353,
 -12539, -12724, -12909, -13094, -13278, -13462, -13645, -13827,
 -14009, -14191, -14372, -14552, -14732, -14911, -15090, -15268,
 -15446, -15623, -15799, -15975, -16150, -16325, -16499, -16672,
 -16845, -17017, -17189, -17360, -17530, -17699, -17868, -18036,
 -18204, -18371, -18537, -18702, -18867, -19031, -19194, -19357,
 -19519, -19680, -19840, -20000, -20159, -20317, -20474, -20631,
 -20787, -20942, -21096, -21249, -21402, -21554, -21705, -21855,
 -22004, -22153, -22301, -22448, -22594, -22739, -22883, -23027,
 -23169, -23311, -23452, -23592, -23731, -23869, -24006, -24143,
 -24278, -24413, -24546, -24679, -24811, -24942, -25072, -25201,
 -25329, -25456, -25582, -25707, -25831, -25954, -26077, -26198,
 -26318, -26437, -26556, -26673, -26789, -26905, -27019, -27132,
 -27244, -27355, -27466, -27575, -27683, -27790, -27896, -28001,
 -28105, -28208, -28309, -28410, -28510, -28608, -28706, -28802,
 -28897, -28992, -29085, -29177, -29268, -29358, -29446, -29534,
 -29621, -29706, -29790, -29873, -29955, -30036, -30116, -30195,
 -30272, -30349, -30424, -30498, -30571, -30643, -30713, -30783,
 -30851, -30918, -30984, -31049, -31113, -31175, -31236, -31297,
 -31356, -31413, -31470, -31525, -31580, -31633, -31684, -31735,
 -31785, -31833, -31880, -31926, -31970, -32014, -32056, -32097,
 -32137, -32176, -32213, -32249, -32284, -32318, -32350, -32382,
 -32412, -32441, -32468, -32495, -32520, -32544, -32567, -32588,
 -32609, -32628, -32646, -32662, -32678, -32692, -32705, -32717,
 -32727, -32736, -32744, -32751, -32757, -32761, -32764, -32766,
};

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
    mfcc->preempha = (short)(preempha * (1 << 15));

    // Round-up to nearest power of 2.
    mfcc->frame_len_padded = 512;

    //create window function, hanning
    // By processing data through HANNING before applying FFT, more realistic results can be obtained.
    float temp;
    float frame_len_inv = 1.0/(float)frame_len;
    for (int i = 0; i < frame_len; i++){
        temp = 0.5f - 0.5f*cosf((float)M_2PI * ((float)i) * frame_len_inv);
        mfcc->window_func[i] = (short)(temp* (1 << 15));
    }

    //create mel filterbank
    create_mel_fbank(mfcc);

    //create DCT matrix
    create_dct_matrix(mfcc->num_fbank , num_mfcc_features, mfcc);
    return;
}


void create_dct_matrix(int32_t input_length, int32_t coefficient_count, mfcc_t* mfcc)
{
    int32_t k, n;
    float normalizer = sqrtf(2.0f/(float)input_length);
    float input_length_inv = 1.0/(float)input_length;
    
    for (k = 0; k < coefficient_count; k++)
    {
        for (n = 0; n < input_length; n++)
        {
            mfcc->dct_matrix[k*input_length+n] = normalizer * cosf( ((float)M_PI)*input_length_inv * (n + 0.5f) * k );
        }
    }
    return;
}
void create_mel_fbank(mfcc_t *mfcc) {

    // compute points evenly spaced in mels
    float mel_low_freq = MelScale(MEL_LOW_FREQ);
    float mel_high_freq = MelScale(MEL_HIGH_FREQ);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (mfcc->num_fbank +1);

    float SAMP_FREQ_inv = 1.0/ SAMP_FREQ;
    for (int i=0; i<mfcc->num_fbank+2; i++)
    {
        mfcc->mel_fbins[i] = mel_low_freq + mel_freq_delta*i;
        mfcc->mel_fbins[i] = floor((mfcc->frame_len_padded+1)*InverseMelScale(mfcc->mel_fbins[i])*SAMP_FREQ_inv);
    }

    return;
}

short data_im2[512];
short q15_frame[512];
#define PRECISION 24
#define SCALE 16777216
void mfcc_compute(mfcc_t *mfcc, const int16_t * audio_data, float* mfcc_out)
{
    int32_t i, j, bin;
    // EF_GPIO8_writeData(PORT_C , 2); // windowing + preempha 
    
    short last = audio_data[0];
    q15_frame[0] =  FIX_MPY( mfcc->frame[0], mfcc->window_func[0]);
    
    for (i = 1; i < mfcc->frame_len; i++) {
        q15_frame[i] =  audio_data[i] - FIX_MPY(last, mfcc->preempha);
        q15_frame[i] =  FIX_MPY( q15_frame[i], mfcc->window_func[i]);
        last =  audio_data[i];
    }

    // EF_GPIO8_writeData(PORT_C , 3); // fix fft 

    for (int i = 0; i < 512; ++i) {
        data_im2[i] = 0;
    }
    fix_fft(q15_frame, data_im2, 9, 0);
    
   // FFT data structure
    // only need half (N/2+1)
    float temp_div = 0.0000305185094;
    for (int i = 0; i <= 256; i++) {
        mfcc->frame[i] = (q15_frame[i] * q15_frame[i] + data_im2[i]* data_im2[i]);
        mfcc->frame[i] *= temp_div;
    }
    
    // EF_GPIO8_writeData(PORT_C , 4); // apply filter banks 
    
    apply_filter_banks2(mfcc);


    // EF_GPIO8_writeData(PORT_C , 5); // log 

    //Take log
    float scale_div = 1.0f / SCALE;
    for (bin = 0; bin < mfcc->num_fbank; bin++)
    {
        mfcc->mel_energies[bin] = logfix(mfcc->mel_energies[bin] * SCALE, PRECISION) * scale_div;
    }

    // EF_GPIO8_writeData(PORT_C , 6); // DCT

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

    // EF_GPIO8_writeData(PORT_C , 7);
}

/*
  FIX_MPY() - fixed-point multiplication & scaling.
  Substitute inline assembly for hardware-specific
  optimization suited to a particluar DSP processor.
  Scaling ensures that result remains 16-bit.
*/

short FIX_MPY(short a, short b)
{
    /* shift right one less bit (i.e. 15-1) */
    int c = ((int)a * (int)b) >> 14;
    /* last bit shifted out = rounding-bit */
    b = c & 0x01;
    /* last shift + rounding bit */
    a = (c >> 1) + b;
    return a;
}

/* fix_fft.c - Fixed-point in-place Fast Fourier Transform  */
/*
  All data are fixed-point short integers, in which -32768
  to +32768 represent -1.0 to +1.0 respectively. Integer
  arithmetic is used for speed, instead of the more natural
  floating-point.

  For the forward FFT (time -> freq), fixed scaling is
  performed to prevent arithmetic overflow, and to map a 0dB
  sine/cosine wave (i.e. amplitude = 32767) to two -6dB freq
  coefficients. The return value is always 0.

  For the inverse FFT (freq -> time), fixed scaling cannot be
  done, as two 0dB coefficients would sum to a peak amplitude
  of 64K, overflowing the 32k range of the fixed-point integers.
  Thus, the fix_fft() routine performs variable scaling, and
  returns a value which is the number of bits LEFT by which
  the output must be shifted to get the actual amplitude
  (i.e. if fix_fft() returns 3, each value of fr[] and fi[]
  must be multiplied by 8 (2**3) for proper scaling.
  Clearly, this cannot be done within fixed-point short
  integers. In practice, if the result is to be used as a
  filter, the scale_shift can usually be ignored, as the
  result will be approximately correctly normalized as is.

  Written by:  Tom Roberts  11/8/89
  Made portable:  Malcolm Slaney 12/15/94 malcolm@interval.com
  Enhanced:  Dimitrios P. Bouras  14 Jun 2006 dbouras@ieee.org
*/
short  fr[512];
short  fi[512];
int fix_fft(short fr1[], short fi1[], short m, short inverse){
    
    for (int i =0; i<512 ; i++ ){
        
        fr[i]= fr1[i];
        fi[i]= fi1[i];
    }
    
    int mr, nn, i, j, l, k, istep, n, scale, shift;
    short qr, qi, tr, ti, wr, wi;

    n = 1 << m;

    /* max FFT size = N_WAVE */
    if (n > N_WAVE)
        return -1;

    mr = 0;
    nn = n - 1;
    scale = 0;

    /* decimation in time - re-order data */
    for (m=1; m<=nn; ++m) {
        l = n;
        do {
            l >>= 1;
        } while (mr+l > nn);
        mr = (mr & (l-1)) + l;

        if (mr <= m)
            continue;
        tr = fr[m];
        fr[m] = fr[mr];
        fr[mr] = tr;
        ti = fi[m];
        fi[m] = fi[mr];
        fi[mr] = ti;
    }

    l = 1;
    k = LOG2_N_WAVE-1;
    while (l < n) {
        if (inverse) {
            /* variable scaling, depending upon data */
            shift = 0;
            for (i=0; i<n; ++i) {
                j = fr[i];
                if (j < 0)
                    j = -j;
                m = fi[i];
                if (m < 0)
                    m = -m;
                if (j > 16383 || m > 16383) {
                    shift = 1;
                    break;
                }
            }
            if (shift)
                ++scale;
        } else {
            /*
              fixed scaling, for proper normalization --
              there will be log2(n) passes, so this results
              in an overall factor of 1/n, distributed to
              maximize arithmetic accuracy.
            */
            shift = 1;
        }
        /*
          it may not be obvious, but the shift will be
          performed on each data point exactly once,
          during this pass.
        */
        
        istep = l << 1;
        fr[0] =fr[0];
        fi[0] =fi[0];
        for (m=0; m<l; ++m) {
            j = m << k;
            /* 0 <= j < N_WAVE/2 */
            wr =  Sinewave[j+N_WAVE/4];
            wi = -Sinewave[j];
            if (inverse)
                wi = -wi;
            if (shift) {
                wr >>= 1;
                wi >>= 1;
            }
            for (i=m; i<n; i+=istep) {
                j = i + l;
                tr = FIX_MPY(wr,fr[j]) - FIX_MPY(wi,fi[j]);
                ti = FIX_MPY(wr,fi[j]) + FIX_MPY(wi,fr[j]);
                qr = fr[i];
                qi = fi[i];
                if (shift) {
                    qr >>= 1;
                    qi >>= 1;
                }
                fr[j] = qr - tr;
                fi[j] = qi - ti;
                fr[i] = qr + tr;
                fi[i] = qi + ti;
            }
        }
        --k;
        l = istep;
    }
    
    for (int i =0; i<512 ; i++ ){
        
        fr1[i]= fr[i];
        fi1[i]= fi[i];
    }
    return scale;
}
inline float int_based_division(int a, int b){
    int c = a * 100000 / b;
    float d = (float) c * 0.00001f;
    return d;
}


void apply_filter_banks2(mfcc_t *mfcc){

    int right_this, center_this;
    int left_next = mfcc->mel_fbins[0];
    int center_next = mfcc->mel_fbins[1];
    
    float mel_energy_this = 0;
    float mel_energy_next = 0;
    float mel_factor;
    
    // calc first left
    
    for (int i = left_next + 1; i < center_next; i++) {
        mel_factor = int_based_division(i - left_next, center_next - left_next);
        mel_energy_this += mfcc->frame[i] * mel_factor;
    }
    
    for (int j = 0; j < mfcc->num_fbank-1 ; j++){
        
        right_this  = mfcc->mel_fbins[j + 2];
        center_this = mfcc->mel_fbins[j + 1];
        
        left_next   = mfcc->mel_fbins[j + 1];
        center_next = mfcc->mel_fbins[j + 2];
       
        mel_energy_this += mfcc->frame[center_this];
        
        int calc_base = (center_next - left_next);

        int i , k;
        for( i = left_next + 1,  k = right_this - 1; (i < center_next) && (k > center_this); i++, k--){
            mel_factor = int_based_division(i - left_next, calc_base);
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
        mel_factor = int_based_division(i - center_this, right_this - center_this);
        mel_energy_this += mfcc->frame[i] * mel_factor;
    }

    if (mel_energy_this == 0.0f)
        mel_energy_this = FLT_MIN;
    
    mfcc->mel_energies[mfcc->num_fbank-1] = mel_energy_this;
    mel_energy_this=mel_energy_next;
    mel_energy_next=0;
}


void quantize_data(float* din, int8_t* dout, uint32_t size, uint32_t int_bit) {
    #define _MAX(x, y) (((x) > (y)) ? (x) : (y))
    #define _MIN(x, y) (((x) < (y)) ? (x) : (y))
    float limit = (1 << int_bit);
    float inv_limit = 1.0f / limit;    // Precompute the inverse of limit
    float inv_128 = 1.0f / 128.0f;     // Precompute the inverse of 128
    float d;

    for (uint32_t i = 0; i < size; i++) {
        d = round(_MAX(_MIN(din[i], limit), -limit) * inv_limit * 128);
        d *= inv_128; // Equivalent to d = d / 128.0f;
        dout[i] = round(d * 127);
    }
}

int math_errno;
int* __errno() { return &math_errno; }


