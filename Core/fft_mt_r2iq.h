#pragma once

#include "r2iq.h"
#include "fftw3.h"

// use up to this many threads
#define N_MAX_R2IQ_THREADS 4


struct Fast_R2C_Decimation
{
    // N = fftSize
    // filterDiv = 2, 4, 8, .. - must be power of 2!
    // decimation: complex output rate = (real) input rate / 2^(decimation+1)
    Fast_R2C_Decimation(int N = 8192, int filterDivLog2_ = 2, int decimation_ = 0)
        : fwdDim(N)
        , filterDivLog2(filterDivLog2_)
        , filterDiv(1 << filterDivLog2_)
        , numFwdBins(N / 2)
        , invDim(numFwdBins / (1 << decimation_))   // was mInvFftDim = mfftdim[k] = halfFft / 2^k
        , filterLen((N / filterDiv) + 1)
        , inputStep(N - (N / filterDiv))
        , decimation(decimation_)
        , mixGranularity(filterDiv)  // = multiple of N / (N-N/div) = multiple of div/(div-1) => div
    { }

    static bool is_power_of_two(int N) {
        /* https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */
        bool r = N && !(N & (N - 1));
        return r;
    }

    static bool next_power_of_two(int N) {
        /* https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 */
        /* compute the next highest power of 2 of 32-bit v */
        unsigned v = N;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    static int filterDivLog2ForRequiredTaps(int requestedFilterLen, int N = 8192)
    {
        if (requestedFilterLen > (N / 4 + 1))
            return 1;
        int d = (N + requestedFilterLen - 1) / requestedFilterLen;  // round up
        d = next_power_of_two(d);
        int filterLen = N / d + 1;
        if (filterLen < requestedFilterLen)
            d /= 2;
        int lg2 = 0;
        while ((1 << lg2) != d)
            ++lg2;
        return lg2;
    }

    int fwdDim;     // FFT forward dimension N (power of 2)
    int filterDivLog2;  // log2 of divider for filterLen
    int filterDiv;      // divider for filterLen
    int numFwdBins; // number of complex bins of forward FFT
                    // = halfFft = FFTN_R_ADC / 2;    // half the size of the first fft at ADC 64Msps real rate (8192)
    int invDim;     // was mInvFftDim = mfftdim[k] = halfFft / 2^k
    int filterLen;  // filter length M
    int inputStep;  // how many samples to advance per step
    int decimation; // complex output rate = (real) input rate / 2^(decimation+1)
    int mixGranularity;   // the tuneBin needs to be a multiple of this
};

struct BufferModValues
{
    int prevSumFFTs;
    int prevCopyCount;
    int prevCopyOff;
    int numFFTs;
};

struct BufferMods
{
    int RepCount;
    int Queue_Size;
    int maxTimeBuffer;
    int maxNumFFTs;
    int sumNumFFTs;
    BufferModValues* modValues;
};




class fft_mt_r2iq : public r2iqControlClass
{
public:
    fft_mt_r2iq();
    virtual ~fft_mt_r2iq();

    float setFreqOffset(float offset);

    void Init(float gain, int16_t** buffers, float* obuffers, int* num_joint_obuffers, int* joint_num_samples) override;
    void TurnOn() override;
    void TurnOff(void) override;
    bool IsOn(void) override;
    void DataReady(void) override;

private:
    int16_t** buffers;    // pointer to input buffers
    float* obuffers;   // pointer to output buffers
    int bufIdx;         // index to next buffer to be processed
    volatile std::atomic<int> cntr;           // counter of input buffer to be processed
    r2iqThreadArg* lastThread;

    float GainScale;
    Fast_R2C_Decimation mDecimCore[NDECIDX];  // decimation core
    const BufferMods* mBufferMods[NDECIDX];

    //int mInvFftDim[NDECIDX]; // FFT N dimensions: mfftdim[k] = halfFft / 2^k

    int mtunebin;

    void *r2iqThreadf(r2iqThreadArg *th);   // thread function

    int halfFft;    // half the size of the first fft at ADC 64Msps real rate (2048)

    fftwf_complex *filterHw[NDECIDX]; // Hw complex to each decimation ratio

	fftwf_plan plan_t2f_r2c;          // fftw plan buffers Freq to Time complex to complex per decimation ratio
	fftwf_plan plans_f2t_c2c[NDECIDX];

    uint32_t processor_count;
    r2iqThreadArg* threadArgs[N_MAX_R2IQ_THREADS];
    std::condition_variable cvADCbufferAvailable;  // unlock when a sample buffer is ready
    std::mutex mutexR2iqControl;                   // r2iq control lock
    std::thread* r2iq_thread[N_MAX_R2IQ_THREADS]; // thread pointers
};
