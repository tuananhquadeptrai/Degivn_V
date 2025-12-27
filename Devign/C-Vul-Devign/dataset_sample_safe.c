/* Sample from Devign Dataset - FFmpeg wav header parser */
/* Label: SAFE (target=0) */

#include <stdint.h>

typedef struct AVFormatContext AVFormatContext;
typedef struct AVIOContext AVIOContext;
typedef struct AVCodecContext AVCodecContext;

#define AVERROR_INVALIDDATA -1
#define AVERROR_PATCHWELCOME -2
#define AVERROR(x) -(x)
#define ENOMEM 12
#define AVMEDIA_TYPE_AUDIO 1
#define AV_EF_EXPLODE 1
#define AV_CODEC_ID_AAC_LATM 100
#define AV_CODEC_ID_ADPCM_G726 101
#define INT_MAX 2147483647
#define FFMIN(a,b) ((a) < (b) ? (a) : (b))
#define AV_RL16(x) (*(uint16_t*)(x))
#define AV_RL32(x) (*(uint32_t*)(x))

int ff_get_wav_header(AVFormatContext *s, AVIOContext *pb,
                      AVCodecContext *codec, int size, int big_endian)
{
    int id;
    uint64_t bitrate;

    if (size < 14) {
        return AVERROR_INVALIDDATA;
    }

    if (!big_endian) {
        id = 0;
        if (id != 0x0165) {
            bitrate = 0;
        }
    } else {
        id = 0;
        bitrate = 0;
    }

    if (size == 14) {
        /* We're dealing with plain vanilla WAVEFORMAT */
    } else {
        if (!big_endian) {
            /* bits_per_coded_sample */
        } else {
            /* bits_per_coded_sample */
        }
    }

    if (id == 0xFFFE) {
        /* codec_tag = 0 */
    } else {
        /* codec_tag = id */
    }

    if (size >= 18 && id != 0x0165) {
        int cbSize = 0;
        if (big_endian) {
            return AVERROR_PATCHWELCOME;
        }
        size -= 18;
        cbSize = FFMIN(size, cbSize);
        if (cbSize >= 22 && id == 0xfffe) {
            cbSize -= 22;
            size -= 22;
        }
        if (cbSize > 0) {
            size -= cbSize;
        }
        if (size > 0) {
            /* skip garbage */
        }
    } else if (id == 0x0165 && size >= 32) {
        int nb_streams = 0;
        int i;
        size -= 4;
        if (size < 8 + nb_streams * 20)
            return AVERROR_INVALIDDATA;
        for (i = 0; i < nb_streams; i++) {
            /* process streams */
        }
    }

    if (bitrate > INT_MAX) {
        return AVERROR_INVALIDDATA;
    }

    return 0;
}
