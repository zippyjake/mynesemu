#include <stdint.h>
#include <signal.h>
#include <assert.h>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <cstdio>

#include <SDL.h>
#include <vector>

#include <gd.h>

unsigned INPUT_BPP = 32;
const char VIDEO_CMD[] = "/mnt/te/test3.mkv";
bool Terminate = false;

#include "avi.hh"
const unsigned long long PPU_CyclesPerSecond = 1789800 * 3;
static unsigned long long Skipped_PPU_Cycles = 0;

int TurboLevel = 10;

// Integer types
typedef uint_least8_t  u8;  typedef int_least8_t  s8;
typedef uint_least16_t u16; typedef int_least16_t s16;
typedef uint_least32_t u32; typedef int_least32_t s32;

// Bitfield utilities
template<unsigned bitno, unsigned nbits=1, typename T=u8>
struct RegBit
{
    T data;
    template<typename T2>
    RegBit& operator=(T2 val)
    {
        if(nbits == 1) data = (data & ~mask()) | (!!val << bitno);
        else           data = (data & ~mask()) | ((val & mask(0)) << bitno);
        return *this;
    }
    operator unsigned() const    { return (data >> bitno) & mask(0); }
    static inline constexpr unsigned mask(unsigned shift=bitno)
        { return ((1 << nbits)-1) << shift; }
    RegBit& operator++ ()           { return *this = *this + 1; }
    unsigned operator++ (int)       { unsigned r = *this; ++*this; return r; }
    RegBit& operator^= (unsigned v) { return *this = *this ^ v; }
    RegBit& operator+= (unsigned v) { return *this = *this + v; }
};

struct rgb { float c[3]; };
class Lanczos
{
    struct Data
    {
        int                start;
        std::vector<float> contrib;   // index: 0<=i<in_size 
    };
    std::vector<Data> data; // index: 0<=i<out_size
public:
    Lanczos(int in_size, int out_size)
    {
        const int      FilterRadius = 3;
        const double   blur         = 1.0;
        const double   factor       = out_size / double(in_size);
        const double   scale        = std::min(factor, 1.0) / blur;
        const double   support      = FilterRadius / scale;
        //const unsigned contrib_size = std::min(in_size, 5 + int(2*support));

        auto LanczosPi = [](double x_pi) -> double
        {
            if(x_pi == 0.0) return 1.0;
            //if(x_pi <= -(FilterRadius*M_PI) || x_pi >= (FilterRadius*M_PI)) return 0.0;
            double x_pi_div_Radius = x_pi / FilterRadius;
            return std::sin(x_pi) * std::sin(x_pi_div_Radius) / (x_pi * x_pi_div_Radius);
        };
        data.resize(out_size);
        for(int outpos=0; outpos < out_size; ++outpos)
        {
            double center   = (outpos+0.5) / factor;
            const int start = std::max((int)(center-support+0.5), 0);
            const int end   = std::min((int)(center+support+0.5), in_size);
            const int nmax = end-start;
            const double scale_pi = scale * M_PI;
            const double s_min = -FilterRadius*M_PI;
            const double s_max =  FilterRadius*M_PI;
            double s_pi     = (start-center+0.5) * scale_pi;
            double density  = 0.0;
            auto& contrib = data[outpos].contrib; contrib.resize(nmax);
            { int n=0;
              for(; n < nmax && (s_pi < s_min); ++n, s_pi += scale_pi)
                {}
              for(; n < nmax && (s_pi < s_max); ++n, s_pi += scale_pi)
              {
                double l = LanczosPi(s_pi);
                contrib[n] = l;
                density += l;
              }
            }
            if(density > 0.0) for(auto& v: contrib) v /= density;
            data[outpos].start   = start;
            //assert(start + contrib.size() <= in_size);
        }
    }
    const Data& operator[] (unsigned i) const { return data.at(i); }
};


static X264_MKV avi;

namespace Pak
{
    std::vector<u8> ROM, VRAM(0x2000), PRAM(0x2000);
    unsigned mappernum;
    unsigned char NRAM[0x1000];
    const unsigned VROM_Granularity = 0x0400, VROM_Pages = 0x2000 / VROM_Granularity;
    const unsigned ROM_Granularity  = 0x2000, ROM_Pages = 0x10000 / ROM_Granularity;
    unsigned char* Vbanks[VROM_Pages] = {};
    unsigned char* banks[ROM_Pages]  = {};
    unsigned char* Nta[4]    = { NRAM+0x0000, NRAM+0x0400, NRAM+0x0800, NRAM+0x0C00 };
//////////////
 /*
    class VRAM_AccessTracker
    {
        struct t
        {
            u8 UsedAsSprite;
            u8 UsedAsBackground;
            u8 IOwritten, IOread;
        };
        struct t8
        {
            t row[8];

            void UpdateColor()
            {
            }
        };
    public:
        std::vector<t8> VRAM;

        void Render(u32* p, unsigned width, unsigned height)
        {
            const unsigned n_tiles = VRAM.size() / 16;
            // 512 tiles on 0x2000. Each is 8x8.
            // Figure out a tile size.
            static unsigned tiles_horizontally = 0, tiles_vertically = 0;
            if(!tiles_vertically)
            {
                tiles_horizontally = n_tiles;
                tiles_vertically   = 1;
                while( tiles_horizontally > tiles_vertically
                    && double(tiles_horizontally  ) / double(tiles_vertically  )
                    >  double(tiles_horizontally/2) / double(tiles_vertically*2) )
                {
                    tiles_horizontally /= 2;
                    tiles_vertically   *= 2;
                }
            }
            std::vector<rgb> graphic( tiles_horizontally*tiles_vertically*8*8 );
        }
    } vram_tracker;
  */
}

namespace IO
{
    SDL_Surface *s;

    /* Correct stretching aspect ratio: 292/240. Render to 256x240 and stretch to 292x240. Or render to 280x240 and stretch to 320x240. */
    const int xres=2230,yres=1832;               // Pixel ratio: 73/60   (1.216667)

    const int extended_xres = 4522;
    const int extended_yres = 2544;
    const int padding_left  = 2292;

    int joydata1[2]={0,0}, joydata2[2]={0,0}, joypos[2]={0,0};

    void Init()
    {
        SDL_Init(SDL_INIT_VIDEO);
        SDL_InitSubSystem(SDL_INIT_VIDEO);
        s = SDL_SetVideoMode(extended_xres, extended_yres, 32,0);
        signal(SIGINT, SIG_DFL);
    }

    class AudioBuffer
    {
        static const unsigned rate = PPU_CyclesPerSecond / 3, down = 10, length = rate / down;
        static const unsigned per_frame_length = length / 60;
        static const unsigned sync_room = per_frame_length * 2 / 3;
        static const unsigned render_width = per_frame_length - sync_room;
        unsigned pos_lo, pos_hi;
        int data[length];
        int previous[render_width];
    public:
        AudioBuffer(): pos_lo(0), pos_hi(0), data(), previous() { }
        void AddSample(int s)
        {
            data[pos_lo] += s;
            if(++pos_hi >= down) { pos_hi = 0; if(++pos_lo >= length) pos_lo = 0; data[pos_lo] = 0; }
        }
        unsigned FindBeginning()
        {
            //return (pos_lo + length - per_frame_length) % length;
            unsigned best_diff = ~0u, chosen_begin = 0;
            for(unsigned try_begin = 0; try_begin < sync_room; ++try_begin)
            {
                unsigned begin = (pos_lo + length - 1 - per_frame_length + try_begin) % length;
                int prev_s = 0, test_s = 0;
                unsigned avg_diff = 0;
                unsigned index = begin;
                for(unsigned l=0; l<render_width; ++l)
                {
                    int prev_diff = previous[l] - prev_s; prev_s = previous[l];
                    int test_diff = data[index] - test_s; test_s = data[index];
                    index = (index+1<length ? index+1 : 0); // go to next sample
                    int diff_diff = prev_diff - test_diff;
                    if(diff_diff < 0) diff_diff = -diff_diff;
                    avg_diff += diff_diff /* * diff_diff */ * (render_width-l);
                }
                if(avg_diff < best_diff) { best_diff = avg_diff; chosen_begin = begin; }
            }
            return chosen_begin;
        }
        void Render(u32* target, unsigned width, unsigned height)
        {
            const unsigned back = 0x01010A, line = 0x0C2040, spot = 0x5050E0;
            const unsigned begin = FindBeginning();
            float scale = height / (128.f * down);
            auto fade = [=](unsigned x, unsigned color) -> unsigned
            {
                return color;/*
                if(x >= width/10) return color;
                u8 r = color>>16, g = color>>8, b = color;
                return (r * x / (width/10-1)) * 0x10000
                     + (g * x / (width/10-1)) * 0x100
                     + (b * x / (width/10-1)) * 0x1;*/
            };
            for(unsigned x=0; x<width; ++x)
            {
                unsigned color = fade(x, back);
                for(unsigned y=0; y<height; ++y)
                {
                    u32* p = target + y*extended_xres;
                    p[x] = color;
                }
            }
            unsigned index = begin, prev=0;
            for(unsigned p=0; p<render_width; ++p)
            {
                int x = p * width / render_width;
                int s = data[index];
                if(p > 0)
                {
                    int y1 = scale * prev, y2 = scale * s;
                    if(y1 > y2) { int tmp=y1; y1=y2; y2=tmp; }
                    int xx1=x-3; if(xx1<0) xx1=0;
                    int xx2=x+3; if(xx2>=int(width)) xx2=width-1;
                    for(int y=y1; y<y2; ++y)
                    for(int xx=xx1; xx<=xx2; ++xx)
                        target[y * extended_xres + xx] = fade(xx, line);
                }
                prev = s;
                index = (index+1<length ? index+1 : 0); // go to next sample
            }
            index = begin;
            for(unsigned p=0; p<render_width; ++p)
            {
                int x = p * width / render_width;
                int s = data[index];
                int y = scale * s;
                int y1 = y-2; if(y1 < 0) y1 = 0;
                int y2 = y+2; if(y2 >= int(height)) y2 = height-1;
                unsigned color = fade(x, spot);
                for(y=y1; y<=y2; ++y)
                    target[y * extended_xres + x] = color;
                index = (index+1<length ? index+1 : 0); // go to next sample
            }
        }
    } AudioBuffers[5];

    void AudioRendering(const int samples[5])
    {
        // Audio is provided at PPU_CyclesPerSecond / 3 samples per second.
        // Our video rendering rate is about 60.0988 fps.
        // At yres=480, our padding-half is under 134 pixels.
        // Which means about 8000 samples per second.
        // Let's downscale PPU_CyclesPerSecond / 3 by a ratio of 60 (produces 29830).
        // Now the amount of sample data per frame ends up being about 500.
        // In that 500, we can keep a window of about 25 %.
        // This leaves 400 for rendering. 
        // That 400 we can render in a squeezed format (into the 134 pixels).
        for(int s=0; s<4; ++s) AudioBuffers[s].AddSample(123 - samples[s]*8);
        for(int s=4; s<5; ++s) AudioBuffers[s].AddSample(127 - samples[s]  );
        // Note: Samples for channel 4 are 0..127, all others have 0..15.
    }

    class AccessTracker
    {
    public:
        static const unsigned MaxTimer = 11;
        static const unsigned PixelScaling = 8;

        struct t
        {
            u8 Increased, Decreased, Written, Read;
            float color[3];

            void UpdateColor()
            {
                static const float Bright[MaxTimer] = // (0 + (255-0)*pow(c/10,2)) / 255
                {0.0000,0.0100,0.0400,0.0900,0.1600,
                 0.2500,0.3600,0.4900,0.6400,0.8100,1.0000};
                color[0] =          (Bright[Decreased] + Bright[Written]) * .5f;
                color[1] = std::max((Bright[Increased] + Bright[Written]) * .5f, Bright[Read] * .5f);
                color[2] = std::max(0.f, Bright[Read] - Bright[Written]);
                color[0] = (1 + color[0]*(255-1) ) / 255.0;
                color[1] = (1 + color[1]*(255-1) ) / 255.0;
                color[2] = (10+ color[2]*(255-10)) / 255.0;
            }
            void Tick()
            {
                bool changes = false;
                if(Increased) { changes = true; --Increased; }
                if(Decreased) { changes = true; --Decreased; }
                if(Written)   { changes = true; --Written;   }
                if(Read)      { changes = true; --Read;      }
                if(changes) UpdateColor();
            }
        };
        t RAM[0x800], PRAM[0x2000];
        std::vector<t> ROM;

        void Tick()
        {
            for(auto& a: RAM) a.Tick();
            for(auto& a: PRAM) a.Tick();
            for(auto& a: ROM) a.Tick();
        }

        void Render(const t* pixels,
                    unsigned in_width, unsigned /*in_height*/,
                    unsigned out_width, unsigned out_height,
                    const Lanczos& data_w, const Lanczos& data_h,
                    u32* target)
        {
            // First scale it vertically (horizontal is kept as-is)
            std::vector<rgb> vscaled_data(in_width * out_height);
            #pragma omp parallel for schedule(static)
            for(unsigned y=0; y<out_height; ++y)
            {
                const auto &data = data_h[y];
                for(unsigned p=y*in_width, x=0; x<in_width; ++x)
                {
                    rgb sample = { {} };
                    int sy = data.start; // range: 0 <= sy < in_height*8
                    for(auto d: data.contrib)
                    {
                        const auto pix = pixels[x + in_width * (sy++ / PixelScaling) ];
                        for(unsigned n=0; n<3; ++n)
                            sample.c[n] += d * pix.color[n];
                    }
                    vscaled_data[p++] = sample; // x + in_width*y  where 0<=x<in_width, 0<=y<out_height
                }
            }
            // Scale horizontally
            #pragma omp parallel for schedule(static)
            for(unsigned x=0; x<out_width; ++x)
            {
                int bright = 255;
                /*if(x >= out_width * 9 / 10)
                {
                    bright = (out_width - x) * 255 / (out_width / 10.);
                }*/
                u32* tgt = target + x;
                const auto &data = data_w[x];
                for(unsigned y=0; y<out_height; ++y)
                {
                    rgb sample = { {} };
                    int sx = data.start; // range: 0<=sx<in_width
                    for(auto d: data.contrib)
                    {
                        //printf("x=%u/%u, y=%u/%u, sx=%d/%d\n", x,out_width, y,out_height, sx,in_width);
                        const auto& pix = vscaled_data[in_width*y + (sx++) / PixelScaling];
                        for(unsigned n=0; n<3; ++n)
                            sample.c[n] += d * pix.c[n];
                    }
                    // Add some extra brightness; desaturate where needed.
                    for(unsigned n=0; n<3; ++n) sample.c[n] *= 1.3;
                    float l = sample.c[0]*0.299f + sample.c[1]*0.587f + sample.c[2]*0.114f;
                    if(l >= 1.f)      sample.c[0] = sample.c[1] = sample.c[2] = 1.f;
                    else if(l <= 0.f) sample.c[0] = sample.c[1] = sample.c[2] = 0.f;
                    else
                    {
                        float s = 1.0;
                        for(unsigned n=0; n<3; ++n)
                            if(sample.c[n] > 1.f) s = std::min(s, (l-1.f) / (l-sample.c[n]));
                            else if(sample.c[n] < 0.f) s = std::min(s, l  / (l-sample.c[n]));
                        if(s != 1.f)
                            for(unsigned n=0; n<3; ++n) sample.c[n] = (sample.c[n] - l) * s + l;
                    }
                    // Convert to RGB
                    int r = sample.c[0]*bright;
                    int g = sample.c[1]*bright;
                    int b = sample.c[2]*bright;
                    *tgt = r*0x10000 + g*0x100 + b;
                    tgt += extended_xres;
                }
            }
        }

        void Render(u32* target, unsigned width, unsigned height)
        {
            const unsigned ram_height_in  =128, ram_height_out = height;
            const unsigned ram_width_in   = 16, ram_width_out  = ram_width_in * (ram_height_out/(0.+ram_height_in));
            /*
            const unsigned pram_width_in  =128, pram_width_out = height;
            const unsigned pram_height_in = 64, pram_height_out= pram_width_in * (pram_height_out/(0.+pram_height_in));
            */

            const unsigned rom_height_out = height;
            const unsigned rom_width_out  = width - ram_width_out /*- pram_width_out */;
            static unsigned rom_width_in = 0, rom_height_in = 0;
            if(!rom_height_in)
            {
                rom_height_in  = 1;
                rom_width_in = ROM.size() / rom_height_in;
                double best_badness = 9e99;
                for(unsigned w=16; w<=1024; w *= 2)
                {
                    unsigned h = ROM.size() / w;
                    double pw = rom_height_out / double(w);
                    double ph = rom_width_out  / double(h);
                    double aspect_badness = std::abs(1.0 - pw / ph);
                    if(aspect_badness < best_badness)
                    {
                        best_badness = aspect_badness;
                        rom_height_in  = w;
                        rom_width_in = h;
                    }
                }
            }

            static Lanczos ram_w(ram_width_in * PixelScaling,  ram_width_out );
            static Lanczos ram_h(ram_height_in * PixelScaling, ram_height_out);/*
            static Lanczos pram_w(pram_width_in * PixelScaling,  pram_width_out );
            static Lanczos pram_h(pram_height_in * PixelScaling, pram_height_out);*/
            static Lanczos rom_w(rom_width_in * PixelScaling,  rom_width_out );
            static Lanczos rom_h(rom_height_in * PixelScaling, rom_height_out);
            //printf("Rendering RAM (%ux%u) into %ux%u\n", ram_width_in,ram_height_in, ram_width_out,ram_height_out);
            //printf("and then, ROM (%ux%u) into %ux%u\n", rom_width_in,rom_height_in, rom_width_out,rom_height_out);
          //#pragma omp parallel sections
          {
            u32* t = target;
            Render(&RAM[0], ram_width_in,ram_height_in, ram_width_out,ram_height_out, ram_w,ram_h, t);
            /*
           #pragma omp section
            t = target + ram_width_in;
            Render(&PRAM[0], pram_width_in,pram_height_in, pram_width_out,pram_height_out, pram_w,pram_h, t);
            */
          // #pragma omp section
            t = target + width - rom_width_out;
            Render(&ROM[0], rom_width_in,rom_height_in, rom_width_out,rom_height_out, rom_w,rom_h, t);
          }
        }
    } access;

    void RenderController(int ctrlno, u32* p, unsigned width, unsigned height)
    {
        static int prevdata[2] = {0,0};
        int ctrldata = joydata2[ctrlno];
        double arrow_w = width*2.5/22, arrow_h=height*2.5/6;
        double small_w = width*2.2/22, small_h=height*1.8/6;
        double large_w = width*3. /22, large_h=height*3. /6;
        auto MakeButton = [&] (bool state,bool was, double x,double y, double w,double h)
        {
            h *= 0.5; w *= 0.5;
            u32 color = 0xFF2010;
            for(int round=0; round<2; ++round)
            {
                double miny = y-h, maxy = y+h;
                for(int yy = int(miny); yy < int(maxy+0.999); ++yy)
                {
                    if(yy < 0 || yy >= int(height)) continue;
                    double yr = (yy+0.5 - y) / h;
                    // r = sqrt(yr*yr + xr*xr); solve(xr) -> sqrt(r - yr*yr)
                    double dist = std::sqrt(1.0 - yr*yr) * w;
                    double minx = x-dist, maxx = x+dist;
                    for(int xx = int(minx); xx < int(maxx+0.999); ++xx)
                    {
                        if(xx < 0 || xx >= int(width)) continue;
                        p[yy * extended_xres + xx] = color;
                    }
                }
                w -= 5;
                h -= 5;
                // Inner disc color
                color = state ? 0xFFA030 : (was ? 0xC09020 : 0x101010);
            }
        };
        /*
                 XX                     0
                 XX            x   x    1
               XX  XX  XX XX  XXX XXX   2
               XX  XX  XX XX  XXX XXX   3
                 XX            x   x    4
                 XX                     5
               0123456789012345678901
        */
        int c=ctrldata, d=prevdata[ctrlno];
        MakeButton(c&0x01, d&0x01, width*5.5 /22, height*0.5,   arrow_w,arrow_h); // R
        MakeButton(c&0x02, d&0x02, width*2.0 /22, height*0.5,   arrow_w,arrow_h); // L
        MakeButton(c&0x04, d&0x04, width*3.75/22, height*1.5/6, arrow_w,arrow_h); // U
        MakeButton(c&0x08, d&0x08, width*3.75/22, height*4.5/6, arrow_w,arrow_h); // D
        MakeButton(c&0x10, d&0x10, width*16. /22, height*0.5,  large_w,large_h); // B
        MakeButton(c&0x20, d&0x20, width*19.5/22, height*0.5,  large_w,large_h); // A
        MakeButton(c&0x40, d&0x40, width*9. /22, height*0.5,  small_w,small_h); // SE
        MakeButton(c&0x80, d&0x80, width*12./22, height*0.5,  small_w,small_h); // ST
        prevdata[ctrlno] = ctrldata;
    }

    void RenderControllers(u32* p, unsigned width, unsigned height)
    {
        //RenderController(0, p,                   width/2, height);
        //RenderController(1, p + (width-width/2), width/2, height);
        static bool first_time = true;
        if(first_time)
        {
            first_time = false;
            for(unsigned y=0; y<height; ++y)
                for(unsigned x=0; x<width; ++x)
                    p[y*extended_xres+x] = 0x01010A; // Back
        }
        RenderController(0, p,                                   width, height/2);
        RenderController(1, p + (height-height/2)*extended_xres, width, height/2);
    }

    void TrackRAMaccess(unsigned addr, bool write, int increment)
    {
        if(write)
        {
            access.RAM[addr].Written = AccessTracker::MaxTimer-1;
            if(increment > 0) access.RAM[addr].Increased = AccessTracker::MaxTimer-1;
            if(increment < 0) access.RAM[addr].Decreased = AccessTracker::MaxTimer-1;
        }
        else
            access.RAM[addr].Read = AccessTracker::MaxTimer-1;
        access.RAM[addr].UpdateColor();
    }
    void TrackPRAMaccess(unsigned addr, bool write, int increment)
    {
        if(write)
        {
            access.PRAM[addr].Written = AccessTracker::MaxTimer-1;
            if(increment > 0) access.PRAM[addr].Increased = AccessTracker::MaxTimer-1;
            if(increment < 0) access.PRAM[addr].Decreased = AccessTracker::MaxTimer-1;
        }
        else
            access.PRAM[addr].Read = AccessTracker::MaxTimer-1;
        access.PRAM[addr].UpdateColor();
    }
    void TrackROMaccess(unsigned addr)
    {
        access.ROM[addr].Read = AccessTracker::MaxTimer-1;
        access.ROM[addr].UpdateColor();
    }
    void TrackROMsize(unsigned size)
    {
        access.ROM.resize(size);
        for(auto& a: access.RAM) a.UpdateColor();
        for(auto& a: access.ROM) a.UpdateColor();
    }
    /*void TrackVRAMsize(unsigned size)
    {
        Pak::vram_tracker.VRAM.resize(size);
        for(auto& a: Pak::vram_tracker.VRAM) a.UpdateColor();
    }*/

    static unsigned framecounter=0;
    static rgb prev2[3][240][xres]={{{{{}}}}};
    static unsigned colorbursts[240], Colorburst=4;
    static u16 prev1[3][240][256]={{{}}}; // NES pixels corresponding to each screen location in each colorburst offset
    static bool diffs[240] = {false};     // Whether this scanline has changed from what it was the last time

    static unsigned long long CycleCounter_FullSeconds = 0;
    static unsigned long long CycleCounter_Fractional  = 0;
    static long double frame_begin_when = 0.0l;

    void PutPixel(unsigned px,unsigned py, unsigned pixel)
    {
        if(Skipped_PPU_Cycles) pixel |= 0x40; // Force de-emphasis to indicate fastforward

        u16 v = 0x8000^pixel, &p = prev1[Colorburst/4][py][px];
        if(p != v) { p = v; diffs[py] = true; }
    }
    void FlushScanline(unsigned py, unsigned length)
    {
        //if(py==239) ++framecounter; return;
        if(py < 240) colorbursts[py] = Colorburst;
        if(py == 239)
        {
            const int mul = 0x4000000;
            const unsigned Ylow = 12, Ilow = 31, Qlow = 86;
            static float coeffY[Ylow], coeffI[Ilow], coeffQ[Qlow];
            static float sinusoids[12][12];
            static int signals[512][12];
            static bool initialized = false;
            if(!initialized)
            {
                // Create crude lowpass filters
                auto makeCoeff = [] (int size, float* buffer)
                {
                    auto sinc = [] (float x) -> float
                        { if(x == 0.f) return 1.f; x *= M_PI; return std::sin(x)/x; };
                    float density = 0.f;
                    for(int q=-size/2, p=0; p<size; ++p,++q)
                    {
                        float x = q * 1.f / size;
                        buffer[p] = (size - 2*(q<0?-q:q)) + size*0.5f + 0.25f*size*sinc(4*x);
                        if(size == 12) buffer[p] = 1;
                        density += buffer[p];
                    }
                    for(int p=0; p<size; ++p) buffer[p] /= density;
                };
                makeCoeff(Ylow, coeffY);
                makeCoeff(Ilow, coeffI);
                makeCoeff(Qlow, coeffQ);
                // Create sinusoids for matching the color information
                float sine[12];
                for(int p=0; p<12; ++p)
                    sine[p] = std::sin(M_PI * (0.9 + p) / 6);
                for(int p=0; p<12; ++p)
                    { int q = p; for(auto& f: sinusoids[p]) f = sine[q++ % 12]; }
                // Create NES color to NTSC signal mappings
                for(int pixel=0; pixel<512; ++pixel)
                for(int p=0; p<12; ++p)
                {
                    // Decode the color index.
                    int color = (pixel & 0x0F), level = color<0xE ? (pixel>>4) & 3 : 1;
                    // Voltage levels, relative to synch voltage
                    static const float black=.518f, white=1.962f, attenuation=.746f,
                      levels[8] = {.350f, .518f, .962f,1.550f,  // Signal low
                                  1.094f,1.506f,1.962f,1.962f}; // Signal high
                    auto wave = [](int p) { return p%12 < 6; };
                    // NES NTSC modulator (square wave between two voltage levels):
                    float spot = levels[level + 4*(color <= 12 * wave(p+color))];
                    // De-emphasis bits attenuate a part of the signal:
                    if(((pixel & 0x40) && wave(p+4))
                    || ((pixel & 0x80) && wave(p+8))
                    || ((pixel &0x100) && wave(p+0))) spot *= attenuation;
                    // Normalize:
                    signals[pixel][p] = (spot - black) * mul / (white-black);
                }
                initialized = true;
            }

            // Vertical pixelation factor before lanczos-scaling to final size
            const unsigned PixelScaling = 8;

            static rgb RGBdata[240][xres] = {{{{0}}}};
            static Lanczos RGB_scaling(240 * PixelScaling, yres);

            // Simulate TV NTSC demodulator for this scanline
            #pragma omp parallel for schedule(static)
            for(py=0; py<240; ++py)
            {
                unsigned colorburst = colorbursts[py];
                auto& target = prev2[colorburst/4][py];
                auto& line   = prev1[colorburst/4][py];
                if(diffs[py])
                {
                    // I and Q are synthesized at 3.579545 Mhz    (wavelength ratio of 1.00000 -> 12)
                    // I is supposed to be low-passed at 1400 kHz (wavelength ratio of 2.55682 -> 30.6818)
                    // Q is supposed to be low-passed at  500 kHz (wavelength ratio of 7.15909 -> 85.909)
                    // Actually, Y should be scaled to 256 elements
                    //           I should be scaled to 100 elements
                    //           Q should be scaled to 36 elements
                    // .. And then all upscaled to xres. But who bothers to do that.
                    const float gamma = 1.8f;
                    int d07=0, d15=0;
                    float sigy1[256*8]; // Original (degraded) signal (non demodulated Y)
                    float sigi1[256*8]; // Demodulated I
                    float sigq1[256*8]; // Demodulated Q

                    // I and Q sinusoids (57 and 147 degrees at colorburst respectively)
                    const float* sigi0 = sinusoids[(8+colorburst-2)%12];
                    const float* sigq0 = sinusoids[(8+colorburst-5)%12];

                    // BEGIN PART: PPU
                    for(int p=0, x=0; x<256; ++x)               // For each pixel,
                    for(int c = line[x]%512, q=0; q<8; ++q,++p) // produce 8 cycles of NTSC signal
                    {
                        // Generate NTSC signal
                        int v = signals[c][ (colorburst+p)%12 ];
                        // Apply slight signal degradation to it
                        v = v - mul / 2;            // Remove DC offset
                        d07 = d07 * 3/10 +  v*7/10; // RC filter for lowpass
                        d15 = d15 *-5/10 + v*15/10; // RC filter with feedback for generating artifacts
                        v = mul/2 + d07*7/10 + d15*3/10; // Combine and re-add DC offset
                        // Save the raw signal.
                        float vv = v/float(mul);
                        sigy1[p] = vv;
                        // END PART: PPU
                        // BEGIN PART: TELEVISION::NTSC DECODER
                        // Demodulate the signal sinusoids matching the colorburst frequency
                        sigi1[p] = vv * sigi0[p%12];
                        sigq1[p] = vv * sigq0[p%12];
                    }
                    for(int x=0; x<xres; ++x)
                    {
                        auto lowpass = [] (float buffer[], int center, int width, const float coeff[]) -> float
                        {
                            int firsts = center-width/2;
                            int firstp = 0;
                            int limitp = width;
                            if(firsts < 0)
                                firstp = -firsts;
                            if(firsts + limitp > 256*8)
                                limitp = 256*8 - firsts;
                            float result = 0.f;
                            for(int p = firstp; p < limitp; ++p)
                                result += buffer[p+firsts] * coeff[p];
                            return result;
                        };
                        // Apply a FIR filter to calculate each of the components from the signal.
                        float y = lowpass(sigy1, x*256*8/xres, Ylow, coeffY);
                        float i = lowpass(sigi1, x*256*8/xres, Ilow, coeffI) * 1.5;
                        float q = lowpass(sigq1, x*256*8/xres, Qlow, coeffQ) * 1.5;
                        //END PART: TELEVISION::NTSC DECODER
                        //BEGIN PART: TELEVISION::RGB SYNTHESIZER 
                        auto gammafix = [=](float f) { return f <= 0.f ? 0.f : std::pow(f, 2.2f / gamma); };

                        target[x].c[0] = 255.9f * gammafix(y +  0.946882f*i +  0.623557f*q);
                        target[x].c[1] = 255.9f * gammafix(y + -0.274788f*i + -0.635691f*q);
                        target[x].c[2] = 255.9f * gammafix(y + -1.108545f*i +  1.709007f*q);
                    }
                    //END PART: TELEVISION::RGB SYNTHESIZER
                    diffs[py] = false;
                }
                std::memcpy(&RGBdata[py], &target, sizeof(target));
            }

            #pragma omp parallel for schedule(static)
            for(unsigned y=0; y< (unsigned) yres; ++y)
            {
                u32* pix = ((u32*) s->pixels) + y*extended_xres + padding_left;
                auto clamp = [](int v) { return v<0 ? 0 : v>255 ? 255 : v; };

                const auto& data = RGB_scaling[y];
                for(unsigned x=0; x< (unsigned) xres; ++x)
                {
                    rgb sample = { {} };
                    int sy = data.start; // range: 0 <= sy < 240*8
                    for(auto d: data.contrib)
                    {
                        const auto pix = RGBdata[sy++ / PixelScaling][x];
                        for(unsigned n=0; n<3; ++n)
                            sample.c[n] += d * pix.c[n];
                    }
                    pix[x] = 0x10000 * clamp(sample.c[0])
                           + 0x00100 * clamp(sample.c[1])
                           + 0x00001 * clamp(sample.c[2]);
                }
            }

            #pragma omp parallel for schedule(dynamic,1) ordered
            for(int c=1; c<=7; ++c)
            {
                // Dimensions of keypad overlay. Location is atop the source code.
                enum { ControllerHeight = (extended_yres-yres) * 30 / 100 };
                enum { ControllerWidth = ControllerHeight * 23 / 10 };

                // Amount of vertical space below the game window.
                enum { TrailerHeight = extended_yres - yres };
                enum { TrailerWidth  = extended_xres - padding_left };

                // Dimensions of the wave data. Location: Right edge.
                enum { WaveHeight = TrailerHeight };
                enum { WaveWidth  = TrailerWidth * 18 / 100};

                enum { VRAMwatchWidth  = 0/*TrailerHeight*/  }; // Make it square
                enum { VRAMwatchHeight = 0/*TrailerHeight*/ }; // Make it square

                enum { MemWatchWidth = TrailerWidth - WaveWidth - VRAMwatchWidth };
                enum { MemWatchHeight = TrailerHeight };

              #pragma omp ordered
              {
                if(c == 5)
                {
                    access.Render( ((u32*) IO::s->pixels) + padding_left + yres*extended_xres,
                                   MemWatchWidth, MemWatchHeight);
                }
                else if(c == 6)
                {
                    char Buf[512];
                    std::sprintf(Buf, "/mnt/te/nestrace-gif/imageframe%06d.gif", framecounter);
                    FILE* fp = 0;
                    for(int tries=0; tries<400; ++tries)
                    {
                        fp = std::fopen(Buf, "rb");
                        if(fp) break;
                        std::perror(Buf);
                        usleep(1000000);
                    }
                    if(fp)
                    {
                        gdImagePtr im = 0;
                        for(int tries=0; tries<400; ++tries)
                        {
                            std::fseek(fp, 0, SEEK_END);
                            int p = std::ftell(fp);
                            if(p < 250000) continue;
                            std::rewind(fp);
                            im = gdImageCreateFromGif(fp);
                            if(im) break;
                            std::fprintf(stderr, "Failed to read GIF\n");
                            usleep(1000000);
                        }
                        std::fprintf(stderr, "GIF successfully opened\n");
                        std::fclose(fp);
                        if(im)
                        {
                            int w = gdImageSX(im), h = gdImageSY(im);
                            gdImagePtr im2 = gdImageCreateTrueColor(w,h);
                            gdImageCopy(im2, im, 0,0, 0,0, w,h);
                            gdImageDestroy(im);

                            for(int y=0; y<h; ++y)
                            {
                                u32* tgt = ((u32*) IO::s->pixels) + y * extended_xres;
                                int* src = &gdImageTrueColorPixel(im2, 0,y); // a macro
                                std::memcpy(tgt, src, w * 4);
                            }
                            gdImageDestroy(im2);
                        }
                    }
                    else
                        std::perror(Buf);
                }
                else if(c == 7)
                {
                    int y1 = extended_yres - ControllerHeight;
                    int width = ControllerWidth;
                    u32* p = (u32*) IO::s->pixels + (padding_left - width) + y1 * extended_xres;
                    RenderControllers(p, width, ControllerHeight);
                }
                /*else if(c == 8)
                {
                    Pak::vram_tracker.Render
                        ( ((u32*) IO::s->pixels) + padding_left + MemWatchWidth + yres*extended_xres,
                          MemWatchWidth, MemWatchHeight);
                }*/
                else
                {
                    int chno = (c-1);
                    int width = WaveWidth;
                    int ydim = WaveHeight;
                    int y1 = chno     * ydim / 4;
                    int y2 = (chno+1) * ydim / 4;
                    int height = y2-y1;
                    y1    += height/10;
                    height = height*8/10;
                    y1 += yres;
                    u32* p = (u32*) IO::s->pixels + (extended_xres - width) + y1 * extended_xres;
                    AudioBuffers[chno].Render(p, width, height);
                }
              }
            }

            if(true) // SCREENSHOT
            {
              gdImagePtr im = gdImageCreateTrueColor(extended_xres, extended_yres);
              for(unsigned y=0; y<extended_yres; ++y)
                  for(unsigned x=0; x<extended_xres; ++x)
                  {
                      u32* p = (u32*) IO::s->pixels;
                      gdImageSetPixel(im, x,y, p[y*extended_xres+x]&0xFFFFFF);
                  }
              char Buf[64];
              sprintf(Buf, "/mnt/te/nestrace-png/tmp-%05d.png", framecounter);
              FILE* fp = fopen(Buf, "wb");
              gdImagePng(im, fp);
              fclose(fp);
              gdImageDestroy(im);
              //exit(0);
            }

            avi.VideoPtr(extended_xres, extended_yres, frame_begin_when, (const unsigned char*) s->pixels);
            frame_begin_when =
                (long double)CycleCounter_FullSeconds
              + (long double)CycleCounter_Fractional / (long double)PPU_CyclesPerSecond;
            if(++framecounter%10 == 0) SDL_Flip(s);
            access.Tick();

        }
        Colorburst = (Colorburst + length*8) % 12;

        if(Skipped_PPU_Cycles > 0)
        {
            int decrement = std::min( (unsigned long long) length, Skipped_PPU_Cycles);
            length             -= decrement;
            Skipped_PPU_Cycles -= decrement;
        }

        CycleCounter_Fractional  += length;
        CycleCounter_FullSeconds += CycleCounter_Fractional / PPU_CyclesPerSecond;
        CycleCounter_Fractional %= PPU_CyclesPerSecond;
    }

    void JoyStrobe(unsigned v)
    {
        if(v) { joydata1[0] = joydata2[0]; joypos[0]=0; }
        if(v) { joydata1[1] = joydata2[1]; joypos[1]=0; }
    }
    u8 JoyRead(unsigned idx)
    {
        static const u8 masks[8] = {0x20,0x10,0x40,0x80,0x04,0x08,0x02,0x01};
        return ((joydata1[idx] & masks[joypos[idx]++ & 7]) ? 1 : 0);
    }
}

namespace CPU
{
    bool reset=true, nmi=false, nmi_edge_detected=false, intr=false;
    template<bool write> u8 MemAccess(u16 addr, u8 v=0);
    u8 RB(u16 addr)      { return MemAccess<0>(addr); }
    u8 WB(u16 addr,u8 v) { return MemAccess<1>(addr, v); }
    void tick();
}

namespace Pak
{
    template<unsigned npages,unsigned char*(&b)[npages], std::vector<u8>& r, unsigned granu>
    static void SetPages(unsigned size, unsigned baseaddr, unsigned index)
    {
        for(unsigned v = r.size() + index * size,
                     p = baseaddr / granu;
                     p < (baseaddr + size) / granu && p < npages;
                     ++p, v += granu)
            b[p] = &r[v % r.size()];
    }
    auto& SetROM  = SetPages< ROM_Pages, banks, ROM, ROM_Granularity>;
    auto& SetVROM = SetPages<VROM_Pages,Vbanks,VRAM,VROM_Granularity>;

    u8 Access(unsigned addr, u8 value, bool write)
    {
        //printf("[%04X]<%02X\n",addr,v);
//        if(write && addr >= 0x6004 && addr < 0x6200 && value) { fputc(value, stderr); }
        //if(addr == 0x02) fprintf(stderr, "[02] <- 0x%02X\n", value);
        //if(addr == 0x03) fprintf(stderr, "[03] <- 0x%02X\n", value);
        //if(addr == 0x4002) fprintf(stderr, "[4002] <- 0x%02X\n", value);
//        if(write && addr == 0x6000) { /*fprintf(stderr, "[6000] <- 0x%02X\n", value);*/ if(value < 0x80) exit(value); }
        //if(addr == 0x6001) fprintf(stderr, "[6001] <- 0x%02X\n", value);
        //if(addr == 0x6002) fprintf(stderr, "[6002] <- 0x%02X\n", value);
        //if(addr == 0x6003) fprintf(stderr, "[6003] <- 0x%02X\n", value);
/**/
        if(write &&  (addr >> 13) == 3 )
        {
            IO::TrackPRAMaccess(addr & 0x1FFF, write, value-PRAM[addr&0x1FFF]);
            return PRAM[addr & 0x1FFF ] = value;
        } // 6000..7FFF range
        //banks[ (addr >> 14) & 3] [addr & 0x3FFF] = value;
        if(write && addr >= 0x8000 && mappernum == 7) // Wizards & Warriors
        {
            SetROM(0x8000, 0x8000, (value&7));
            Nta[0] = Nta[1] = Nta[2] = Nta[3] = &NRAM[0x400 * ((value>>4)&1)];
        }
        if(write && addr >= 0x8000 && mappernum == 2) // Rockman, Castlevania
        {
            //printf("write(%04X,%02X) & %02X\n", addr,value, Read(addr));
            SetROM(0x4000, 0x8000, value);
        }
        if(write && addr >= 0x8000 && mappernum == 3) // Kage, Solomon's Key
        {
            //printf("write(%04X,%02X) & %02X\n", addr,value, Read(addr));
            value &= Access(addr,0,false); // Simulate bus conflict
            SetVROM(0x2000, 0x0000, (value&3));
        }
        if(write && addr >= 0x8000 && mappernum == 1) // Rockman 2, Simon's Quest
        {
            static u8 regs[4]={0x0C,0,0,0}, counter=0, cache=0;
            if(value & 0x80) { regs[0]=0x0C; goto configure; }
            cache |= (value&1) << counter;
            if(++counter == 5)
            {
                regs[ (addr>>13) & 3 ] = value = cache;
            configure:
                cache = counter = 0;
                static const u8 sel[4][4] = { {0,0,0,0}, {1,1,1,1}, {0,1,0,1}, {0,0,1,1} };
                for(unsigned m=0; m<4; ++m) Nta[m] = &NRAM[0x400 * sel[regs[0]&3][m]];
                SetVROM(0x1000, 0x0000, ((regs[0]&16) ? regs[1] : ((regs[1]&~1)+0)));
                SetVROM(0x1000, 0x1000, ((regs[0]&16) ? regs[2] : ((regs[1]&~1)+1)));
                switch( (regs[0]>>2)&3 )
                {
                    case 0: case 1:
                        SetROM(0x8000, 0x8000, (regs[3] & 0xE) / 2);
                        break;
                    case 2:
                        SetROM(0x4000, 0x8000, 0);
                        SetROM(0x4000, 0xC000, (regs[3] & 0xF));
                        break;
                    case 3:
                        SetROM(0x4000, 0x8000, (regs[3] & 0xF));
                        SetROM(0x4000, 0xC000, ~0);
                        break;
                }
            }
        }
        if(write && addr >= 0x8000 && mappernum == 4) // MMC3 (complicated!)
        {
            static u8 reg0 = 0, reg1[8] = {0}, reg2 = 0, reg3 = 0, reload = 0, counter = 0, irqenable = 0;
            static u16 last_ppu_addr = 0, a12_low_count = 0;
            if(addr & 0x10000)
            {
                if(((addr ^ last_ppu_addr) & addr & 0x1000) && a12_low_count >= 8)
                {
                    if(!counter--) counter = reload;
                    else if(irqenable) CPU::intr = true;
                }
                last_ppu_addr = addr;
                if(addr & 0x1000) a12_low_count = 0; else ++a12_low_count;
                return 0;
            }
            /*fprintf(stderr, "MMC3 write %02X to %04X = register %u\n",
                value, addr, (addr & 1) + ((addr >> 12) & 6));*/
            switch((addr & 1) + ((addr >> 12) & 6))
            {
                case 0: reg0 = value; break; // bank select
                case 1: reg1[reg0 & 7] = value; break; // bank data
                case 2: reg2 = value; break; // mirroring, &1=horizontal (0=vertical)
                case 3: reg3 = value; break; // &0x40=prg ram protect, &0x80=enable chip
                case 4: reload = value; break; // irq latch (irq reload value)
                case 5: counter = 0; break; // irq reload (at next tick, counter will be reloaded)
                case 6: irqenable = 0; CPU::intr = false; break; // irq disable & acknowledge any irq
                case 7: irqenable = 1; break; // irq enable
            }
            for(unsigned m=0; m<4; ++m) Nta[m] = &NRAM[0x400 * (m & ((reg2&1) ? 2 : 1))];
            for(unsigned n=0; n<2; ++n) SetVROM(0x0800, ((reg0&0x80)?0x1000:0x0000) + n*0x800, reg1[0+n]/2);
            for(unsigned n=0; n<4; ++n) SetVROM(0x0400, ((reg0&0x80)?0x0000:0x1000) + n*0x400, reg1[2+n]);
            SetROM(0x2000, ((reg0&0x40)?0xC000:0x8000), reg1[6]);
            SetROM(0x2000, 0xA000, reg1[7]);
            SetROM(0x2000, ((reg0&0x40)?0x8000:0xC000), -2);
        }
        for(unsigned n=0; n<ROM_Pages; ++n)
            if(banks[n] > &ROM[ROM.size()-ROM_Granularity])
                fprintf(stderr, "ERROR: By write(%04X,%02X), bank %u became pointer to %06zX..%06zX, while ROM limit is %06zX\n",
                    addr, value, n,
                    banks[n]-&ROM[0],
                    banks[n]-&ROM[0]+ROM_Granularity-1,
                    ROM.size());
        for(unsigned n=0; n<VROM_Pages; ++n)
            if(Vbanks[n] > &VRAM[VRAM.size()-VROM_Granularity])
                fprintf(stderr, "ERROR: By write(%04X,%02X), V-bank %u became pointer to %06zX..%06zX, while limit is %06zX\n",
                    addr, value, n,
                    Vbanks[n]-&VRAM[0],
                    Vbanks[n]-&VRAM[0]+VROM_Granularity,
                    VRAM.size());
        for(unsigned n=0; n<4; ++n)
            if(Nta[n] > &NRAM[0x2000-0x400])
                fprintf(stderr, "ERROR: By write(%04X,%02X), NTA %u became pointer to %04X..%04X, while NTA limit is %04X\n",
                    addr, value, n,
                    (unsigned) (0x0000+Nta[n]-&NRAM[0]),
                    (unsigned) (0x3FFF+Nta[n]-&NRAM[0]),
                    (unsigned) sizeof(NRAM));
        if( (addr >> 13) == 3 )
        {
            IO::TrackPRAMaccess(addr & 0x1FFF, false, 0);
            return PRAM[addr & 0x1FFF ];
        }
        if(!write)
        {
            const u8& b = banks[ (addr >> 14) & 3] [addr & 0x3FFF];
            IO::TrackROMaccess(&b - &ROM[0]);
        }
        return banks[ (addr / ROM_Granularity) % ROM_Pages] [addr % ROM_Granularity];
    }
    void Init(unsigned mirroring)
    {
        for(unsigned v=0; v<4; ++v) Nta[v] = &NRAM[0x400 * ((mirroring >> (v*4)) & 0xF) ];
        for(unsigned v=0; v<4; ++v) SetROM(0x4000, v*0x4000, v==3 ? -1 : 0);
        SetVROM(0x2000, 0x0000, 0);
    }
}

namespace PPU
{
    union regtype // PPU register file
    {
        u32 value;
        // Reg0 (write)             // Reg1 (write)             // Reg2 (read)
        RegBit<0,8,u32> sysctrl;    RegBit< 8,8,u32> dispctrl;  RegBit<16,8,u32> status;
        RegBit<0,2,u32> BaseNTA;    RegBit< 8,1,u32> Grayscale; RegBit<21,1,u32> SPoverflow;
        RegBit<2,1,u32> Inc;        RegBit< 9,1,u32> ShowBG8;   RegBit<22,1,u32> SP0hit;
        RegBit<3,1,u32> SPaddr;     RegBit<10,1,u32> ShowSP8;   RegBit<23,1,u32> InVBlank;
        RegBit<4,1,u32> BGaddr;     RegBit<11,1,u32> ShowBG;    // Reg3 (write)
        RegBit<5,1,u32> SPsize;     RegBit<12,1,u32> ShowSP;    RegBit<24,8,u32> OAMaddr;
        RegBit<6,1,u32> SlaveFlag;  RegBit<11,2,u32> ShowBGSP;  RegBit<24,2,u32> OAMdata;
        RegBit<7,1,u32> NMIenabled; RegBit<13,3,u32> EmpRGB;    RegBit<26,6,u32> OAMindex;
    } reg;
    // Raw memory data as read&written by the game
    u8 banks[2][0x1000], palette[32], OAM[256];
    // Decoded sprite information, used & changed during each scanline
    struct { u8 sprindex, y, index, attr, x; u16 pattern; } OAM2[8], OAM3[8];

    union scrolltype
    {
        RegBit<3,16,u32> raw;       // raw VRAM address (16-bit)
        RegBit<0, 8,u32> xscroll;   // low 8 bits of first write to 2005
        RegBit<0, 3,u32> xfine;     // low 3 bits of first write to 2005
        RegBit<3, 5,u32> xcoarse;   // high 5 bits of first write to 2005
        RegBit<8, 5,u32> ycoarse;   // high 5 bits of second write to 2005
        RegBit<13,2,u32> basenta;   // name-table index (copied from 2000)
        RegBit<13,1,u32> basenta_h; // horizontal nametable index
        RegBit<14,1,u32> basenta_v; // vertical   nametable index
        RegBit<15,3,u32> yfine;     // low 3 bits of first write to 2005
        RegBit<11,8,u32> vaddrhi;   // first write to 2006 (with high 2 bits set to zero)
        RegBit<3, 8,u32> vaddrlo;   // second write to 2006
    } scroll, vaddr;

    unsigned pat_addr, sprinpos, sproutpos, sprrenpos, sprtmp;
    u16 tileattr, tilepat, ioaddr;
    u32 bg_shift_pat, bg_shift_attr;

    int scanline=241, x=0, scanline_end=341;
    bool even_odd_toggle=false, offset_toggle=false;
    u8 read_buffer, open_bus;
    int VBlankState = 0;
    unsigned open_bus_decay_timer=0;

    u8& mmap(int i)
    {
        i &= 0x3FFF;
        if(i >= 0x3F00) { if(i%4==0) i &= 0x0F; return palette[i & 0x1F]; }
        if(Pak::mappernum == 4)
            Pak::Access(0x10000|i, 0, true); // For IRQ generation
        if(i < 0x2000) return Pak::Vbanks[(i / Pak::VROM_Granularity) % Pak::VROM_Pages]
                                         [i % Pak::VROM_Granularity];
        return Pak::Nta[   (i>>10)&3][i&0x3FF];
    }
    u8 RefreshOpenBus(u8 v)
    {
        open_bus             = v;
        open_bus_decay_timer = 77777;
        return v;
    }

    // External I/O: read or write
    u8 Access(u16 index, u8 v, bool write)
    {
        //if(write)
        //{
            //unsigned char c = u8(written);
            //if(c < ' ' || (c >= 0x7F && c < 0xA0)) c = '?';
            //printf("Wrote 20%02X as %02X ('%c') at scanline=%d,x=%u\n", index, u8(written), c, scanline,x);
        //}
        u8 res = open_bus;
        if(write) RefreshOpenBus(v);
        switch(index) // Which port from $200x?
        {
            case 0: if(write) { reg.sysctrl  = v; scroll.basenta = reg.BaseNTA; } break;
            case 1: if(write) { reg.dispctrl = v; } break;
            case 2: if(write) break;
                    res = reg.status | (open_bus & 0x1F);
                    reg.InVBlank = false;  // Reading $2002 clears the vblank flag.
                    offset_toggle = false; // Also resets the toggle for address updates.
                    if(VBlankState != -5)
                        VBlankState = 0; // This also may cancel the setting of InVBlank.
                    break;
            case 3: if(write) reg.OAMaddr        = v; break; // Index into Object Attribute Memory
            case 4: if(write) OAM[reg.OAMaddr++] = v; // Write or read the OAM (sprites).
                    else res = RefreshOpenBus(OAM[reg.OAMaddr] & (reg.OAMdata==2 ? 0xE3 : 0xFF));
                    break;
            case 5: if(!write) break; // Set background scrolling offset
                if(offset_toggle) { scroll.yfine   = v & 7; scroll.ycoarse = v >> 3; }
                else              { scroll.xscroll = v; }
                offset_toggle = !offset_toggle;
                break;
            case 6: if(!write) break; // Set video memory position for reads/writes
                if(offset_toggle) { scroll.vaddrlo = v; vaddr.raw = (unsigned) scroll.raw; }
                else              { scroll.vaddrhi = v & 0x3F; }
                offset_toggle = !offset_toggle;
                break;
            case 7:
                res = read_buffer;
                u8& t = mmap(vaddr.raw); // Access the video memory.
                if(write) res = t = v;
                else { if((vaddr.raw & 0x3F00) == 0x3F00) // palette?
                          res = read_buffer = (open_bus & 0xC0) | (t & 0x3F);
                       read_buffer = t; }
                RefreshOpenBus(res);
                vaddr.raw += reg.Inc ? 32 : 1; // The address is automatically updated.
                break;
        }
        //printf("Reading back 20%02X as %02X, at scanline=%d,x=%u\n", index, res, scanline,x);
        return res;
    }
    void rendering_tick()
    {
        bool tile_decode_mode = 0x10FFFF & (1u << (x/16)); // 0..255, 320..335

        // Each action happens in two steps: 1) select memory address; 2) receive data and react on it.
        switch(x % 8)
        {
            case 2:
                ioaddr = 0x23C0 + 0x400*vaddr.basenta + 8*(vaddr.ycoarse/4) + (vaddr.xcoarse/4);
                if(tile_decode_mode) break;
            case 0:
                ioaddr = 0x2000 + (vaddr.raw & 0xFFF);
                // Reset sprite data
                if(x == 0) { sprinpos = sproutpos = 0; if(reg.ShowSP) reg.OAMaddr = 0; }
                if(!reg.ShowBG) break;
                // Reset scrolling (vertical once, horizontal each scanline)
                if(x == 304 && scanline == -1) vaddr.raw = (unsigned) scroll.raw;
                if(x == 256) { vaddr.xcoarse   = (unsigned)scroll.xcoarse;
                               vaddr.basenta_h = (unsigned)scroll.basenta_h;
                               sprrenpos = 0; }
                break;
            case 1:
                if(x == 337 && scanline == -1 && even_odd_toggle && reg.ShowBG) scanline_end = 340;
                // Name table access
                pat_addr = 0x1000*reg.BGaddr + 16*mmap(ioaddr) + vaddr.yfine;
                if(!tile_decode_mode) break;
                // Push the current tile into shift registers.
                // The bitmap pattern is 16 bits, while the attribute is 2 bits, repeated 8 times.
                bg_shift_pat  = (bg_shift_pat  >> 16) + 0x00010000 * tilepat;
                bg_shift_attr = (bg_shift_attr >> 16) + 0x55550000 * tileattr;
                break;
            case 3:
                // Attribute table access
                // Increment the X and Y coordinates to the display memory
                if(tile_decode_mode)
                {
                    tileattr = (mmap(ioaddr) >> ((vaddr.xcoarse&2) + 2*(vaddr.ycoarse&2))) & 3;
                    // Go to the next tile horizontally (and switch nametable if it wraps)
                    if(!++vaddr.xcoarse) { vaddr.basenta_h ^= 1; }
                    // At the edge of the screen, do the same but vertically
                    if(x==251 && !++vaddr.yfine && ++vaddr.ycoarse == 30)
                        { vaddr.ycoarse = 0; vaddr.basenta_v ^= 1; }
                }
                else if(sprrenpos < sproutpos)
                {
                    // Select sprite pattern instead of background pattern
                    auto& o = OAM3[sprrenpos]; // Sprite to render on next scanline
                    memcpy(&o, &OAM2[sprrenpos], sizeof(o));
                    unsigned y = (scanline) - o.y;
                    if(o.attr & 0x80) y ^= (reg.SPsize ? 15 : 7);
                    pat_addr = 0x1000 * (reg.SPsize ? (o.index & 0x01) : reg.SPaddr);
                    pat_addr +=  0x10 * (reg.SPsize ? (o.index & 0xFE) : (o.index & 0xFF));
                    pat_addr += (y&7) + (y&8)*2;
                }
                break;
            // Pattern table bytes
            case 5: tilepat = mmap(pat_addr|0); break;
            case 7: // Interleave the bits of the two pattern bytes
                unsigned p = tilepat | (mmap(pat_addr|8) << 8);
                p = (p&0xF00F) | ((p&0x0F00)>>4) | ((p&0x00F0)<<4);
                p = (p&0xC3C3) | ((p&0x3030)>>2) | ((p&0x0C0C)<<2);
                p = (p&0x9999) | ((p&0x4444)>>1) | ((p&0x2222)<<1);
                tilepat = p;
                // When decoding sprites, save the sprite graphics and move to next sprite
                if(!tile_decode_mode && sprrenpos < sproutpos)
                    OAM3[sprrenpos++].pattern = tilepat;
                break;
        }
        // Find which sprites are visible on next scanline (TODO: implement crazy 9-sprite malfunction)
        switch(x>=64 && x<256 && x%2 ? (reg.OAMaddr++ & 3) : 4)
        {
            default:
                // Access OAM memory
                sprtmp = OAM[reg.OAMaddr];
                break;
            case 0:
                if(sprinpos >= 64) { reg.OAMaddr=0; break; }
                ++sprinpos; // next sprite
                if(sproutpos<8) OAM2[sproutpos].y        = sprtmp;
                if(sproutpos<8) OAM2[sproutpos].sprindex = reg.OAMindex;
               {int y1 = sprtmp, y2 = sprtmp + (reg.SPsize?16:8);
                if(!( scanline >= y1 && scanline < y2 ))
                    reg.OAMaddr = sprinpos != 2 ? reg.OAMaddr+3 : 8;}
                break;
            case 1: if(sproutpos<8) OAM2[sproutpos].index = sprtmp; break;
            case 2: if(sproutpos<8) OAM2[sproutpos].attr  = sprtmp; break;
            case 3:
                //printf("Sprite %u (%u) saved\n", sproutpos, OAM2[sproutpos].sprindex);
                if(sproutpos<8) OAM2[sproutpos].x = sprtmp;
                if(sproutpos<8) ++sproutpos; else reg.SPoverflow = true;
                if(sprinpos == 2) reg.OAMaddr = 8;
        }
    }
    void render_pixel()
    {
        bool edge   = 0x80000001u & (1u << (x/8));  // 0..7, 248..255
        bool showbg = reg.ShowBG && (!edge || reg.ShowBG8);
        bool showsp = reg.ShowSP && (!edge || reg.ShowSP8);

        // Render the background
        unsigned fx = scroll.xfine, xpos = 15 - (( (x&7) + fx + 8*!!(x&7) ) & 15);

        unsigned pixel = 0, attr = 0;
        if(showbg) // Pick a pixel from the shift registers
        {
            pixel = (bg_shift_pat  >> (xpos*2)) & 3;
            attr  = (bg_shift_attr >> (xpos*2)) & (pixel ? 3 : 0);
        }
        else if( (vaddr.raw & 0x3F00) == 0x3F00 && !reg.ShowBGSP )
            pixel = vaddr.raw;

        // Overlay the sprites
        if(showsp)
            for(unsigned sno=0; sno<sprrenpos; ++sno)
            {
                auto& s = OAM3[sno];
                // Check if this sprite is horizontally in range
                unsigned xdiff = x - s.x;
                if(xdiff >= 8) continue; // Also matches negative values
                // Determine which pixel to display; skip transparent pixels
                if(!(s.attr & 0x40)) xdiff = 7-xdiff;
                u8 spritepixel = (s.pattern >> (xdiff*2)) & 3;
                if(!spritepixel) continue;
                // Register sprite-0 hit if applicable
                if(x < 255 && pixel && s.sprindex == 0) reg.SP0hit = true;
                // Render the pixel unless behind-background placement wanted
                if(!(s.attr & 0x20) || !pixel)
                {
                    attr = (s.attr & 3) + 4;
                    pixel = spritepixel;
                }
                // Only process the first non-transparent sprite pixel.
                break;
            }
        pixel = palette[ (attr*4 + pixel) & 0x1F ] & (reg.Grayscale ? 0x30 : 0x3F);
        IO::PutPixel(x, scanline, pixel | (reg.EmpRGB << 6));
    }

    void tick()
    {
        // Set/clear vblank where needed
        switch(VBlankState)
        {
            case -5: reg.status = 0; break;
            case 2: reg.InVBlank = true; break;
            case 0: CPU::nmi = reg.InVBlank && reg.NMIenabled; break;
        }
        if(VBlankState != 0) VBlankState += (VBlankState < 0 ? 1 : -1);
        if(open_bus_decay_timer) if(!--open_bus_decay_timer) open_bus = 0;

        // Graphics processing scanline?
        if(scanline < 240)
        {
            /* Process graphics for this cycle */
            if(reg.ShowBGSP) rendering_tick();
            if(scanline >= 0 && x < 256) render_pixel();
        }

        // Done with the cycle. Check for end of scanline.
        if(++x >= scanline_end)
        {
            // Begin new scanline
            IO::FlushScanline(scanline, scanline_end);
            scanline_end = 341;
            x            = 0;
            // Does something special happen on the new scanline?
            switch(scanline += 1)
            {
                case 261: // Begin of rendering
                    scanline = -1; // pre-render line
                    even_odd_toggle = !even_odd_toggle;
                    // Clear vblank flag
                    VBlankState = -5;
                    break;
                case 241: // Begin of vertical blanking
                    // I cheat here: I did not bother to learn how to use SDL events,
                    // so I simply read button presses from a movie file, which happens
                    // to be a TAS, rather than from the keyboard or from a joystick.
                    static FILE* fp = fopen("tmp.fmv", "rb");
                    if(fp)
                    {
                        static unsigned ctrlmask = 0;
                        if(!ftell(fp))
                        {
                            fseek(fp, 0x05, SEEK_SET);
                            ctrlmask = fgetc(fp);
                            fseek(fp, 0x90, SEEK_SET);
                        }
                        if(ctrlmask & 0x80) { IO::joydata2[0] = fgetc(fp); if(feof(fp)) IO::joydata2[0] = 0; }
                        if(ctrlmask & 0x40) { IO::joydata2[1] = fgetc(fp); if(feof(fp)) IO::joydata2[1] = 0; }
                    }
                    // Set vblank flag
                    VBlankState = 2;
            }
        }
    }
}

namespace APU
{
    static const u8 LengthCounters[32] = { 10,254,20, 2,40, 4,80, 6,160, 8,60,10,14,12,26,14,
                                           12, 16,24,18,48,20,96,22,192,24,72,26,16,28,32,30 };
    static const u16 NoisePeriods[16] = { 2,4,8,16,32,48,64,80,101,127,190,254,381,508,1017,2034 };
    static const u16 DMCperiods[16] = { 428,380,340,320,286,254,226,214,190,160,142,128,106,84,72,54 };

    bool FiveCycleDivider = false, IRQdisable = true, ChannelsEnabled[5] = { false };
    bool PeriodicIRQ = false, DMC_IRQ = false;
    bool count(int& v, int reset) { return --v < 0 ? (v=reset),true : false; }

    struct channel
    {
        union
        {
            // 4000, 4004, 400C, 4012:            // 4001, 4005, 4013:            // 4002, 4006, 400A, 400E:          
            RegBit<0,8,u32> reg0;                 RegBit< 8,8,u32> reg1;          RegBit<16,8,u32> reg2;
            RegBit<6,2,u32> DutyCycle;            RegBit< 8,3,u32> SweepShift;    RegBit<16,4,u32> NoiseFreq;
            RegBit<4,1,u32> EnvDecayDisable;      RegBit<11,1,u32> SweepDecrease; RegBit<23,1,u32> NoiseType;
            RegBit<0,4,u32> EnvDecayRate;         RegBit<12,3,u32> SweepRate;     RegBit<16,11,u32> WaveLength;
            RegBit<5,1,u32> EnvDecayLoopEnable;   RegBit<15,1,u32> SweepEnable;   // 4003, 4007, 400B, 400F, 4010:    
            RegBit<0,4,u32> FixedVolume;          RegBit< 8,8,u32> PCMlength;     RegBit<24,8,u32> reg3;
            RegBit<5,1,u32> LengthCounterDisable;                                 RegBit<27,5,u32> LengthCounterInit;
            RegBit<0,7,u32> LinearCounterInit;                                    RegBit<30,1,u32> LoopEnabled;
            RegBit<7,1,u32> LinearCounterDisable;                                 RegBit<31,1,u32> IRQenable;
        } reg;
        int length_counter, linear_counter, address, envelope;
        int sweep_delay, env_delay, wave_counter, hold, phase, level;

        // Function for updating the wave generators and taking the sample for each channel.
        template<unsigned c>
        int tick()
        {
            channel& ch = *this;
            if(!ChannelsEnabled[c]) return c==4 ? 64 : 8;
            int wl = (ch.reg.WaveLength+1) * (c >= 2 ? 1 : 2);
            if(c == 3) wl = NoisePeriods[ ch.reg.NoiseFreq ];
            int volume = ch.length_counter ? ch.reg.EnvDecayDisable ? ch.reg.FixedVolume : ch.envelope : 0;
            // Sample may change at wavelen intervals.
            auto& S = ch.level;
            if(!count(ch.wave_counter, wl)) return S;
            switch(c)
            {
                default:// Square wave. With four different 8-step binary waveforms (32 bits of data total).
                    if(wl < 8) return S = 8;
                    return S = (0x9F786040u & (1u << (ch.reg.DutyCycle * 8 + ch.phase++ % 8))) ? volume : 0;

                case 2: // Triangle wave
                    if(!ch.length_counter || !ch.linear_counter || wl < 3) return S = 8;
                    ++ch.phase;
                    return S = (ch.phase & 15) ^ ((ch.phase & 16) ? 15 : 0);

                case 3: // Noise: Linear feedback shift register
                    if(!ch.hold) ch.hold = 1;
                    ch.hold = (ch.hold >> 1)
                          | (((ch.hold ^ (ch.hold >> (ch.reg.NoiseType ? 6 : 1))) & 1) << 14);
                    return S = (ch.hold & 1) ? 0 : volume;

                case 4: // Delta modulation channel (DMC)
                    // hold = 8 bit value, phase = number of bits buffered
                    if(ch.phase == 0) // Nothing in sample buffer?
                    {
                        if(!ch.length_counter && ch.reg.LoopEnabled) // Loop?
                        {
                            ch.length_counter = ch.reg.PCMlength*16 + 1;
                            ch.address        = (ch.reg.reg0 | 0x300) << 6;
                        }
                        if(ch.length_counter > 0) // Load next 8 bits if available
                        {
                            // Note: Re-entrant! But not recursive, because even
                            // the shortest wave length is greater than the read time.
                            // TODO: proper clock
                            if(ch.reg.WaveLength>20)
                                for(unsigned t=0; t<3; ++t) CPU::RB(u16(ch.address) | 0x8000); // timing
                            //CPU::tick(); // Load address
                            ch.hold  = CPU::RB(u16(ch.address++) | 0x8000); // Fetch byte
                            ch.phase = 8;
                            --ch.length_counter;
                        }
                        else // Otherwise, disable channel or issue IRQ
                            ChannelsEnabled[4] = ch.reg.IRQenable && (CPU::intr = DMC_IRQ = true);
                    }
                    if(ch.phase != 0) // Update the signal if sample buffer nonempty
                    {
                        int v = ch.linear_counter;
                        if(ch.hold & (0x80 >> --ch.phase)) v += 2; else v -= 2;
                        if(v >= 0 && v <= 0x7F) ch.linear_counter = v;
                    }
                    return S = ch.linear_counter;
            }
        }
    } chn[5] = { };

    struct { short lo, hi; } hz240counter = { 0,0 };

    void Write(u8 index, u8 value)
    {
        channel& ch = chn[(index/4) % 5];
        switch(index<0x10 ? index%4 : index)
        {
            case 0: ch.reg.reg0 = value; break;
            case 1: ch.reg.reg1 = value; ch.sweep_delay = ch.reg.SweepRate; break;
            case 2: ch.reg.reg2 = value; break;
            case 3:
                ch.reg.reg3 = value;
                if(ChannelsEnabled[index/4])
                    ch.length_counter = LengthCounters[ch.reg.LengthCounterInit];
                ch.linear_counter = ch.reg.LinearCounterInit;
                ch.env_delay      = ch.reg.EnvDecayRate;
                ch.envelope       = 15;
                if(index < 8) ch.phase = 0;
                break;
            case 0x10: ch.reg.reg3 = value; ch.reg.WaveLength = DMCperiods[value&0x0F]; break;
            case 0x11: ch.linear_counter = value & 0x7F; break; // dac value
            case 0x12: ch.reg.reg0 = value; ch.address = (ch.reg.reg0 | 0x300) << 6; break;
            case 0x13: ch.reg.reg1 = value; ch.length_counter = ch.reg.PCMlength*16 + 1; break; // sample length
            case 0x15:
                for(unsigned c=0; c<5; ++c)
                    ChannelsEnabled[c] = value & (1 << c);
                for(unsigned c=0; c<5; ++c)
                    if(!ChannelsEnabled[c])
                        chn[c].length_counter = 0;
                    else if(c == 4 && chn[c].length_counter == 0)
                        chn[c].length_counter = ch.reg.PCMlength*16 + 1;
                break;
            case 0x17:
                FiveCycleDivider = value & 0x80;
                IRQdisable       = value & 0x40;
                hz240counter     = { 0,0 };
                if(IRQdisable) PeriodicIRQ = DMC_IRQ = false;
        }
    }
    u8 Read()
    {
        u8 res = 0;
        for(unsigned c=0; c<5; ++c) res |= (chn[c].length_counter ? 1 << c : 0);
        if(PeriodicIRQ) res |= 0x40; PeriodicIRQ = false;
        if(DMC_IRQ)     res |= 0x80; DMC_IRQ     = false;
        CPU::intr = false;
        return res;
    }

    void tick() // Invoked at CPU's rate.
    {
        // Divide CPU clock by 7457.5 to get a 240 Hz, which controls certain events.
        if((hz240counter.lo += 2) >= 14915)
        {
            hz240counter.lo -= 14915;
            if(++hz240counter.hi >= 4+FiveCycleDivider) hz240counter.hi = 0;

            // 60 Hz interval: IRQ. IRQ is not invoked in five-cycle mode (48 Hz).
            if(!IRQdisable && !FiveCycleDivider && hz240counter.hi==0)
                CPU::intr = PeriodicIRQ = true;

            // Some events are invoked at 96 Hz or 120 Hz rate. Others, 192 Hz or 240 Hz.
            bool HalfTick = (hz240counter.hi&5)==1, FullTick = hz240counter.hi < 4;
            for(unsigned c=0; c<4; ++c)
            {
                channel& ch = chn[c];
                //if(!ChannelsEnabled[c]) continue;
                int wl = ch.reg.WaveLength;

                // Length tick (all channels except DMC, but different disable bit for triangle wave)
                if(HalfTick && ch.length_counter
                && !(c==2 ? ch.reg.LinearCounterDisable : ch.reg.LengthCounterDisable))
                    ch.length_counter -= 1;

                // Sweep tick (square waves only)
                if(HalfTick && c < 2 && count(ch.sweep_delay, ch.reg.SweepRate))
                    if(wl >= 8 && ch.reg.SweepEnable && ch.reg.SweepShift)
                    {
                        int s = wl >> ch.reg.SweepShift, d[4] = {s, s, ~s, -s};
                        wl += d[ch.reg.SweepDecrease*2 + c];
                        if(wl < 0x800) ch.reg.WaveLength = wl;
                    }

                // Linear tick (triangle wave only)
                if(FullTick && c == 2)
                    ch.linear_counter = ch.reg.LinearCounterDisable
                    ? ch.reg.LinearCounterInit
                    : (ch.linear_counter > 0 ? ch.linear_counter - 1 : 0);

                // Envelope tick (square and noise channels)
                if(FullTick && c != 2 && count(ch.env_delay, ch.reg.EnvDecayRate))
                    if(ch.envelope > 0 || ch.reg.EnvDecayLoopEnable)
                        ch.envelope = (ch.envelope-1) & 15;
            }
        }

        // Mix the audio
        int s[5] = { chn[0].tick<0>(), chn[1].tick<0>(), // Note: The second 0 is intentional.
                     chn[2].tick<2>(), chn[3].tick<3>(), //       It identifies channel type. Not index.
                     chn[4].tick<4>() };
        IO::AudioRendering(s);
        auto v = [](float m,float n, float d) { return n!=0.f ? m/n : d; };
        short sample = 30000 *
          (v(95.88f,  (100.f + v(8128.f, s[0] + s[1], -100.f)), 0.f)
        +  v(159.79f, (100.f + v(1.0, s[2]/8227.f + s[3]/12241.f + s[4]/22638.f, -100.f)), 0.f)
          - 0.5f
          );

        // I cheat here: I did not bother to learn how to use SDL mixer, let alone use it in <5 lines of code,
        // so I simply use a combination of external programs for outputting the audio.
        // Hooray for Unix principles! A/V sync will be ensured in post-process.

        static const unsigned short AudioBufferSize = 20000;
        static short AudioBuffer[AudioBufferSize], AudioBufferPos = 0;
        static unsigned skip = 0;
        if((skip += 10) >= TurboLevel)
        {
            skip -= TurboLevel;
            AudioBuffer[AudioBufferPos++] = sample;
        }
        else
        {
            Skipped_PPU_Cycles += 3;
        }
        if(AudioBufferPos == AudioBufferSize)
        {
            avi.AudioPtr(PPU_CyclesPerSecond / 3, 16, 1,
                         (const unsigned char*)&AudioBuffer[0], AudioBufferSize);
            AudioBufferPos = 0;
        }

        static FILE* fp = fopen("audio.dat", "w");
                          //popen("resample mr1789800 r48000 | aplay -Dhw:1 -fdat 2>/dev/null", "w");
        fputc(sample, fp);
        fputc(sample/256, fp);
    }
}

namespace CPU
{
    // CPU registers:
    u16 PC=0xC000;
    u8 RAM[0x800], A=0,X=0,Y=0,S=0;
    union /* Status flags: */
    {
        u8 raw;
        RegBit<0> C; // carry
        RegBit<1> Z; // zero
        RegBit<2> I; // interrupt enable/disable
        RegBit<3> D; // decimal mode (unsupported on NES, but flag exists)
        // 4,5 (0x10,0x20) don't exist
        RegBit<6> V; // overflow
        RegBit<7> N; // negative
    } P;

//#include "dasm.hh"

    void tick()
    {
        // nestest.nes compatibility: No clock for PPU while reset is being signalled
        if(reset) return;
        // PPU clock: 3 times the CPU rate
        for(unsigned n=0; n<3; ++n) PPU::tick();
        // APU clock: 1 times the CPU rate
        for(unsigned n=0; n<1; ++n) APU::tick();
    }

    template<bool write> u8 MemAccess(u16 addr, u8 v)
    {
        // Memory writes are turned into reads while reset is being signalled
        if(reset && write) return MemAccess<0>(addr);

        tick();
        // Map the memory from CPU's viewpoint.
        /**/ if(addr < 0x2000)
        {
            IO::TrackRAMaccess(addr & 0x7FF, write, v-RAM[addr&0x7FF]);
            u8& r = RAM[addr & 0x7FF];
            if(!write)return r; r=v;
        }
        else if(addr < 0x4000) return PPU::Access(addr&7, v, write);
        else if(addr < 0x4018)
            switch(addr & 0x1F)
            {
                case 0x14: // OAM DMA: Copy 256 bytes from RAM into PPU's sprite memory
                    if(write) for(unsigned b=0; b<256; ++b) WB(0x2004, RB((v&7)*0x0100+b));
                    return 0;
                case 0x15: if(!write) return APU::Read();    APU::Write(0x15,v); break;
                case 0x16: if(!write) return IO::JoyRead(0); IO::JoyStrobe(v); break;
                case 0x17: if(!write) return IO::JoyRead(1); // write:passthru
                default: if(!write) break;
                         APU::Write(addr&0x1F, v);
            }
        else return Pak::Access(addr, v, write);
        return 0;
    }

    u16 wrap(u16 oldaddr, u16 newaddr)  { return (oldaddr & 0xFF00) + u8(newaddr); }
    void Misfire(u16 old, u16 addr) { u16 q = wrap(old, addr); if(q != addr) RB(q); }
    u8   Pop()        { return RB(0x100 | u8(++S)); }
    void Push(u8 v)   { WB(0x100 | u8(S--), v); }

    template<u16 op>
    void Ins()
    {
        // Note: op 0x100 means "NMI", 0x101 means "Reset", 0x102 means "IRQ". They are implemented in terms of "BRK".
        // User is responsible for ensuring that WB() will not store into memory while Reset is being processed.
        unsigned addr=0, d=0, t=0xFF, c=0, sb=0, pbits = op<0x100 ? 0x30 : 0x20;

        // Define the opcode decoding matrix, which decides which micro-operations constitute
        // any particular opcode. (Note: The PLA of 6502 works on a slightly different principle.)
        enum { o8 = op/8, o8m = 1 << (op%8) };
        // Fetch op'th item from a bitstring encoded in a data-specific variant of base64,
        // where each character transmits 8 bits of information rather than 6.
        // This peculiar encoding was chosen to reduce the source code size.
        #define t(s,code) { enum { \
            i=o8m & (s[o8]>90 ? (130+" (),-089<>?BCFGHJLSVWZ[^hlmnxy|}"[s[o8]-94]) \
                              : (s[o8]-" (("[s[o8]/39])) }; if(i) { code; } }

        /* Decode address operand */
        t("                                !", addr = 0xFFFA) // NMI vector location
        t("                                *", addr = 0xFFFC) // Reset vector location
        t("!                               ,", addr = 0xFFFE) // Interrupt vector location
        t("zy}z{y}zzy}zzy}zzy}zzy}zzy}zzy}z ", addr = RB(PC++))
        t("2 yy2 yy2 yy2 yy2 XX2 XX2 yy2 yy ", d = X) // register index
        t("  62  62  62  62  om  om  62  62 ", d = Y)
        t("2 y 2 y 2 y 2 y 2 y 2 y 2 y 2 y  ", addr=u8(addr+d); d=0; tick())              // add zeropage-index
        t(" y z!y z y z y z y z y z y z y z ", addr=u8(addr);   addr+=256*RB(PC++))       // absolute address
        t("3 6 2 6 2 6 286 2 6 2 6 2 6 2 6 /", addr=RB(c=addr); addr+=256*RB(wrap(c,c+1)))// indirect w/ page wrap
        t("  *Z  *Z  *Z  *Z      6z  *Z  *Z ", Misfire(addr, addr+d)) // abs. load: extra misread when cross-page
        t("  4k  4k  4k  4k  6z      4k  4k ", RB(wrap(addr, addr+d)))// abs. store: always issue a misread
        /* Load source operand */
        t("aa__ff__ab__,4  ____ -  ____     ", t &= A) // Many operations take A or X as operand. Some try in
        t("                knnn     4  99   ", t &= X) // error to take both; the outcome is an AND operation.
        t("                9989    99       ", t &= Y) // sty,dey,iny,tya,cpy
        t("                       4         ", t &= S) // tsx, las
        t("!!!!  !!  !!  !!  !   !!  !!  !!/", t &= P.raw|pbits; c = t)// php, flag test/set/clear, interrupts
        t("_^__dc___^__            ed__98   ", c = t; t = 0xFF)        // save as second operand
        t("vuwvzywvvuwvvuwv    zy|zzywvzywv ", t &= RB(addr+d)) // memory operand
        t(",2  ,2  ,2  ,2  -2  -2  -2  -2   ", t &= RB(PC++))   // immediate operand
        /* Operations that mogrify memory operands directly */
        t("    88                           ", P.V = t & 0x40; P.N = t & 0x80) // bit
        t("    nink    nnnk                 ", sb = P.C)       // rol,rla, ror,rra,arr
        t("nnnknnnk     0                   ", P.C = t & 0x80) // rol,rla, asl,slo,[arr,anc]
        t("        nnnknink                 ", P.C = t & 0x01) // lsr,sre, ror,rra,asr
        t("ninknink                         ", t = (t << 1) | (sb << 0))
        t("        nnnknnnk                 ", t = (t >> 1) | (sb << 7))
        t("                 !      kink     ", t = u8(t - 1))  // dec,dex,dey,dcp
        t("                         !  khnk ", t = u8(t + 1))  // inc,inx,iny,isb
        /* Store modified value (memory) */
        t("kgnkkgnkkgnkkgnkzy|J    kgnkkgnk ", WB(addr+d, t))
        t("                   q             ", WB(wrap(addr, addr+d), t &= ((addr+d) >> 8))) // [shx,shy,shs,sha?]
        /* Some operations used up one clock cycle that we did not account for yet */
        t("rpstljstqjstrjst - - - -kjstkjst/", tick()) // nop,flag ops,inc,dec,shifts,stack,transregister,interrupts
        /* Stack operations and unconditional jumps */
        t("     !  !    !                   ", tick(); t = Pop())                        // pla,plp,rti
        t("        !   !                    ", RB(PC++); PC = Pop(); PC |= (Pop() << 8)) // rti,rts
        t("            !                    ", RB(PC++))  // rts
        t("!   !                           /", d=PC+(op?-1:1); Push(d>>8); Push(d))      // jsr, interrupts
        t("!   !    8   8                  /", PC = addr) // jmp, jsr, interrupts
        t("!!       !                      /", Push(t))   // pha, php, interrupts
        /* Bitmasks */
        t("! !!  !!  !!  !!  !   !!  !!  !!/", t = 1)
        t("  !   !                   !!  !! ", t <<= 1)
        t("! !   !   !!  !!       !   !   !/", t <<= 2)
        t("  !   !   !   !        !         ", t <<= 4)
        t("   !       !           !   !____ ", t = u8(~t)) // sbc, isb,      clear flag
        t("`^__   !       !               !/", t = c | t)  // ora, slo,      set flag
        t("  !!dc`_  !!  !   !   !!  !!  !  ", t = c & t)  // and, bit, rla, clear/test flag
        t("        _^__                     ", t = c ^ t)  // eor, sre
        /* Conditional branches */
        t("      !       !       !       !  ", if(t)  { tick(); Misfire(PC, addr = s8(addr) + PC); PC=addr; })
        t("  !       !       !       !      ", if(!t) { tick(); Misfire(PC, addr = s8(addr) + PC); PC=addr; })
        /* Addition and subtraction */
        t("            _^__            ____ ", c = t; t += A + P.C; P.V = (c^t) & (A^t) & 0x80; P.C = t & 0x100)
        t("                        ed__98   ", t = c - t; P.C = ~t & 0x100) // cmp,cpx,cpy, dcp, sbx
        /* Store modified value (register) */
        t("aa__aa__aa__ab__ 4 !____    ____ ", A = t)
        t("                    nnnn 4   !   ", X = t) // ldx, dex, tax, inx, tsx,lax,las,sbx
        t("                 !  9988 !       ", Y = t) // ldy, dey, tay, iny
        t("                   4   0         ", S = t) // txs, las, shs
        t("!  ! ! !!  !   !       !   !   !/", P.raw = t & ~0x30) // plp, rti, flag set/clear
        /* Generic status flag updates */
        t("wwwvwwwvwwwvwxwv 5 !}}||{}wv{{wv ", P.N = t & 0x80)
        t("wwwv||wvwwwvwxwv 5 !}}||{}wv{{wv ", P.Z = u8(t) == 0)
        t("             0                   ", P.V = (((t >> 5)+1)&2))         // [arr]

        /* All implemented opcodes are cycle-accurate and memory-access-accurate.
         * Cycle-accuracy depends on memory access functions RB and WB executing tick().
         * [] in the comments indicates that this particular separate rule
         * exists only because of the indicated unofficial opcode(s).
         * These unofficial opcodes are supported by this matrix without extra effort:
         * SLO, RLA, SRE, RRA, LAX, LAS(BB), DCP, ISB, SAX, ASR(4B)
         * These unofficial opcodes are supported by this matrix, through extra effort:
         * ANC(0B,2B) -- calculates C flag from bit 7 of result (in e.g. ADC, it is calculated from bit 8)
         * ARR(6B)    -- calculates C flag from bit 7 of input (in LSR, it is calculated from bit 0)
         *            -- also calculates V flag in an unique manner
         * SHX,Y,A,S  -- upper byte of address contributes to result; store address quirky.
         * These unofficial opcodes are only partially implemented:
         * 8B:XAA/ANE:     Unreliable. No matter what you choose to do it's probably incorrect.
         * CB:SBX/AAX/AXS: Test-driven development. Passes Blargg's instr_test-v3.
         * 9C:SHY/SYA:     Ditto. May still be wrong. Many different formula passed the test.
         * 9E:SHX/SXA:     Ditto. The key was to store into wrapped address rather than the normal one.
         * 9B:SHS:         Extrapolated from above and from the documentation. There was no test.
         * 9F:SHA:         Ditto. However, the store mechanism was changed for op 93 in order-
         * 93:SHA:         -to not mess the internal logic of Blargg's cpu_timing_test.nes.
         * Note: Unofficial NOP opcodes may do memory accesses that
         *       they are not supposed to do. There was no comprehensive test.
         * Note: All KIL/JAM opcodes do (some) actual work instead of halting.
         * Note: Reset,Interrupt,NMI are implemented as instructions 101,102,100 respectively.
         *       They are variants of BRK. This is the same as what real 6502 does.
         *       Note that for Reset to work properly, you will have to disable RAM writes
         *       while the instruction is in progress.
         */
    }

    void Op()
    {
        /* Check the state of NMI flag */
        bool nmi_now = nmi;

        if(PC == 0xD116) TurboLevel = 12; // TimeDelayWithAllObjectsHalted, grab level-end, megaman/boss killed
        if(PC == 0xD130) TurboLevel = 10;

        if(PC == 0xD109) TurboLevel = 12; // TimeDelayWithSpriteUpdates, shutters
        if(PC == 0xD115) TurboLevel = 10;

        if(PC == 0xC20F) TurboLevel = 70; // After level end.
        if(PC == 0xC218) TurboLevel = 10; // Saving: ((60+255+128)*6/7 * 9) / 60 = ~56 s
           // total_length (x) - became_length (x/7) = saved_length (x*6/7)

        if(PC == 0x913B) TurboLevel = 15; // "READY" becomes 2 seconds from 3 seconds
        if(PC == 0x9146) TurboLevel = 10; // This saves a total of 7 seconds

        unsigned op = RB(PC++);
        //R:;  // for nestest.nes
        if(reset)                              { op=0x101; }
        else if(nmi_now && !nmi_edge_detected) { op=0x100; nmi_edge_detected = true; }
        else if(intr && !P.I)                  { op=0x102; }
        if(!nmi_now) nmi_edge_detected=false;

        //if(0) Dasm(op);

        #define c(n) Ins<0x##n>,Ins<0x##n+1>,
        #define o(n) c(n)c(n+2)c(n+4)c(n+6)
        static void(*const i[0x108])() =
        {
            o(00)o(08)o(10)o(18)o(20)o(28)o(30)o(38)o(40)o(48)o(50)o(58)o(60)o(68)o(70)o(78)
            o(80)o(88)o(90)o(98)o(A0)o(A8)o(B0)o(B8)o(C0)o(C8)o(D0)o(D8)o(E0)o(E8)o(F0)o(F8)o(100)
        };
        #undef o
        #undef c
        i[op]();
        // For Kevin's nestest.nes:
        //if(reset) { PC = 0xC000; op = RB(PC++); reset=false; goto R; }
        reset = false;
    }
}

int main(int/*argc*/, char** argv)
{
    // Open the ROM file specified on commandline
    FILE* fp = fopen(argv[1], "rb");

    // Read the ROM file header
    assert(fgetc(fp)=='N' && fgetc(fp)=='E' && fgetc(fp)=='S' && fgetc(fp)=='\32');
    u8 rom16count = fgetc(fp);
    u8 vrom8count = fgetc(fp);
    u8 ctrlbyte   = fgetc(fp);
    u8 mappernum  = fgetc(fp) | (ctrlbyte>>4);
    fgetc(fp);fgetc(fp);fgetc(fp);fgetc(fp);fgetc(fp);fgetc(fp);fgetc(fp);fgetc(fp);
    if(mappernum >= 0x40) mappernum &= 15;
    Pak::mappernum = mappernum;

    // Read the ROM data
    if(rom16count) Pak::ROM.resize(rom16count * 0x4000);
    if(vrom8count) Pak::VRAM.resize(vrom8count * 0x2000);
    fread(&Pak::ROM[0], rom16count, 0x4000, fp);
    fread(&Pak::VRAM[0], vrom8count, 0x2000, fp);

    fclose(fp);
    printf("%u * 16kB ROM, %u * 8kB VROM, mapper %u, ctrlbyte %02X\n", rom16count, vrom8count, mappernum, ctrlbyte);

    // Start emulation
    Pak::Init(0x1010);
    IO::Init();
    PPU::reg.value = 0;
    IO::TrackROMsize( Pak::ROM.size() );
    /*IO::TrackVRAMsize( Pak::VRAM.size() );*/

    // Pre-initialize RAM for Wizards & Warriors TAS
    for(unsigned a=0; a<0x800; ++a)
        CPU::RAM[a] = (a&4) ? 0xFF : 0x00;

    // Run the CPU until the program is killed.
    //for(;;) CPU::Op();
    while(IO::frame_begin_when < (14*60 + 59.5))
        for(unsigned n=0; n<65536; ++n)
            CPU::Op();
}
