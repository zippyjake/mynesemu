#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <signal.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>

#include <map>
#include <set>
#include <bitset>

#define MAIN_FILE 4

/**

IDEA: SET BREAKPOINT ON EVERY LOCATION
      RUN
         RECORD TRAP LOCATION, REMOVE BREAKPOINT
      REPEAT UNTIL SDL_FLIP HIT
      REPEAT
**/

struct LineInfo
{
    unsigned file, row, column;
};

enum { ip_range = 0x80000 };

std::map<unsigned long, LineInfo> lines;
std::bitset<ip_range>                used_lines;

unsigned long sdl_flip_ip = 0;
static unsigned framecounter = 0;
//static const unsigned FRAME_LIMIT = 42200; // For Gradius
static const unsigned FRAME_LIMIT = 99840; // For Mega Man II

void LogIP(unsigned long ip)
{
    static std::set<unsigned long> ip_list_prev_frame;
    static std::bitset<ip_range>   ip_map_prev_frame;

    unsigned p = ip - 0x400000;
    if(p < ip_range)
        ip_map_prev_frame.set(p);
    else
        ip_list_prev_frame.insert(ip);

    if(ip == sdl_flip_ip)
    {
        static FILE* fp = fopen("nestrace2.log", "wb");
        if(!fp)
            { std::perror("nestrace.log"); return; }
        if(ftell(fp) == 0)
            std::fprintf(fp, "%u\n", (unsigned) sizeof(ip_map_prev_frame));

        std::fwrite(&ip_map_prev_frame, sizeof(ip_map_prev_frame), 1, fp);
        std::fflush(fp);

        std::fprintf(stderr, "Frame %u done\n", framecounter);
        std::fflush(stderr);
        ++framecounter;

        // NEXT FRAME
        ip_list_prev_frame.clear();
        ip_map_prev_frame.reset();
    }
}

struct BreakPoint
{
    long breakpoint;
    long original;
};

std::map<unsigned long, BreakPoint> mappings, cur_mappings;

long PtraceHelp(const char* what, enum __ptrace_request request, pid_t pid, void* a=0, void* d=0)
{
    long r = ptrace(request, pid, a, d);
    if(r < 0 && r >= -1400)
    {
        char Buf[64]="";
        snprintf(Buf, 63, "ptrace[%s]=0x%lX", what, r);
        perror(Buf);
    }
    return r;
}

#define Ptrace(r, p...) PtraceHelp(#r, r, p)

int main(int argc, char** argv)
{
    unsigned prog_main_ip = 0;
    FILE* fp = fopen("line-listings.txt", "rt");
    used_lines.reset();
    if(fp)
    {
        char Buf[512];
        while(fgets(Buf, sizeof(Buf), fp))
        {
            unsigned long ip;
            unsigned file, id, row, column;
            if(sscanf(Buf, "%lX %*c %s", &ip, Buf))
            {
                if(strcmp(Buf, "main") == 0)
                {
                    //prog_main_ip = ip;
                }
                else if(sscanf(Buf, "BisqLine_%d_%d_%d_%d", &file,&id,&row,&column))
                {
                    if(file == MAIN_FILE && row >= 811 && row <= 880
                                         && column < 100)
                    {
                        // HACK: Ignore the preprocessor part
                        //       It should not be included anyway, but it is because of -O0
                        continue;
                    }
                    
                    //printf("ip %lX: <%s>: %d,%d,%d,%d\n", ip, Buf, file,id,row,column);
                    lines[ip] = {file,row,column};

                    unsigned p = ip-0x400000;
                    if(p < ip_range)
                        used_lines.set(p);
                    else
                        fprintf(stderr, "WARNING: ip %lX will not be tracked\n", ip);

                    if(file == MAIN_FILE && (row == 915 && row <= 917) && prog_main_ip == 0)
                        prog_main_ip = ip;

                    if(file == MAIN_FILE && row == 96 && sdl_flip_ip == 0)
                        sdl_flip_ip = ip; // This is where SDL_Flip is called
                }
            }
        }
        fclose(fp);
    }


    int pid = fork();
    if(!pid)
    {
        ptrace(PTRACE_TRACEME, 0,0,0);
        char a0[] = "./nesemu1-dbg";
        argv[0] = a0;
        execv(a0, argv);
        perror("execv");
        _exit(0);
    }

    int status;
    long r;
    struct user u;

    //Ptrace(PTRACE_ATTACH, pid, 0,0);
    Ptrace(PTRACE_SETOPTIONS, pid, 0,  (void*)(
        //PTRACE_O_TRACEFORK|
        PTRACE_O_TRACEEXEC
    ));
    wait(NULL);

    printf("Creating breakpoint maps...\n");
    fflush(stdout);

    // Create a mapping of the program where every single *known* instruction is bombed with a breakpoint.
    for(auto i: lines)
    {
        /* Set breakpoints IF
         *  - File is nesemu1.cc (main program)
         */
        if(i.second.file != MAIN_FILE) continue;

        const unsigned long modulo = i.first % sizeof(long);
        const unsigned long location = i.first - modulo;

        auto is_valid = [&](long addr, const unsigned char* bytes) -> bool
        {
            bool ok = true;
            /*if(bytes[modulo+0] == 0x48
            && bytes[modulo+1] == 0x81) // mov sp, ..
                ok = false;*/
            return ok;
        };

        auto j = mappings.find(location);
        if(j == mappings.end() || j->first != location)
        {
            // New breakpoint.
            long w = Ptrace(PTRACE_PEEKTEXT, pid, (void*)(i.first - modulo), 0);
            BreakPoint p = { w, w };
            if(is_valid(i.first, (unsigned char*)&p.original))
            {
                // Insert breakpoint (one-byte INT 3 opcode)
                ((unsigned char*)&p.breakpoint)[ modulo ] = 0xCC;
                mappings.insert(j, {location,p});
                0&&printf("Insert breakpoint at %lX (%02X << %016lX)\n",(unsigned long) i.first,
                    ((unsigned char*)&p.original)[modulo], p.original);
            }
            else
                0&&printf("NOT inserting breakpoint %lX\n", (unsigned long)i.first);
        }
        else
        {
            // Existing breakpoint.
            auto& p = j->second;
            if(is_valid(i.first, (unsigned char*)&p.original))
            {
                // Insert breakpoint (one-byte INT 3 opcode)
                ((unsigned char*)&p.breakpoint)[ modulo ] = 0xCC;
                0&&printf("Added breakpoint at %lX (%02X << %016lX)\n",
                    (unsigned long) i.first,
                    ((unsigned char*)&p.original)[modulo], p.original);
            }
            else
                0&&printf("NOT adding breakpoint %lX\n", (unsigned long)i.first);
        }
    }
    printf("- %lu breakpoints created...\n", (unsigned long) mappings.size());

    bool started = false;

FindCode:
    printf("Finding program...\n");

    cur_mappings = mappings;

    // Start: Set all breakpoints
    if(started)
    {
        for(auto i: mappings)
            Ptrace(PTRACE_POKETEXT, pid, (void*)i.first, (void*)i.second.breakpoint);
    }
    else
    {
        // Set breakpoint at main() only
        auto i = mappings.find(prog_main_ip - (prog_main_ip % sizeof(long)));
        Ptrace(PTRACE_POKETEXT, pid, (void*)i->first, (void*)i->second.breakpoint);
        do {
            // Continue running
            Ptrace(PTRACE_CONT, pid, 0, 0);
            wait(&status);
            if(WIFEXITED(status)) return -1;
        } while(!WIFSTOPPED(status));
        started = true;
        r = Ptrace(PTRACE_PEEKUSER, pid, (void*)((char*)&u.regs.rip - (char*)&u), 0);
        printf("WIFSTOPPED at eip=%lX, main=%lX\n", r-1, (long)prog_main_ip);
        goto FindCode;
    }

    while(framecounter < FRAME_LIMIT)
    {
        // Read EIP
        r = Ptrace(PTRACE_PEEKUSER, pid, (void*)((char*)&u.regs.rip - (char*)&u), 0);

        if(WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP)
        {
            // Did we hit a breakpoint?
            auto i = lines.find(r-1);
            if(i != lines.end())
            {
                long break_location = r-1;

                0&&printf("- found at %lX (file %d, line %d:%d)...\n",
                       break_location, i->second.file, i->second.row, i->second.column);
                bool new_frame = (break_location) == (long)sdl_flip_ip;
                if(new_frame)
                    printf("- - it's a new frame\n");

                fflush(stdout);

                LogIP(break_location);

                // Set EIP to that same breakpoint
                Ptrace(PTRACE_POKEUSER, pid, (void*)((char*)&u.regs.rip - (char*)&u), (void*)(break_location) );

                // Disable the breakpoint so that we can run the same instruction again
                const unsigned long modulo   = (break_location) % sizeof(long);
                const unsigned long location = (break_location) - modulo;
                auto j = cur_mappings.find(location);
                if(j != cur_mappings.end())
                {
                    BreakPoint& bp = j->second;

                    ((unsigned char*)&bp.breakpoint)[ modulo ] =
                        ((unsigned char*)&bp.original)[ modulo ];

                    Ptrace(PTRACE_POKETEXT, pid, (void*)location, (void*)bp.breakpoint);

                    if(bp.breakpoint == bp.original)
                        cur_mappings.erase(j);
                }

                if(new_frame)
                {
                    // Run
                    Ptrace(PTRACE_CONT, pid, 0, 0);
                    wait(NULL);
                    goto FindCode;
                }
            }
            else
            {
                printf("Unknown location: %08lX, was stopped by status=%04X\n", r, status);
                fflush(stdout);
                return -1;
            }
        }

        // Continue running
        Ptrace(PTRACE_CONT, pid, 0, 0);

        wait(&status);
        if(WIFEXITED(status)) return -1;
        if(!WIFSTOPPED(status))
        {
            printf("Child wasn't stopped by trace!\n");
        }
    }

/*
    // Remove breakpoints
    for(auto i: mappings)
        Ptrace(PTRACE_POKETEXT, pid, (void*)i.first, (void*)i.second.original);
*/
}
