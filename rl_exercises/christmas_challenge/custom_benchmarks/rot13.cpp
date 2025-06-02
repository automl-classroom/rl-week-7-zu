#include "rot13.h"

std::string ROTEncode(std::string instring, int rot)
{
    std::string result;
    for (char a : instring)
    {
        if (IsBigLetter(a))
            result += ((int)a - (int)'A' + rot) % 26 + 'A';
        else if (IsSmallLetter(a))
            result += ((int)a - (int)'a' + rot) % 26 + 'a';
        else
            result += a;
    }
    return result;
}
std::string ROTDecode(std::string instring, int rot)
{
    std::string result;
    for (char a : instring)
    {
        if (IsBigLetter(a))
            result += ((int)a - (int)'A' - rot + 26) % 26 + 'A';
        else if (IsSmallLetter(a))
            result += ((int)a - (int)'a' - rot + 26) % 26 + 'a';
        else
            result += a;
    }
    return result;
}
