#include <string>

#define IsBigLetter(a) a >= 'A' && a <= 'Z'
#define IsSmallLetter(a) a >= 'a' && a <= 'z'

std::string ROTEncode(std::string instring, int rot);
std::string ROTDecode(std::string instring, int rot);