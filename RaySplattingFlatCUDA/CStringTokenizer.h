#pragma once

#include <string.h>

class CStringTokenizer {
	private:
		bool occArr[256];
		char lastDel;
		char *str;
		long long int ind;
		long long int len;
	public:
		CStringTokenizer(char *str) : str(str), ind(0), len(strlen(str)) {
			memset(occArr, 0, sizeof(bool) * 256);
		}

		char *NextTokenIE(const char *delims);
		char *NextTokenAE(const char *delims);
		void Dispose();
};
