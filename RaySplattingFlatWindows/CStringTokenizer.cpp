#include "CStringTokenizer.h"

char *CStringTokenizer::NextTokenIE(const char *delims) {
	char *res;
	long long int delLen = strlen(delims);
	
	if (ind > 0) str[ind - 1] = lastDel;
	if (ind < len) {
		for (long long int i = 0; i < delLen; ++i) occArr[delims[i]] = true;
		while ((ind < len) && (occArr[str[ind]])) ++ind;
		if (ind < len) {
			res = &str[ind];
			while ((ind < len) && (!occArr[str[ind]])) ++ind;
			lastDel = str[ind];
			str[ind++] = 0;
		} else
			res = NULL;
		for (long long int i = 0; i < delLen; ++i) occArr[delims[i]] = false;
	} else
		res = NULL;
	return res;
}

char *CStringTokenizer::NextTokenAE(const char *delims) {
	char *res;
	long long int delLen = strlen(delims);
	
	if (ind > 0) str[ind - 1] = lastDel;
	if (ind < len) {
		res = &str[ind];
		for (long long int i = 0; i < delLen; ++i) occArr[delims[i]] = true;
		while ((ind < len) && (!occArr[str[ind]])) ++ind;
		lastDel = str[ind];
		str[ind++] = 0;
		for (long long int i = 0; i < delLen; ++i) occArr[delims[i]] = false;
	} else
		res = NULL;
	return res;
}

void CStringTokenizer::Dispose() {
	if (ind > 0) str[ind - 1] = lastDel;
}
