#ifndef arrayhelper_h
#define arrayhelper_h

class ArrayHelper {
public:
	static double** getArray(unsigned long rows, unsigned long cols, bool zero = false);
	static double* getZero1D(unsigned long length);
};

#endif