/* Header file for DNS */

#ifndef DNS_H
#define DNS_H

#include <math.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <fstream>

using namespace std;

#define		PI			4*atan(1)
#define		MAX_FNAME_LEN		128

#define		TRUE			1
#define		FALSE			0

/* macros */
#define		MIN(a,b)		( (a<b) ? a : b )
#define		MAX(a,b)		( (a>b) ? a : b )
#define		N			24*24*24

#endif	// DNS_H

double rand_norm()
{
	double x1,x2,w,r1,r2;
	double eps = 1e-10;
	do
	{
		r1 = (double) rand() / ( (double)(RAND_MAX) + (double)(1) );
		r2 = (double) rand() / ( (double)(RAND_MAX) + (double)(1) );
		x1 = 2.0*r1 - 1.0;
		x2 = 2.0*r2 - 1.0;
		w = x1*x1 + x2*x2 + eps;
	} 	while (w>= 1.0);
		
	w = sqrt( (-2.0* log(w))/w);
	r1 = x1*w;
	r2 = x2*w;
	
	return(r1);
}

inline double rand_one()
{
	return ( (double) rand() / ( (double)(RAND_MAX) + (double)(1) ) );
}
