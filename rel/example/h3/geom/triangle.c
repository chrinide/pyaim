#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
  double x;
  double y;
} Point;

Point vertex(Point p1, Point p2){
  double s60 = sin(60*M_PI/180.0);    
  double c60 = cos(60*M_PI/180.0);

  Point v = {
    c60*(p1.x - p2.x) - s60*(p1.y - p2.y) + p2.x,
    s60*(p1.x - p2.x) + c60*(p1.y - p2.y) + p2.y
  };

  return v;
}

int main (int argc, char **argv) {
	Point p1,p2,p3;
	double x; 
  x = atof(argv[1]); 
	p1.x = 0.0;
	p1.y = 0.0;
	p2.x = x;
	p2.y = 0.0;
  p3 = vertex(p1,p2);
	printf("H %2.16f %2.16f %2.16f\n",p1.x,p1.y,0.0);
	printf("H %2.16f %2.16f %2.16f\n",p2.x,p2.y,0.0);
	printf("H %2.16f %2.16f %2.16f\n",p3.x,p3.y,0.0);
}
