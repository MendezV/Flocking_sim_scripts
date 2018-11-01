#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(x,y) x<y?x:y

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


double ran1(long *idum);
double gasdev(long *idum);
void write_to_file(double *arr, int size);
double *init_array(int n_steps, double val);
float *load_matrix(char *filename, int *n, int *m);
double *init_matrix( int n_row, int n_cols, double val);
void print_array(double *x, int n_steps);
void print_array_int(int *x, int n_steps);
void print_matrix(double *A,  int n_rowA, int n_colsA);
double gammln(double xx);
double factrl(int n);
double factrl_doub(double n);
void matrix_multiply(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void matrix_multiply_doub_int(double *matrix_C ,double *A,int *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void matrix_multiply_int_doub(double *matrix_C ,int *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
double sum_array(double *x, int N);
double std_array(double *x, int N);
void std_array_along(double *matrix_C ,double *A, int axis, int n_rowA, int n_colsA);
void sum_array_along(double *matrix_C ,double *A, int axis, int n_rowA, int n_colsA);
void array_multiply(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void array_multiply_doub_int(double *matrix_C ,double *A, int *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void array_divide(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void array_sum(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void array_log(double *matrix_C ,double *A,int n_rowA, int n_colsA);
void array_exp(double *matrix_C ,double *A,int n_rowA, int n_colsA);
void array_factrl(double *matrix_C ,int *A,int n_rowA, int n_colsA);
void array_factrl_doub(double *matrix_C ,double *A,int n_rowA, int n_colsA);
void scalar_sum(double *matrix_C ,double a, double *A,int n_rowA, int n_colsA);
void scalar_multiply(double *matrix_C ,double a, double *A,int n_rowA, int n_colsA);
void array_normal(double *matrix_C ,long *idum ,int n_rowA, int n_colsA);
void array_assign(double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void array_assign_floa(double *A,float *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void diag(double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB);


double *init_vel_matrix(double v0, int n_rowv, int n_colsv, long *idum);
void convective_term_plane(double *matrix_C, double *A, double *B, double scale_fact,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void convective_term_outplane(double *matrix_C, double *A, double *B, double scale_fact,int n_rowA,int n_colsA,int n_rowB,int n_colsB);
void ising_tensor(double *vivj,double *v,int n_rowv,int n_colsv);
void nearest_neighbour_matrix(double *nij, double *r,double N_N_rad,int Time_dep_nij, int n_rowr,int n_colsr);
void ising_term(double *matrix_C, double *nij ,double *vivj,double *temp3 ,double J,double dt, double v0,int n_rownij,int n_colsnij,int n_rowvivj,int n_colsvivj);


void update_v(double *vnew, double Chi, double dt, double *s, double *v,int n_rowv,int n_colsv ,int n_rows, int n_colss);
void update_s(double *snew, double Chi, double Mu_damp, double N_N_rad,double J, double sig, double v0, double dt,int Time_dep_nij, long *idum, double *temp1, double *temp2, double *temp3, double *xi, double *s, double *v, double *r, double *nij ,double *vivj ,int n_rowv,int n_colsv,int n_rows,int n_colss, int n_rowxi,int n_colsxi,int n_rownij,int n_colsnij,int n_rowvivj, int n_colsvivj, int n_rowr,int n_colsr);
void update_r(double *rnew, double dt, double *temp4, double *v, double *r,int n_rowv,int n_colsv, int n_rowr,int n_colsr);






int main(int argc, char **argv){
	
	
	int i,j;
	int N_part; /*number of agents*/
	double J; /*flocking strength*/
	double N_N_rad; /*radius of interaction*/
	double Chi; /*generalized inertia*/
	double sig; /*noise strength*/
	int Niter; /*number of iterations*/
	double Mu_damp; /*damping coefficient*/
	double v0; /*speed of the flock */
	double T; /*total runtime*/
	int Boundary_cond;
	double *s;
	double *v;
	double *r;
	double *snew;
	double *vnew;
	double *rnew;
	double *xi;
	double *nij;
	double *vivj;
	int Dim=2;
	int Time_dep_nij=1;
	double dt;
	double *temp1, *temp2, *temp3, *temp4;
	int n_rows,n_rowv,n_rowr,n_rowxi,n_rowvivj, n_rownij, n_colss, n_colsv, n_colsr, n_colsxi, n_colsvivj, n_colsnij;
	
	N_part=atoi(argv[1]);
	J=atof(argv[2]);
	N_N_rad=atof(argv[3]);
	Chi=atof(argv[4]);
	sig=atof(argv[5]);
	Niter=atoi(argv[6]);
	Mu_damp=atof(argv[7]);
	v0=atof(argv[8]);
	T=atof(argv[9]);
	Boundary_cond=0;
	
	
	
	/*variables for use of random number geneator*/
	long *idum;
	long ed=-50;
	idum = &ed;
	
	
	
	/* initializing all matrices and arrays */
	n_rows=N_part;
	n_colss=1;
	s=init_matrix( n_rows, n_colss, 10.0); /*everyone has the same generalized momentum*/
	n_rowv=N_part;
	n_colsv=Dim;
	v=init_vel_matrix( v0, n_rowv, n_colsv, idum);  /*equal velocites must be revisited*/
	n_rowr=N_part;
	n_colsr=Dim;
	r=init_matrix( n_rowr, n_colsr, 0.0);
	array_normal(r , idum , n_rowr, n_colsr); /*random initial conditions for position*/
	n_rowxi=N_part;
	n_colsxi=Dim;
	xi=init_matrix( n_rowxi, n_colsxi, 0.0);
	n_rownij=N_part;
	n_colsnij=N_part;
	nij=init_matrix( n_rownij, n_colsnij, 0.0);
	n_rowvivj=N_part;
	n_colsvivj=N_part;
	vivj=init_matrix( n_rowvivj, n_colsvivj, 0.0);
	
	snew=init_matrix( n_rows, n_colss, 0.0);
	vnew=init_matrix( n_rowv, n_colsv, 0.0);
	rnew=init_matrix( n_rowr, n_colsr, 0.0);
	
	temp1=init_matrix( n_rows, n_colss, 0.0);
	temp2=init_matrix( n_rows, n_colss, 0.0);
    temp3=init_matrix( n_rownij, n_colsnij, 0.0);
	temp4=init_matrix( n_rowr, n_colsr, 0.0);
	/*************/
	
	dt=T/Niter;
	

	/***time evolution***/

	for(i=0;i<Niter;i++){
		update_v(vnew, Chi,  dt,  s,  v, n_rowv, n_colsv , n_rows,  n_colss);
		update_s( snew,  Chi,   Mu_damp,  N_N_rad,  J,   sig,   v0,   dt, Time_dep_nij, idum, temp1, temp2, temp3, xi, s, v, r, nij,  vivj , n_rowv, n_colsv,  n_rows,  n_colss,  n_rowxi, n_colsxi, n_rownij,  n_colsnij, n_rowvivj,   n_colsvivj,   n_rowr, n_colsr);
		update_r(rnew, dt, temp4, v,  r,  n_rowv,  n_colsv, n_rowr,  n_colsr);
		
		array_assign(v,vnew,n_rowv,n_colsv,n_rowv,n_colsv);
		/*array_assign(s,snew,n_rows,n_colss,n_rows,n_colss);*/
		array_assign(r,rnew,n_rowr,n_colsr,n_rowr,n_colsr);
		
		printf("%f %f \n",r[4],r[5]);
		
	}

	/*******************/
}




double *init_vel_matrix(double v0, int n_rowv, int n_colsv, long *idum){
	double *matrix;
	double theta;
	double gasdev(long *idum);
	
	int i;
	
	
	matrix = malloc(n_rowv * n_colsv * sizeof(double));
	
	for(i=0;i<n_rowv;i++){
		    theta=gasdev(idum);
			matrix[i*n_colsv + 0]=v0*cos(theta);
			matrix[i*n_colsv + 1]=v0*sin(theta);
		
	}
	
	return matrix;
	
	
}


void convective_term_plane(double *matrix_C, double *A, double *B, double scale_fact,int n_rowA, int n_colsA,int n_rowB,int n_colsB){
	int i;
	
	for(i=0;i<n_rowA;i++){
		matrix_C[i] = scale_fact*(A[i*n_colsA + 0]*B[i*n_colsB + 1]-A[i*n_colsA + 1]*B[i*n_colsB + 0]);
	}
	
}


void convective_term_outplane(double *matrix_C, double *A, double *B, double scale_fact,int n_rowA, int n_colsA,int n_rowB,int n_colsB){
	int i,j;
	
	for(i=0;i<n_rowA;i++){
		for(j=0;j<n_colsA;j++){
			matrix_C[i*n_colsA + j] = scale_fact*(A[i]*B[i*n_colsB + 1-j])*pow(-1,1+j);
	}
	}
	
}


/*room for optimization by taking advantage of the fact this matrix is antisymmetric*/
void ising_tensor(double *vivj,double *v,int n_rowv,int n_colsv){
	int i,j;
	for(i=0;i<n_rowv;i++){
		for(j=0;j<n_rowv;j++){
			vivj[i*n_rowv + j] =(v[i*n_colsv + 0]*v[j*n_colsv + 1]-v[i*n_colsv + 1]*v[j*n_colsv + 0]);
		}
	}
}


void nearest_neighbour_matrix(double *nij, double *r,double N_N_rad,int Time_dep_nij, int n_rowr,int n_colsr){
	scalar_multiply(nij , 0.0, nij, n_rowr, n_rowr);
	
	int i,j;
	if(Time_dep_nij==0){
		nij[0*n_rowr + 1] =1.0;
		/*nij[0*n_rowr + 0] =1.0;*/ /*self interaction?*/
		for(i=1;i<n_rowr-1;i++){
				/*nij[i*n_rowr + i] =1.0;*/ /*self interaction?*/
				nij[i*n_rowr + i-1] =1.0;
				nij[i*n_rowr + i+1] =1.0;
		}
		nij[(n_rowr-1)*n_rowr + n_rowr-2] =1.0;
		/*nij[(n_rowr-1)*n_rowr + (n_rowr-1)] =1.0;*/ /*self interaction?*/
	}
	else{
		double dsq=0;
		for(i=0;i<n_rowr;i++){
			for(j=0;j<i;j++){
				dsq=(r[i*n_rowr + 0]-r[j*n_rowr + 0])*(r[i*n_rowr + 0]-r[j*n_rowr + 0])+(r[i*n_rowr + 1]-r[j*n_rowr + 1])*(r[i*n_rowr + 1]-r[j*n_rowr + 1]);
				if(dsq<N_N_rad*N_N_rad){
					nij[i*n_rowr + j]=1.0;
					nij[j*n_rowr + i]=1.0;
				}
				else{
					nij[i*n_rowr + j]=0.0;
					nij[j*n_rowr + i]=0.0;
				}
			}
		}
		
	}
	
}


void ising_term(double *matrix_C, double *nij ,double *vivj, double *temp3 ,double J,double dt, double v0,int n_rownij,int n_colsnij,int n_rowvivj,int n_colsvivj){
	double scale_fact=-J*dt/(v0*v0); /*minus comes from transposing the velocity matrix to multiply with connectivity matrix*/
	matrix_multiply( temp3 ,nij,vivj,n_rownij,n_colsnij,n_rowvivj,n_colsvivj);
	scalar_multiply( temp3 , scale_fact, matrix_C, n_rownij, n_colsvivj);
	diag(matrix_C, temp3, n_rownij, 1, n_rowvivj, n_colsvivj);
	
}


void update_v(double *vnew, double Chi, double dt, double *s, double *v,int n_rowv,int n_colsv ,int n_rows, int n_colss){
	int i,j;
	double scale_fact=dt/Chi;
	convective_term_outplane(vnew, s, v, scale_fact,n_rowv, n_colsv, n_rows,n_colss);
	array_sum(vnew, v, vnew,n_rowv, n_colsv, n_rowv, n_colsv);
}


void update_s(double *snew, double Chi, double Mu_damp, double N_N_rad,double J, double sig, double v0, double dt,int Time_dep_nij, long *idum, double *temp1, double *temp2, double *temp3, double *xi, double *s, double *v, double *r, double *nij ,double *vivj ,int n_rowv,int n_colsv,int n_rows,int n_colss, int n_rowxi,int n_colsxi,int n_rownij,int n_colsnij,int n_rowvivj, int n_colsvivj, int n_rowr,int n_colsr){
	
	
	double scale_fact_1=1-dt*Mu_damp/Chi;
	double scale_fact_2=pow(dt,0.5)/v0;
	
	/*generate random force*/
	array_normal(xi , idum , n_rowxi, n_colsxi);
	scalar_multiply(xi , sig, xi, n_rowxi, n_colsxi);
	
	nearest_neighbour_matrix(nij, r, N_N_rad, Time_dep_nij, n_rowr, n_colsr);
	ising_tensor( vivj, v,  n_rowv, n_colsv);
	ising_term(snew, nij , vivj, temp3 , J, dt, v0, n_rownij, n_colsnij, n_rowvivj, n_colsvivj);
	scalar_multiply(temp1 , scale_fact_1, s, n_rows, n_colss);
	convective_term_plane(temp2, v, xi, scale_fact_2, n_rowv, n_colsv, n_rowxi, n_colsxi);
	array_sum(snew, temp1, snew,n_rows, n_colss, n_rows, n_colss);
	array_sum(snew, temp2, snew,n_rows, n_colss, n_rows, n_colss);
	
	
}


void update_r(double *rnew, double dt, double *temp4, double *v, double *r,int n_rowv,int n_colsv, int n_rowr,int n_colsr){
	
	scalar_multiply(temp4 , dt , v, n_rowv, n_colsv);
	array_sum(rnew , temp4, r, n_rowr, n_colsr, n_rowr, n_colsr);
	
}


double ran1(long *idum)
/*
 “Minimal” random number generator of Park and Miller with Bays-Durham shuffle and added
 safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint values). Call with idum a negative integer to initialize; thereafter, do not alter idum between
 successive deviates in a sequence. RNMX should approximate the largest floating value that is less than 1.*/
{
	int j;
	long k;
	static long iy=0;
	static long iv[NTAB];
	double temp;
	if (*idum <= 0 || !iy) { /*Initialize.*/
		if (-(*idum) < 1) *idum=1; /*Be sure to prevent idum = 0.*/
		else *idum = -(*idum);
		for (j=NTAB+7;j>=0;j--) { /*Load the shuffle table (after 8 warm-ups).*/
			k=(*idum)/IQ;
			*idum=IA*(*idum-k*IQ)-IR*k;
			if (*idum < 0) *idum += IM;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ; /*Start here when not initializing.*/
	*idum=IA*(*idum-k*IQ)-IR*k; /*Compute idum=(IA*idum) % IM without overflows*/
	if(*idum < 0) *idum += IM; /* by Schrage’s method.*/
	j=iy/NDIV; /*Will be in the range 0..NTAB-1.*/
	iy=iv[j]; /*Output previously stored value and refill the shuffle table.*/
	iv[j] = *idum;
	if ((temp=AM*iy) > RNMX) return RNMX; /*Because users don’t expect endpoint values.*/
	else return temp;
}

double gasdev(long *idum)
/*Returns a normally distributed deviate with zero mean and unit variance, using ran1(idum)
 as the source of uniform deviates.*/
{
	double ran1(long *idum);
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;
	if (*idum < 0) iset=0; /*Reinitialize.*/
	if (iset == 0) { /*We don’t have an extra deviate handy, so*/
		do {
			v1=2.0*ran1(idum)-1.0; /*pick two uniform numbers in the square extending from -1 to +1 in each direction,*/
			v2=2.0*ran1(idum)-1.0;
			rsq=v1*v1+v2*v2; /*see if they are in the unit circle, and if they are not, try again.*/
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		/*Now make the Box-Muller transformation to get two normal deviates. Return one and
		 save the other for next time.*/
		gset=v1*fac;
		iset=1; /*Set flag.*/
		return v2*fac;
	} else { /*We have an extra deviate handy,*/
		iset=0; /*so unset the flag,*/
		return gset; /*and return it.*/
	}
}

double * init_array(int n_steps,double val){
	int i;
	double *x;
	if(!(x=malloc(sizeof(double) * n_steps))){
		fprintf(stderr,"problem with malloc\n");
		exit(1);
	}
	for(i=0;i<n_steps;i++){
		x[i] = val;
	}
	return x;
}

float *load_matrix(char *filename, int *n, int *m){
	float *matrix;
	FILE *in;
	int n_row, n_cols;
	int i;
	int j;
	
	if(!(in=fopen(filename, "r"))){
		printf("Problem opening file %s\n", filename);
		exit(1);
	}
	
	fscanf(in, "%d %d\n", &n_row, &n_cols);
	/*printf("%d %d\n", n_row, n_cols);*/
	
	matrix = malloc(n_row * n_cols * sizeof(float));
	
	for(i=0;i<n_row;i++){
		for(j=0;j<n_cols;j++){
			fscanf(in, "%f", &matrix[i*n_cols + j]);
		}
	}
	*n = n_row;
	*m = n_cols;
	return matrix;
}


double * init_matrix( int n_row, int n_cols, double val){
	
	double *matrix;

	int i;
	int j;
	
	
	matrix = malloc(n_row * n_cols * sizeof(double));
	
	for(i=0;i<n_row;i++){
		for(j=0;j<n_cols;j++){
			matrix[i*n_cols + j]=val;
		}
	}
	
	return matrix;
	
}


void print_array(double *x, int n_steps){
	int i;
	for(i=0;i<n_steps-1;i++){
		fprintf(stdout, "%f,", x[i]);
	}
	fprintf(stdout, "%f\n", x[n_steps-1]);
}

void print_array_int(int *x, int n_steps){
	int i;
	for(i=0;i<n_steps-1;i++){
		fprintf(stdout, "%d,", x[i]);
	}
	fprintf(stdout, "%d\n", x[n_steps-1]);
}


void print_matrix(double *A,  int n_rowA, int n_colsA){
	int i,j;
	for(i=0;i<n_rowA;i++){
		for(j=0;j<n_colsA-1;j++){
			fprintf(stdout, "%f,", A[i*n_colsA + j]);
		}
		fprintf(stdout, "%f\n", A[i*n_colsA + (n_colsA -1)]);
	}
}



double gammln(double xx)
/*Returns the value ln[gamma(xx)] for xx > 0.*/
{
	/*Internal arithmetic will be done in double precision, a nicety that you can omit if five-figure
	 accuracy is good enough.*/
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5};
	int j;
	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}

double factrl(int n)
/*Returns the value n! as a floating-point number.*/
{
	double gammln(double xx);
	void nrerror(char error_text[]);
	static int ntop=4;
	static double a[33]={1.0,1.0,2.0,6.0,24.0}; /*Fill in table only as required.*/
	int j;
	if (n < 0) printf("Negative factorial in routine factrl");
	if (n > 32) return exp(gammln(n+1.0));
	/*Larger value than size of table is required. Actually, this big a value is going to overflow
	 on many computers, but no harm in trying.*/
	while (ntop<n) { /*Fill in table up to desired value.*/
		j=ntop++;
		a[ntop]=a[j]*ntop;
	}
	return a[n];
}

double factrl_doub(double n)
/*Returns the value n! as a floating-point number.*/
{
	int m;
	m= (int)n;
	
	double gammln(double xx);
	void nrerror(char error_text[]);
	static int ntop=4;
	static double a[33]={1.0,1.0,2.0,6.0,24.0}; /*Fill in table only as required.*/
	int j;
	if (m < 0) printf("Negative factorial in routine factrl");
	if (m > 32) return exp(gammln(m+1.0));
	/*Larger value than size of table is required. Actually, this big a value is going to overflow
	 on many computers, but no harm in trying.*/
	while (ntop<m) { /*Fill in table up to desired value.*/
		j=ntop++;
		a[ntop]=a[j]*ntop;
	}
	return a[m];
}


void matrix_multiply(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i, j,k;
	
	
	for(i=0;i<n_rowA;i++){
		for(j=0;j<n_colsB;j++){
			matrix_C[i*n_colsB + j]=0.0;
			for(k=0;k<n_colsA;k++){
				matrix_C[i*n_colsB + j] = matrix_C[i*n_colsB + j]+A[i*n_colsA + k]*B[k*n_colsB + j];
			}
		}
		
	}
	
}

void matrix_multiply_doub_int(double *matrix_C ,double *A,int *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i, j,k;
	
	for(i=0;i<n_rowA;i++){
		for(j=0;j<n_colsB;j++){
			matrix_C[i*n_colsB + j]=0.0;
			for(k=0;k<n_colsA;k++){
				matrix_C[i*n_colsB + j] = matrix_C[i*n_colsB + j]+A[i*n_colsA + k]*B[k*n_colsB + j];
			}
			
		}
		
	}
	
	
}

void matrix_multiply_int_doub(double *matrix_C ,int *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i, j,k;
	
	for(i=0;i<n_rowA;i++){
		for(j=0;j<n_colsB;j++){
			matrix_C[i*n_colsB + j]=0.0;
			for(k=0;k<n_colsA;k++){
				matrix_C[i*n_colsB + j] = matrix_C[i*n_colsB + j]+A[i*n_colsA + k]*B[k*n_colsB + j];
			}
		}
	}
}

double sum_array(double *x, int N){
	int i;
	double Sum=0.0;
	for (i=0; i<N; i++) {
		Sum+=x[i];
	}
	return Sum;
}

double std_array(double *x, int N)
{
	double sum = 0.0, mean, tmp = 0.0;
	int i;
	
	for(i=0; i<N; ++i)
	{
		sum += x[i];
	}
	mean = sum/N;
	
	for(i=0; i<N; ++i)
	tmp += pow(x[i] - mean, 2);
	
	return sqrt(tmp/N);
}


void std_array_along(double *matrix_C ,double *A, int axis, int n_rowA, int n_colsA)
{
	double sum = 0.0, mean, tmp = 0.0;
	int i,j;
	
	if(axis==0){
		/*matrix_C = malloc(n_colsA* sizeof(double));*/
		for(j=0;j<n_colsA;j++){
			
			matrix_C[j]=0.0;
			mean=0.0;
			
			for(i=0;i<n_rowA;i++){
				mean=matrix_C[j]+A[i*n_colsA + j];
			}
			mean=mean/n_rowA;
			
			for(i=0; i<n_rowA; ++i){
				tmp += pow(A[i*n_colsA + j] - mean, 2);
			}
			matrix_C[j]=sqrt(tmp/n_rowA);
		}
		
	}
	else if(axis==1){
		/*matrix_C = malloc(n_rowA * sizeof(double));*/
		for(i=0;i<n_rowA;i++){
			
			matrix_C[i]=0.0;
			mean=0.0;
			
			for(j=0;j<n_colsA;j++){
				matrix_C[i]=matrix_C[i]+A[i*n_colsA + j];
			}
			mean=mean/n_colsA;
			
			for(j=0; j<n_colsA; ++j){
				tmp += pow(A[i*n_colsA + j] - mean, 2);
			}
			
			matrix_C[i]=sqrt(tmp/n_colsA);
		}
		
	}
	else{
		printf("axis not valid just 1 or 0 allowed \n");
		exit(1);
		
	}
}

void  sum_array_along(double *matrix_C ,double *A, int axis, int n_rowA, int n_colsA){
	
	int i,j;
	
	if(axis==0){
		/*matrix_C = malloc(n_colsA* sizeof(double));*/
		for(j=0;j<n_colsA;j++){
			matrix_C[j]=0.0;
			for(i=0;i<n_rowA;i++){
				matrix_C[j]=matrix_C[j]+A[i*n_colsA + j];
			}
		}
		
	}
	else if(axis==1){
		/*matrix_C = malloc(n_rowA * sizeof(double));*/
		for(i=0;i<n_rowA;i++){
			matrix_C[i]=0.0;
			for(j=0;j<n_colsA;j++){
				matrix_C[i]=matrix_C[i]+A[i*n_colsA + j];
			}
		}
		
	}
	else{
		printf("axis not valid just 1 or 0 allowed \n");
		exit(1);
		
	}
}

void array_multiply(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i,size;
	
	
	if(n_rowA!=n_rowB ||n_colsA!=n_colsB){
		printf("dimensions do not coincide \n");
		exit(1);
	}
	else{
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			matrix_C[i]=A[i]*B[i];
		}
		
	}
	
}

void array_divide(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i,size;
	
	
	if(n_rowA!=n_rowB ||n_colsA!=n_colsB){
		printf("dimensions do not coincide \n");
		exit(1);
	}
	else{
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			matrix_C[i]=A[i]/B[i];
		}
		
	}
	
	
}


void array_sum(double *matrix_C ,double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB){
	
	int i,size;
	
	if(n_rowA!=n_rowB || n_colsA!=n_colsB){
		printf("dimensions do not coincide \n");
		exit(1);
	}
	else{
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			matrix_C[i]=A[i]+B[i];
		}
		
	}
	
	
}
void array_log(double *matrix_C ,double *A, int n_rowA, int n_colsA){
	
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++){
		matrix_C[i]=log(A[i]);
	}
	
}

void array_exp(double *matrix_C,double *A,int n_rowA, int n_colsA)
{
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++){
		matrix_C[i]=exp(A[i]);
	}
	
}

void array_factrl(double *matrix_C ,int *A, int n_rowA, int n_colsA){
	
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++ ){
		matrix_C[i]=factrl(A[i]);
	}
	
}


void array_factrl_doub(double *matrix_C ,double *A, int n_rowA, int n_colsA){
	
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++ ){
		matrix_C[i]=factrl_doub(A[i]);
	}
	
}

void scalar_sum(double *matrix_C ,double a, double *A,int n_rowA, int n_colsA){
	
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++ ){
		matrix_C[i]=A[i]+a;
	}
	
}

void scalar_multiply(double *matrix_C ,double a, double *A,int n_rowA, int n_colsA){
	
	int i,size;
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++ ){
		matrix_C[i]=a*A[i];
	}
	
}


void array_normal(double *matrix_C ,long *idum,int n_rowA, int n_colsA){
	
	int i,size;
	double gasdev(long *idum);
	
	size=n_rowA*n_colsA;
	
	for( i=0;i<size;i++ ){
		matrix_C[i]=gasdev(idum);
		
	}
	
}

void array_assign(double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i,size;
	
	if(n_rowA!=n_rowB || n_colsA!=n_colsB){
		printf("dimensions do not coincide \n");
		exit(1);
	}
	else{
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			A[i]=B[i];
		}
		
	}
}

void array_assign_floa(double *A,float *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i,size;
	
	if(n_rowA!=n_rowB || n_colsA!=n_colsB){
		printf("dimensions do not coincide \n");
		exit(1);
	}
	else{
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			A[i]=B[i];
		}
		
	}
}

void diag(double *A,double *B,int n_rowA,int n_colsA,int n_rowB,int n_colsB)
{
	int i,size;
	
	
		
		size=n_rowA*n_colsA;
		for( i=0;i<size;i++){
			A[i]=B[i*n_colsB + i];
		}
		
	
}



