//==========================================================================================================
// 2D Binary Alloy dendritic Solidification using pseudo spectral based phase field method
// Filtering based dealiasing scheme is used and non-conservative form of the order parameter equation is used
// Arijit Sinhababu (IIT Bombay) 18/08/2022
//==========================================================================================================
#include<fftw3.h>
#include "dns.h"
#include <cmath>
#include <iostream>
#include <complex>
#include <math.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ctime>
using namespace std;
int NP1 = 900, NP2 = NP1, NPH1 = NP1/2, NPH2 = NP2/2;
int NPH2P1=NPH2+1;
// calling the functions within the scope of main
void DERIVATIVE_FOURIER(std::complex<double>** fourier_ux_temp, double** k1_temp, std::complex<double>** uhat0_temp, int NP1_temp, int NP2_temp);
void INVERSE_TRANSFORM(double** uout1_final_temp, std::complex<double>** uc_temp, int NP1_temp, int NP2_temp, int NPH2P1_temp);
void FILTER_INVERSE_TRANSFORM(double** uout1_filt_final_temp, std::complex<double>** uc_filt_temp, std::complex<double>** wk_filt_temp, int NP1_temp, int NP2_temp, int NPHP1_temp);
void FORWARD_TRANSFORM(std::complex<double>** nlt_final_temp, double** uux_temp, int NP1_temp, int NP2_temp, int NPH2P1_temp);
void MULTIPLY_PHYSICAL_SPACE(double** mulphysical_temp, double** uphysical_temp ,double** uxphysical_temp, int NP1_temp, int NP2_temp);
void CALCULATE_ANGLE(double** theta_temp, double** dphidx_temp, double** dphidy_temp, int NP1_temp, int NP2_temp);
void CALCULATE_ANISOTROPY(double** sigma_temp, double delta_temp, double j_temp, double** theta_temp, double theta0_temp, int NP1_temp, int NP2_temp);
void CALCULATE_ANISO_GRAD_ENERGY(double** epsilon_temp, double epsilon_mean_temp, double** sigma_temp, int NP1_temp, int NP2_temp);
void CALCULATE_DERIVATIVE_ANISO_GRAD_ENERGY(double** depsdtheta_temp, double epsilon_mean_temp, double j_temp, double delta_temp, double** theta_temp, double theta0_temp, int NP1_temp, int NP2_temp);
void CALCULATE_PHI_T8_PHYSICAL(double** t8_temp, double** phiphysical_temp, double lambda_temp, double** tphysical_temp, double mc_inf_temp, double** ucphysical_temp, int NP1_temp, int NP2_temp);
void CALCULATE_SPECTRAL_LAPLACIAN(std::complex<double>** laplace_phi_hat_temp, double** k1_temp ,std::complex<double>** phihat_temp, int NP1_temp, int NP2_temp);
void EQRHS_PHI(std::complex<double>** rhs_phi_hat_temp, std::complex<double>** t1_hat_temp, std::complex<double>** t2_hat_temp, std::complex<double>** t3_hat_temp, std::complex<double>** t4_hat_temp, std::complex<double>** t5_hat_temp, std::complex<double>** t6_hat_temp, std::complex<double>** t7_hat_temp, std::complex<double>** t8_hat_temp, int NP1_temp, int NP2_temp);
void EQRHS_TEMPERATURE(std::complex<double>** rhs_t_hat_temp, std::complex<double>** rhs_phi_hat_temp, int NP1_temp, int NP2_temp);
void SSPRK3_TIME_STEP(std::complex<double>** uhat1_temp, std::complex<double>** uhat0_temp, std::complex<double>** uhat_sub_temp, double dt_temp, std::complex<double>** rhs_hat_k3_temp, int NP1_temp, int NP2_temp);
void SSPRK3_TIME_STEP_IF(std::complex<double>** that1_temp, std::complex<double>** that0_temp, std::complex<double>** that_sub_temp, double** kmod_temp, double alpha_temp, double dt_temp, std::complex<double>** rhs_hat_temp, int NP1_temp, int NP2_temp);
int main(int argc, char ** argv) 
{
	
	char* fname_prefix;
	char* output_dir;
	if (!(argc == 3))
	{
		std::cout << "Usage: " << argv[0] << " output_dir fname_prefix" << std::endl;
		exit(-1);
	}
	fname_prefix = (char*) (malloc(strlen(argv[2])+1));
	output_dir = (char*) (malloc(strlen(argv[1])+1));

	strcpy(fname_prefix, argv[2]);
	strcpy(output_dir, argv[1]);
//	writing initial conditions/ field variables
	stringstream phiphysical0_filename;
	phiphysical0_filename << output_dir << fname_prefix << "_phi0.dat";
	ofstream phiphysical0_file(phiphysical0_filename.str().c_str(), ios::out | ios::trunc);

	stringstream ucphysical0_filename;
	ucphysical0_filename << output_dir << fname_prefix << "_uc0.dat";
	ofstream ucphysical0_file(ucphysical0_filename.str().c_str(), ios::out | ios::trunc);

	stringstream tphysical0_filename;
	tphysical0_filename << output_dir << fname_prefix << "_t0.dat";
	ofstream tphysical0_file(tphysical0_filename.str().c_str(), ios::out | ios::trunc);

	stringstream runstatus_filename;
	runstatus_filename << output_dir << fname_prefix << "_runstatus.dat";
	ofstream runstatus_file(runstatus_filename.str().c_str(), ios::out | ios::trunc);

	stringstream phiphysicalf_filename;
	stringstream phiphysical_filename;
	stringstream ucphysicalf_filename;
	stringstream ucphysical_filename;
 	stringstream tphysicalf_filename;
	stringstream tphysical_filename;

//	length of the domain in the x and y directions and the grid spacing in the x and y directions
	double len1 = 1024.0, dx1 = len1/(NP1), len2 = 1024.0,  dx2 = len2/(NP2);
	double LH = len1/2.0;
//	define the values of various parameters
	double epsilon_mean = 1.0, delta = 0.05, d0 = 0.5, a2 = 0.6267, a1 = ((2.0*sqrt(2.0))/3.0)/(16.0/15.0), j = 4.0, theta0 = 0.0,lambda = 2.0, Delta_ucool = -0.55, width = ((d0*lambda)/a1), part_coeff_const=0.15, mc_inf = 0.07, Le_numb = 40.0;
	//lambda = alpha/a2,tau0=(a2*lambda*width*width)/(D_uc_liquid)
	double D_uc_liquid = (a2*lambda), tau0 = 1.0, alpha = (D_uc_liquid*Le_numb);
//	Time increment or step
	//double dt = pow(dx1,3);
	double dt = 0.01;
	dt = 1.0*dt;
//	Data output time duration
	double T_output = (500.0/1.0)*dt;
// 	NT number of time steps
	//int NT = floor(T_total/dt)+1;
	double NT=(60000.0/1.0);
//	Nout number of time steps at which data is outputed
	int NOutput = floor(1.0*T_output/dt)+1;
//	calculation of wavenumbers in 2D
	std::cout <<"calculating the wave numbers k1 and k2"<< std::endl;
	double **k1 = new double*[NP1];
	double **k2 = new double*[NP1];
	double **kmod = new double*[NP1];
	for(int i = 0; i < NP1; i++)
	{
		k1[i] = new double[NPH2P1];
		k2[i] = new double[NPH2P1];
		kmod[i] = new double[NPH2P1];
	}
	for (int i = 0; i < NP1; i++)
	{
		for (int j = 0; j < NPH2P1; j++)
		{
			if(i <= NPH1)
			{
				k1[i][j] = (2.0*PI/len1)*i;
			}
			else
			{
				k1[i][j] = (2.0*PI/len1)*(i-NP1);
			}

			if(j <= NPH2)
			{
				k2[i][j] = (2.0*PI/len2)*j;
			}
			else
			{
				k2[i][j] = (2.0*PI/len2)*(j-NP2);
			}
				kmod[i][j] = sqrt(k1[i][j]*k1[i][j]+k2[i][j]*k2[i][j]);	
		}
	}
//========================================================================================
//	Implementation of Circular Fourier Smoothing Filter Selection W(k1, k2) (Kernel) in 2D
//========================================================================================
	std::complex<double> **wk= new std::complex<double>*[NP1];
	for(int i=0;i<NP1;i++)
	{
		wk[i]=new std::complex<double>[NPH2P1];
	}

	for (int i=0; i<NP1; i++) 
	{
		for (int j=0; j<NPH2P1; j++)
		{
			if((k1[i][j]*k1[i][j]+k2[i][j]*k2[i][j])<=floor(pow(((sqrt(2.0)/3.0)*NP1),2)))
			{
				wk[i][j] = std::complex<double>(1.0,0.0);
			}
			else
			{
				wk[i][j] = std::complex<double>(exp(-6.0*pow((kmod[i][j]/NP1), 6.0)),0.0);
			}
		}
	}
	std::cout <<"calculating initial conditions"<< std::endl;
	double **phiphysical = new double*[NP1];
	double **ucphysical = new double*[NP1];
	double **tphysical = new double*[NP1];
	for(int i = 0; i < NP1; i++)
	{
		phiphysical[i] = new double[NP2];
		ucphysical[i] = new double[NP2];
		tphysical[i] = new double[NP2];
	}
//	initial order parameter and temperature field
	for (int i = 0; i < NP1; i++)
	{
	double x = i*dx1;
		for (int j = 0; j < NP2; j++)
		{
			double y = j*dx2;
			phiphysical[i][j] = tanh(((8.0*dx1)-sqrt((x-LH)*(x-LH)+(y-LH)*(y-LH)))/(sqrt(2.0)*width));
			ucphysical[i][j] = 0.0;
			tphysical[i][j] = (0.5*Delta_ucool)-(0.5*Delta_ucool*tanh(((8.0*dx1)-sqrt((x-LH)*(x-LH)+(y-LH)*(y-LH)))/(sqrt(2.0)*width)));
		}
	}	
	for (int i = 0; i < NP1; i++)
	{
		//double x = i*dx1;
		for(int j = 0; j < NP2; j++)
		{
			//double y = j*dx2;
			phiphysical0_file <<std::fixed << std::setprecision(8)<< phiphysical[i][j] << std::endl;
			ucphysical0_file <<std::fixed << std::setprecision(8)<< ucphysical[i][j] << std::endl;
			tphysical0_file <<std::fixed << std::setprecision(8)<< tphysical[i][j] << std::endl;
		}
	}
	//	fourier-transformation of the initial condition
	std::cout << "calculating the forward tranformation" << std::endl;
	std::complex<double> **phihat0 = new std::complex<double>*[NP1];
	std::complex<double> **uchat0 = new std::complex<double>*[NP1];
	std::complex<double> **that0 = new std::complex<double>*[NP1];
	for(int i = 0; i < NP1; i++)
	{
		phihat0[i] = new std::complex<double>[NPH2P1];
		uchat0[i] = new std::complex<double>[NPH2P1];
		that0[i] = new std::complex<double>[NPH2P1];
	}
	FORWARD_TRANSFORM(phihat0, phiphysical , NP1, NP2, NPH2P1);
	FORWARD_TRANSFORM(uchat0, ucphysical , NP1, NP2, NPH2P1);
	FORWARD_TRANSFORM(that0, tphysical , NP1, NP2, NPH2P1);

	double **phiphysical_cnt_step = new double*[NP1];
	double **ucphysical_cnt_step = new double*[NP1];
	double **tphysical_cnt_step = new double*[NP1];
	for(int i = 0; i < NP1; i++)
	{
		phiphysical_cnt_step[i] = new double[NP2];
		ucphysical_cnt_step[i] = new double[NP2];
		tphysical_cnt_step[i] = new double[NP2];
	}
//=================================================Normal variables====================================================================
	std::complex<double>** fourier_dphidx = new std::complex<double>*[NP1];
	std::complex<double>** fourier_dphidy = new std::complex<double>*[NP1];
	double **dphidx = new double*[NP1];
	double **dphidy = new double*[NP1];
	double **theta = new double*[NP1];
	double **sigma = new double*[NP1];
	double **epsilon = new double*[NP1];
	double **depsilondtheta = new double*[NP1];
	double **multiply_epsilon_depsilondtheta = new double*[NP1];
	double **multiply_epsilon_epsilon = new double*[NP1];
	std::complex<double>** epsilon_depsilondtheta_hat = new std::complex<double>*[NP1];
	std::complex<double>** epsilon_epsilon_hat = new std::complex<double>*[NP1];
	std::complex<double>** fourier_ddy_epsilon_depsilondtheta = new std::complex<double>*[NP1];
	double **ddy_epsilon_depsilondtheta_filtered = new double*[NP1];
	double **dphidx_filtered = new double*[NP1];
	double **multiply_ddy_epsilon_depsilondtheta_dphidx = new double*[NP1];
	std::complex<double>** phi_term1_hat = new std::complex<double>*[NP1];
	std::complex<double>** fourier_ddy_dphidx = new std::complex<double>*[NP1];
	double **epsilon_depsilondtheta_filtered = new double*[NP1];
	double **ddy_dphidx_filtered = new double*[NP1];
	double **multiply_epsilon_depsilondtheta_ddy_dphidx = new double*[NP1];
	std::complex<double>** phi_term2_hat = new std::complex<double>*[NP1];
	std::complex<double>** fourier_ddx_epsilon_depsilondtheta = new std::complex<double>*[NP1];
	double **dphidy_filtered = new double*[NP1];
	double **ddx_epsilon_depsilondtheta_filtered = new double*[NP1];
	double **multiply_ddx_epsilon_depsilondtheta_dphidy = new double*[NP1];
	std::complex<double>** phi_term3_hat = new std::complex<double>*[NP1];
	std::complex<double>** fourier_ddx_dphidy = new std::complex<double>*[NP1];
	double **ddx_dphidy_filtered = new double*[NP1];
	double **multiply_epsilon_depsilondtheta_ddx_dphidy = new double*[NP1];
	std::complex<double>** phi_term4_hat = new std::complex<double>*[NP1];
	double **epsilon_epsilon_filtered = new double*[NP1];
	std::complex<double>** laplace_phi_hat = new std::complex<double>*[NP1];
	double **laplace_phi_filtered = new double*[NP1];
	double **multiply_epsilon_epsilon_laplace_phi = new double*[NP1];
	std::complex<double>** phi_term5_hat = new std::complex<double>*[NP1];
	std::complex<double>** fourier_ddx_epsilon_epsilon = new std::complex<double>*[NP1];
	double **ddx_epsilon_epsilon_filtered = new double*[NP1];
	double **multiply_ddx_epsilon_epsilon_dphidx = new double*[NP1];
	std::complex<double>** phi_term6_hat = new std::complex<double>*[NP1];	
	std::complex<double>** fourier_ddy_epsilon_epsilon = new std::complex<double>*[NP1];
	double **ddy_epsilon_epsilon_filtered = new double*[NP1];
	double **multiply_ddy_epsilon_epsilon_dphidy = new double*[NP1];
	std::complex<double>** phi_term7_hat = new std::complex<double>*[NP1];
	double **phi_term8 = new double*[NP1];
	std::complex<double>** phi_term8_hat = new std::complex<double>*[NP1];
	double **uc_filtered = new double*[NP1];
	double **inv_modified_epsilon_epsilon_filtered = new double*[NP1];	
	double **rhs_phi_filtered = new double*[NP1];
	double **multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered = new double*[NP1];
	std::complex<double>** rhs_phi_hat_k1 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_phi_hat_k2 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_phi_hat_k3 = new std::complex<double>*[NP1];
	std::complex<double>** laplace_uc_hat = new std::complex<double>*[NP1];	
	double **laplace_uc_filtered = new double*[NP1];	
	double **phi_filtered = new double*[NP1];
	double **multiply_onemphi_lapuc_filtered = new double*[NP1];
	std::complex<double>** nuc1_hat = new std::complex<double>*[NP1];	
	std::complex<double>** fourier_ducdx = new std::complex<double>*[NP1];		
	double **ducdx_filtered = new double*[NP1];
	double **multiply_dphidx_ducdx_filtered = new double*[NP1];
	std::complex<double>** nuc2_hat = new std::complex<double>*[NP1];		
	std::complex<double>** fourier_ducdy = new std::complex<double>*[NP1];		
	double **ducdy_filtered = new double*[NP1];
	double **multiply_dphidy_ducdy_filtered = new double*[NP1];
	std::complex<double>** nuc3_hat = new std::complex<double>*[NP1];
	double **nuc4_filtered = new double*[NP1];
	std::complex<double>** nuc4_hat = new std::complex<double>*[NP1];	
	double **rhs_uc_filtered = new double*[NP1];
	double **inv_modcoeff_phi_filtered = new double*[NP1];	
	double **multiply_rhs_uc_inv_modcoeff_phi_filtered = new double*[NP1];	
	std::complex<double>** rhs_uc_hat_k1 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_uc_hat_k2 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_uc_hat_k3 = new std::complex<double>*[NP1];	
	std::complex<double>** rhs_tp_hat_k1 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_tp_hat_k2 = new std::complex<double>*[NP1];
	std::complex<double>** rhs_tp_hat_k3 = new std::complex<double>*[NP1];
	std::complex<double>** phihat_sub = new std::complex<double>*[NP1];
	std::complex<double>** uchat_sub = new std::complex<double>*[NP1];
	std::complex<double>** that_sub = new std::complex<double>*[NP1];
	std::complex<double>** phihat1 = new std::complex<double>*[NP1];
	std::complex<double>** uchat1 = new std::complex<double>*[NP1];
	std::complex<double>** that1 = new std::complex<double>*[NP1];
	for(int i = 0; i < NP1; i++)
	{
		fourier_dphidx[i] = new std::complex<double>[NPH2P1];
		fourier_dphidy[i] = new std::complex<double>[NPH2P1];
		dphidx[i] = new double[NP2];
		dphidy[i] = new double[NP2];
		theta[i] = new double[NP2];
		sigma[i] = new double[NP2];
		epsilon[i] = new double[NP2];
		depsilondtheta[i] = new double[NP2];
		multiply_epsilon_depsilondtheta[i] = new double[NP2];
		multiply_epsilon_epsilon[i] = new double[NP2];
		epsilon_depsilondtheta_hat[i] = new std::complex<double>[NPH2P1];
		epsilon_epsilon_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddy_epsilon_depsilondtheta[i] = new std::complex<double>[NPH2P1];
		ddy_epsilon_depsilondtheta_filtered[i] = new double[NP2];
		dphidx_filtered[i] = new double[NP2];
		multiply_ddy_epsilon_depsilondtheta_dphidx[i] = new double[NP2];
		phi_term1_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddy_dphidx[i] = new std::complex<double>[NPH2P1];
		epsilon_depsilondtheta_filtered[i] = new double[NP2];
		ddy_dphidx_filtered[i] = new double[NP2];
		multiply_epsilon_depsilondtheta_ddy_dphidx[i] = new double[NP2];
		phi_term2_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddx_epsilon_depsilondtheta[i] = new std::complex<double>[NPH2P1];
		dphidy_filtered[i] = new double[NP2];
		ddx_epsilon_depsilondtheta_filtered[i] = new double[NP2];
		multiply_ddx_epsilon_depsilondtheta_dphidy[i] = new double[NP2];
		phi_term3_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddx_dphidy[i] = new std::complex<double>[NPH2P1];
		ddx_dphidy_filtered[i] = new double[NP2];
		multiply_epsilon_depsilondtheta_ddx_dphidy[i] = new double[NP2];
		phi_term4_hat[i] = new std::complex<double>[NPH2P1];
		epsilon_epsilon_filtered[i] = new double[NP2];
		laplace_phi_hat[i] = new std::complex<double>[NPH2P1];
		laplace_phi_filtered[i] = new double[NP2];
		multiply_epsilon_epsilon_laplace_phi[i] = new double[NP2];
		phi_term5_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddx_epsilon_epsilon[i] = new std::complex<double>[NPH2P1];
		ddx_epsilon_epsilon_filtered[i] = new double[NP2];
		multiply_ddx_epsilon_epsilon_dphidx[i] = new double[NP2];
		phi_term6_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ddy_epsilon_epsilon[i] = new std::complex<double>[NPH2P1];
		ddy_epsilon_epsilon_filtered[i] = new double[NP2];
		multiply_ddy_epsilon_epsilon_dphidy[i] = new double[NP2];
		phi_term7_hat[i] = new std::complex<double>[NPH2P1];
		phi_term8[i] = new double[NP2];
		phi_term8_hat[i] = new std::complex<double>[NPH2P1];
		uc_filtered[i] = new double[NP2];
		inv_modified_epsilon_epsilon_filtered[i] = new double[NP2];	
		rhs_phi_filtered[i] = new double[NP2];
		multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered[i] = new double[NP2];
		rhs_phi_hat_k1[i] = new std::complex<double>[NPH2P1];
		rhs_phi_hat_k2[i] = new std::complex<double>[NPH2P1];
		rhs_phi_hat_k3[i] = new std::complex<double>[NPH2P1];
		laplace_uc_hat[i] = new std::complex<double>[NPH2P1];		
		laplace_uc_filtered[i] = new double[NP2];
		phi_filtered[i] = new double[NP2];
		multiply_onemphi_lapuc_filtered[i] = new double[NP2];
		nuc1_hat[i] = new std::complex<double>[NPH2P1];	
		fourier_ducdx[i] = new std::complex<double>[NPH2P1];		
		ducdx_filtered[i] = new double[NP2];
		multiply_dphidx_ducdx_filtered[i] = new double[NP2];
		nuc2_hat[i] = new std::complex<double>[NPH2P1];
		fourier_ducdy[i] = new std::complex<double>[NPH2P1];		
		ducdy_filtered[i] = new double[NP2];
		multiply_dphidy_ducdy_filtered[i] = new double[NP2];
		nuc3_hat[i] = new std::complex<double>[NPH2P1];		
		nuc4_filtered[i] = new double[NP2];
		nuc4_hat[i] = new std::complex<double>[NPH2P1];			
		rhs_uc_filtered[i] = new double[NP2];
		inv_modcoeff_phi_filtered[i] = new double[NP2];
		multiply_rhs_uc_inv_modcoeff_phi_filtered[i] = new double[NP2];
		rhs_uc_hat_k1[i] = new std::complex<double>[NPH2P1];
		rhs_uc_hat_k2[i] = new std::complex<double>[NPH2P1];
		rhs_uc_hat_k3[i] = new std::complex<double>[NPH2P1];			
		rhs_tp_hat_k1[i] = new std::complex<double>[NPH2P1];
		rhs_tp_hat_k2[i] = new std::complex<double>[NPH2P1];
		rhs_tp_hat_k3[i] = new std::complex<double>[NPH2P1];
		phihat_sub[i] = new std::complex<double>[NPH2P1];
		uchat_sub[i] = new std::complex<double>[NPH2P1];
		that_sub[i] = new std::complex<double>[NPH2P1];
		phihat1[i] = new std::complex<double>[NPH2P1];
		uchat1[i] = new std::complex<double>[NPH2P1];
		that1[i] = new std::complex<double>[NPH2P1];
	}
//=======================================================================================
//	write the run status file contains the list of all parameters used in the simulation
//=======================================================================================
	runstatus_file << "L1 = " << len1 <<'\t'<< "L2 = " << len2 <<'\t'<< "NP1 = " << NP1 <<'\t'<< "NP2 = " << NP2<< std::endl;
	runstatus_file << "mc_inf = " << mc_inf <<'\t'<< "lambda = " << lambda <<'\t'<< "partition_coefficient = " << part_coeff_const <<'\t'<< "D_uc_liquid = " << D_uc_liquid<< std::endl;
  	runstatus_file << "dt = " << dt <<'\t'<< "dx = " << dx1 << std::endl; 
	runstatus_file << "epsilon_mean = " << epsilon_mean<< std::endl; 
	runstatus_file << "anisotropy strength (delta) = " << delta <<'\t'<< "relaxation time (tau0) = " << tau0 << std::endl; 
	runstatus_file << "alpha = " << alpha<<'\t'<< "Lewis Number(Le) = " << Le_numb<< std::endl; 
	runstatus_file << "symmetry (j) = " << j << std::endl; 
	runstatus_file << "theta0 = " << theta0 <<'\t'<<"d0 = " << d0 << std::endl; 
	runstatus_file << "undercooling (Delta) = " << Delta_ucool << std::endl;
	runstatus_file << "diffuse interface-width = " << width << std::endl;
//=======================================================================================
//	start time loop for updating
	std::cout <<"starting of time update loop"<< std::endl;
	int counter = 1;
//=======================================================================================
	for(int time_index = 1; time_index < NT; time_index++)
	{
		if ((time_index%NOutput == 0) && (time_index != 0))
		{
			//cout << "Writing data to disk" << '\t' << "t = " << dt*time_index << " T = " << dt*time_index/T_integral << std::endl;
			cout << "Writing data to disk" << '\t' << "t = " << dt*time_index << std::endl;

			phiphysical_filename << output_dir << fname_prefix << "_phi" << counter << ".dat";
			cout << phiphysical_filename.str() << std::endl;
			ofstream phiphysical_file(phiphysical_filename.str().c_str(), ios::out | ios::trunc);

			ucphysical_filename << output_dir << fname_prefix << "_uc" << counter << ".dat";
			cout << ucphysical_filename.str() << std::endl;
			ofstream ucphysical_file(ucphysical_filename.str().c_str(), ios::out | ios::trunc);

			tphysical_filename << output_dir << fname_prefix << "_t" << counter << ".dat";
			cout << tphysical_filename.str() << std::endl;
			ofstream tphysical_file(tphysical_filename.str().c_str(), ios::out | ios::trunc);

			//runstatus_file << "counter = " << '\t' << counter << '\t' << " t = " << dt*time_index << '\t' << "T = " << dt*time_index/T_integral << '\t' << "dt = " << dt<< std::endl;
			runstatus_file << "counter = " << '\t' << counter << '\t' << " t = " << dt*time_index << '\t' << "dt = " << dt<< std::endl;
			/*theta_file << "t = " << dt*time_index << " T = " << dt*time_index/T_integral << std::endl;*/	
			INVERSE_TRANSFORM(phiphysical_cnt_step, phihat0 , NP1, NP2, NPH2P1);
			INVERSE_TRANSFORM(ucphysical_cnt_step, uchat0 , NP1, NP2, NPH2P1);
			INVERSE_TRANSFORM(tphysical_cnt_step, that0 , NP1, NP2, NPH2P1);
			for (int i = 0; i < NP1; i++)
			{
			//double x = i*dx1;
				for(int j = 0; j < NP2; j++)
				{
				//double y = j*dx2;
					phiphysical_file <<std::fixed << std::setprecision(8)<< phiphysical_cnt_step[i][j] << std::endl;
					ucphysical_file <<std::fixed << std::setprecision(8)<< ucphysical_cnt_step[i][j] << std::endl;
					tphysical_file <<std::fixed << std::setprecision(8)<< tphysical_cnt_step[i][j] << std::endl;
				}
			}
			phiphysical_file.close();
			phiphysical_filename.str(std::string());
			ucphysical_file.close();
			ucphysical_filename.str(std::string());
			tphysical_file.close();
			tphysical_filename.str(std::string());
			counter = counter + 1;
		}
//=================================================================================================================================================================================================
//	Solving the phase field, concentration and temperature equations
		INVERSE_TRANSFORM(phiphysical ,phihat0, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(ucphysical , uchat0, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(tphysical ,that0, NP1, NP2, NPH2P1);
//	calculate i*k1*phi_hat and i*k2*phi_hat
		DERIVATIVE_FOURIER(fourier_dphidx, k1, phihat0, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_dphidy, k2, phihat0, NP1, NPH2P1);
//	calculate dphi/dx and dphi/dy
		INVERSE_TRANSFORM(dphidx, fourier_dphidx, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(dphidy, fourier_dphidy, NP1, NP2, NPH2P1);
//	calculate angle (theta) between dphi/dx and dphi/dy
		CALCULATE_ANGLE(theta, dphidx, dphidy, NP1, NP2);
//	calculate sigma (anisotropy)		
		CALCULATE_ANISOTROPY(sigma, delta, j, theta, theta0, NP1, NP2);
//	calculate anisotropic energy gradient epsilon		
		CALCULATE_ANISO_GRAD_ENERGY(epsilon, epsilon_mean, sigma, NP1, NP2);
//	calculate derivative of anisotropic energy gradient epsilon (depsilon/dtheta)
		CALCULATE_DERIVATIVE_ANISO_GRAD_ENERGY(depsilondtheta, epsilon_mean, j, delta, theta, theta0, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta, epsilon, depsilondtheta, NP1, NP2);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon, epsilon, epsilon, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2 in Fourier space
		FORWARD_TRANSFORM(epsilon_depsilondtheta_hat, multiply_epsilon_depsilondtheta, NP1, NP2, NPH2P1);
		FORWARD_TRANSFORM(epsilon_epsilon_hat, multiply_epsilon_epsilon, NP1, NP2, NPH2P1);
			
//=============================================
//	calculate term 1 (phase field phi variable)
//=============================================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_depsilondtheta, k2, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_depsilondtheta_filtered, fourier_ddy_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(dphidx_filtered, fourier_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_depsilondtheta_dphidx, ddy_epsilon_depsilondtheta_filtered, dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term1_hat, multiply_ddy_epsilon_depsilondtheta_dphidx, NP1, NP2, NPH2P1);
//==================
//	calculate term 2
//==================
		DERIVATIVE_FOURIER(fourier_ddy_dphidx, k2, fourier_dphidx, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter		
		FILTER_INVERSE_TRANSFORM(epsilon_depsilondtheta_filtered, epsilon_depsilondtheta_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddy_dphidx_filtered, fourier_ddy_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddy_dphidx, epsilon_depsilondtheta_filtered, ddy_dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term2_hat, multiply_epsilon_depsilondtheta_ddy_dphidx, NP1, NP2, NPH2P1);
//==================
//	calculate term 3
//==================
		//DERIVATIVE_FOURIER(fourier_dphidy_shifted, k2, phihat_shifted, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_depsilondtheta, k1, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter			
		FILTER_INVERSE_TRANSFORM(dphidy_filtered, fourier_dphidy, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_depsilondtheta_filtered, fourier_ddx_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_depsilondtheta_dphidy, ddx_epsilon_depsilondtheta_filtered, dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term3_hat, multiply_ddx_epsilon_depsilondtheta_dphidy, NP1, NP2, NPH2P1);
//==================
//	calculate term 4
//==================
		DERIVATIVE_FOURIER(fourier_ddx_dphidy, k1, fourier_dphidy, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_dphidy_filtered, fourier_ddx_dphidy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddx_dphidy, epsilon_depsilondtheta_filtered, ddx_dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term4_hat, multiply_epsilon_depsilondtheta_ddx_dphidy, NP1, NP2, NPH2P1);
//==================
//	calculate term 5
//==================
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(epsilon_epsilon_filtered, epsilon_epsilon_hat, wk, NP1, NP2, NPH2P1);
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_phi_hat,  kmod, phihat0, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(laplace_phi_filtered, laplace_phi_hat, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon_laplace_phi, epsilon_epsilon_filtered, laplace_phi_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term5_hat, multiply_epsilon_epsilon_laplace_phi, NP1, NP2, NPH2P1);
//==================
//	calculate term 6
//==================
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_epsilon, k1, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_epsilon_filtered, fourier_ddx_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_epsilon_dphidx, ddx_epsilon_epsilon_filtered, dphidx_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term6_hat, multiply_ddx_epsilon_epsilon_dphidx, NP1, NP2, NPH2P1);
//==================
//	calculate term 7
//==================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_epsilon, k2, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_epsilon_filtered, fourier_ddy_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_epsilon_dphidy, ddy_epsilon_epsilon_filtered, dphidy_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term7_hat, multiply_ddy_epsilon_epsilon_dphidy, NP1, NP2, NPH2P1);
//=============================================
//	calculate term 8 (phase field phi variable)
//=============================================
//	term8 in the physical space
		CALCULATE_PHI_T8_PHYSICAL(phi_term8, phiphysical, lambda, tphysical, mc_inf, ucphysical, NP1, NP2);
//	term8 in the Fourier space
		FORWARD_TRANSFORM(phi_term8_hat, phi_term8, NP1, NP2, NPH2P1);
//	calculate the RHS of the equation phi and temperature
		EQRHS_PHI(rhs_phi_hat_k1, phi_term1_hat, phi_term2_hat, phi_term3_hat, phi_term4_hat, phi_term5_hat, phi_term6_hat, phi_term7_hat, phi_term8_hat, NP1, NPH2P1);	

		FILTER_INVERSE_TRANSFORM(uc_filtered, uchat0, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modified_epsilon_epsilon_filtered[i][j] = (1.0/(tau0*((1.0/Le_numb)+(mc_inf*(1.0+(1.0-part_coeff_const)*uc_filtered[i][j])))))*(1.0/epsilon_epsilon_filtered[i][j]);
			}
		}
		FILTER_INVERSE_TRANSFORM(rhs_phi_filtered, rhs_phi_hat_k1, wk, NP1, NP2, NPH2P1);
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, inv_modified_epsilon_epsilon_filtered, rhs_phi_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_phi_hat_k1, multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, NP1, NP2, NPH2P1);
//======================================================CONCENTRATION EQUATION========================================================================
//	compute the concentration equation terms (substep 1)
//----------------------------
// compute NU1 in <substep 1>
//----------------------------
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_uc_hat,  kmod, uchat0, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(laplace_uc_filtered, laplace_uc_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(phi_filtered, phihat0, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				multiply_onemphi_lapuc_filtered[i][j] = (1.0-phi_filtered[i][j])*laplace_uc_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc1_hat, multiply_onemphi_lapuc_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU2 in <substep 1>
//----------------------------
		DERIVATIVE_FOURIER(fourier_ducdx, k1, uchat0, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdx_filtered, fourier_ducdx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidx_ducdx_filtered, dphidx_filtered, ducdx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc2_hat, multiply_dphidx_ducdx_filtered, NP1, NP2, NPH2P1);

//----------------------------
// compute NU3 in <substep 1>
//-----------------------------
		DERIVATIVE_FOURIER(fourier_ducdy, k2, uchat0, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdy_filtered, fourier_ducdy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidy_ducdy_filtered, dphidy_filtered, ducdy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc3_hat, multiply_dphidy_ducdy_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU4 in <substep 1>
//----------------------------
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				nuc4_filtered[i][j] = 0.5*(1.0+((1.0-part_coeff_const)*uc_filtered[i][j]))*rhs_phi_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc4_hat, nuc4_filtered, NP1, NP2, NPH2P1);
		
		// calculate RHS for the uc evolution equation
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NPH2P1; j++)
			{
				rhs_uc_hat_k1[i][j] = (0.5*D_uc_liquid*(nuc1_hat[i][j]-nuc2_hat[i][j]-nuc3_hat[i][j]))+nuc4_hat[i][j];
			}
		}
//-------------------------
//	perform the final convolution	
//-------------------------
		FILTER_INVERSE_TRANSFORM(rhs_uc_filtered, rhs_uc_hat_k1, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modcoeff_phi_filtered[i][j] = (1.0/((0.5*(1.0+part_coeff_const))-(0.5*(1.0-part_coeff_const)*phi_filtered[i][j])));
			}
		}
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_uc_inv_modcoeff_phi_filtered, inv_modcoeff_phi_filtered, rhs_uc_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_uc_hat_k1, multiply_rhs_uc_inv_modcoeff_phi_filtered, NP1, NP2, NPH2P1);
//==================================================TEMPERATURE EQUATION=========================================================
//	compute the temperature equation
		EQRHS_TEMPERATURE(rhs_tp_hat_k1, rhs_phi_hat_k1, NP1, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NPH2P1; j++)
			{
				phihat_sub[i][j] = phihat0[i][j]+1.0*dt*rhs_phi_hat_k1[i][j];
				uchat_sub[i][j] = uchat0[i][j]+1.0*dt*rhs_uc_hat_k1[i][j];
				that_sub[i][j] = exp(-1.0*dt*alpha*kmod[i][j]*kmod[i][j])*(that0[i][j]+1.0*dt*rhs_tp_hat_k1[i][j]);
			}
		}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//===================================
//	calculation for sub-step 2 (k2)
//===================================
		INVERSE_TRANSFORM(phiphysical ,phihat_sub, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(ucphysical , uchat_sub, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(tphysical ,that_sub, NP1, NP2, NPH2P1);
//	calculate i*k1*phi_hat and i*k2*phi_hat
		DERIVATIVE_FOURIER(fourier_dphidx, k1, phihat_sub, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_dphidy, k2, phihat_sub, NP1, NPH2P1);
//	calculate dphi/dx and dphi/dy
		INVERSE_TRANSFORM(dphidx, fourier_dphidx, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(dphidy, fourier_dphidy, NP1, NP2, NPH2P1);
//	calculate angle (theta) between dphi/dx and dphi/dy
		CALCULATE_ANGLE(theta, dphidx, dphidy, NP1, NP2);
//	calculate sigma (anisotropy)		
		CALCULATE_ANISOTROPY(sigma, delta, j, theta, theta0, NP1, NP2);
//	calculate anisotropic energy gradient epsilon		
		CALCULATE_ANISO_GRAD_ENERGY(epsilon, epsilon_mean, sigma, NP1, NP2);
//	calculate derivative of anisotropic energy gradient epsilon (depsilon/dtheta)
		CALCULATE_DERIVATIVE_ANISO_GRAD_ENERGY(depsilondtheta, epsilon_mean, j, delta, theta, theta0, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta, epsilon, depsilondtheta, NP1, NP2);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon, epsilon, epsilon, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2 in Fourier space
		FORWARD_TRANSFORM(epsilon_depsilondtheta_hat, multiply_epsilon_depsilondtheta, NP1, NP2, NPH2P1);
		FORWARD_TRANSFORM(epsilon_epsilon_hat, multiply_epsilon_epsilon, NP1, NP2, NPH2P1);
			
//=========================================================
//	calculate term 1 (phase field phi variable) (substep-2)
//=========================================================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_depsilondtheta, k2, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_depsilondtheta_filtered, fourier_ddy_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(dphidx_filtered, fourier_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_depsilondtheta_dphidx, ddy_epsilon_depsilondtheta_filtered, dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term1_hat, multiply_ddy_epsilon_depsilondtheta_dphidx, NP1, NP2, NPH2P1);
//===============================
//	calculate term 2 (substep-2)
//===============================
		DERIVATIVE_FOURIER(fourier_ddy_dphidx, k2, fourier_dphidx, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter		
		FILTER_INVERSE_TRANSFORM(epsilon_depsilondtheta_filtered, epsilon_depsilondtheta_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddy_dphidx_filtered, fourier_ddy_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddy_dphidx, epsilon_depsilondtheta_filtered, ddy_dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term2_hat, multiply_epsilon_depsilondtheta_ddy_dphidx, NP1, NP2, NPH2P1);
//===============================
//	calculate term 3 (substep-2)
//===============================
		//DERIVATIVE_FOURIER(fourier_dphidy_shifted, k2, phihat_shifted, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_depsilondtheta, k1, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter			
		FILTER_INVERSE_TRANSFORM(dphidy_filtered, fourier_dphidy, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_depsilondtheta_filtered, fourier_ddx_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_depsilondtheta_dphidy, ddx_epsilon_depsilondtheta_filtered, dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term3_hat, multiply_ddx_epsilon_depsilondtheta_dphidy, NP1, NP2, NPH2P1);
//===============================
//	calculate term 4 (substep-2)
//===============================
		DERIVATIVE_FOURIER(fourier_ddx_dphidy, k1, fourier_dphidy, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_dphidy_filtered, fourier_ddx_dphidy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddx_dphidy, epsilon_depsilondtheta_filtered, ddx_dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term4_hat, multiply_epsilon_depsilondtheta_ddx_dphidy, NP1, NP2, NPH2P1);
//===============================
//	calculate term 5 (substep-2)
//===============================
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(epsilon_epsilon_filtered, epsilon_epsilon_hat, wk, NP1, NP2, NPH2P1);
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_phi_hat,  kmod, phihat_sub, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(laplace_phi_filtered, laplace_phi_hat, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon_laplace_phi, epsilon_epsilon_filtered, laplace_phi_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term5_hat, multiply_epsilon_epsilon_laplace_phi, NP1, NP2, NPH2P1);
//===============================
//	calculate term 6 (substep-2)
//===============================
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_epsilon, k1, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_epsilon_filtered, fourier_ddx_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_epsilon_dphidx, ddx_epsilon_epsilon_filtered, dphidx_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term6_hat, multiply_ddx_epsilon_epsilon_dphidx, NP1, NP2, NPH2P1);
//==============================
//	calculate term 7 (substep-2)
//==============================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_epsilon, k2, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_epsilon_filtered, fourier_ddy_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_epsilon_dphidy, ddy_epsilon_epsilon_filtered, dphidy_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term7_hat, multiply_ddy_epsilon_epsilon_dphidy, NP1, NP2, NPH2P1);
//========================================================
//	calculate term 8 (phase field phi variable)(substep-2)
//========================================================
//	term8 in the physical space
		CALCULATE_PHI_T8_PHYSICAL(phi_term8, phiphysical, lambda, tphysical, mc_inf, ucphysical, NP1, NP2);
//	term8 in the Fourier space
		FORWARD_TRANSFORM(phi_term8_hat, phi_term8, NP1, NP2, NPH2P1);
//	calculate the RHS of the equation phi and temperature
		EQRHS_PHI(rhs_phi_hat_k2, phi_term1_hat, phi_term2_hat, phi_term3_hat, phi_term4_hat, phi_term5_hat, phi_term6_hat, phi_term7_hat, phi_term8_hat, NP1, NPH2P1);	

		FILTER_INVERSE_TRANSFORM(uc_filtered, uchat_sub, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modified_epsilon_epsilon_filtered[i][j] = (1.0/(tau0*((1.0/Le_numb)+(mc_inf*(1.0+(1.0-part_coeff_const)*uc_filtered[i][j])))))*(1.0/epsilon_epsilon_filtered[i][j]);
			}
		}
		FILTER_INVERSE_TRANSFORM(rhs_phi_filtered, rhs_phi_hat_k2, wk, NP1, NP2, NPH2P1);
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, inv_modified_epsilon_epsilon_filtered, rhs_phi_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_phi_hat_k2, multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, NP1, NP2, NPH2P1);
//======================================================CONCENTRATION EQUATION========================================================================
//	compute the concentration equation terms (substep 2)
//----------------------------
// compute NU1 in <substep 2>
//----------------------------
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_uc_hat,  kmod, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(laplace_uc_filtered, laplace_uc_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(phi_filtered, phihat_sub, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				multiply_onemphi_lapuc_filtered[i][j] = (1.0-phi_filtered[i][j])*laplace_uc_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc1_hat, multiply_onemphi_lapuc_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU2 in <substep 2>
//----------------------------
		DERIVATIVE_FOURIER(fourier_ducdx, k1, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdx_filtered, fourier_ducdx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidx_ducdx_filtered, dphidx_filtered, ducdx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc2_hat, multiply_dphidx_ducdx_filtered, NP1, NP2, NPH2P1);

//----------------------------
// compute NU3 in <substep 2>
//-----------------------------
		DERIVATIVE_FOURIER(fourier_ducdy, k2, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdy_filtered, fourier_ducdy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidy_ducdy_filtered, dphidy_filtered, ducdy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc3_hat, multiply_dphidy_ducdy_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU4 in <substep 2>
//----------------------------
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				nuc4_filtered[i][j] = 0.5*(1.0+((1.0-part_coeff_const)*uc_filtered[i][j]))*rhs_phi_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc4_hat, nuc4_filtered, NP1, NP2, NPH2P1);
		
		// calculate RHS for the uc evolution equation
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NPH2P1; j++)
			{
				rhs_uc_hat_k2[i][j] = (0.5*D_uc_liquid*(nuc1_hat[i][j]-nuc2_hat[i][j]-nuc3_hat[i][j]))+nuc4_hat[i][j];
			}
		}
//--------------------------------
//	perform the final convolution	
//--------------------------------
		FILTER_INVERSE_TRANSFORM(rhs_uc_filtered, rhs_uc_hat_k2, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modcoeff_phi_filtered[i][j] = (1.0/((0.5*(1.0+part_coeff_const))-(0.5*(1.0-part_coeff_const)*phi_filtered[i][j])));
			}
		}
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_uc_inv_modcoeff_phi_filtered, inv_modcoeff_phi_filtered, rhs_uc_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_uc_hat_k2, multiply_rhs_uc_inv_modcoeff_phi_filtered, NP1, NP2, NPH2P1);
//==================================================TEMPERATURE EQUATION=========================================================
//	compute the temperature equation
		EQRHS_TEMPERATURE(rhs_tp_hat_k2, rhs_phi_hat_k2, NP1, NPH2P1);

		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NPH2P1; j++)
			{
				phihat_sub[i][j] = (3.0/4.0)*phihat0[i][j]+(1.0/4.0)*phihat_sub[i][j]+(1.0/4.0)*dt*rhs_phi_hat_k2[i][j];
				uchat_sub[i][j] = (3.0/4.0)*uchat0[i][j]+(1.0/4.0)*uchat_sub[i][j]+(1.0/4.0)*dt*rhs_uc_hat_k2[i][j];
				that_sub[i][j] = (3.0/4.0)*exp(-0.5*dt*alpha*kmod[i][j]*kmod[i][j])*that0[i][j]+exp(0.5*dt*(alpha)*kmod[i][j]*kmod[i][j])*((1.0/4.0)*that_sub[i][j]+(1.0/4.0)*dt*rhs_tp_hat_k2[i][j]);
			}
		}
		
//================================
//	calculation for sub-step 3 (k3)
//================================
		INVERSE_TRANSFORM(phiphysical ,phihat_sub, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(ucphysical , uchat_sub, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(tphysical ,that_sub, NP1, NP2, NPH2P1);
//	calculate i*k1*phi_hat and i*k2*phi_hat
		DERIVATIVE_FOURIER(fourier_dphidx, k1, phihat_sub, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_dphidy, k2, phihat_sub, NP1, NPH2P1);
//	calculate dphi/dx and dphi/dy
		INVERSE_TRANSFORM(dphidx, fourier_dphidx, NP1, NP2, NPH2P1);
		INVERSE_TRANSFORM(dphidy, fourier_dphidy, NP1, NP2, NPH2P1);
//	calculate angle (theta) between dphi/dx and dphi/dy
		CALCULATE_ANGLE(theta, dphidx, dphidy, NP1, NP2);
//	calculate sigma (anisotropy)		
		CALCULATE_ANISOTROPY(sigma, delta, j, theta, theta0, NP1, NP2);
//	calculate anisotropic energy gradient epsilon		
		CALCULATE_ANISO_GRAD_ENERGY(epsilon, epsilon_mean, sigma, NP1, NP2);
//	calculate derivative of anisotropic energy gradient epsilon (depsilon/dtheta)
		CALCULATE_DERIVATIVE_ANISO_GRAD_ENERGY(depsilondtheta, epsilon_mean, j, delta, theta, theta0, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta, epsilon, depsilondtheta, NP1, NP2);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon, epsilon, epsilon, NP1, NP2);
// calculate epsilon*depsilon/dtheta & epsilon^2 in Fourier space
		FORWARD_TRANSFORM(epsilon_depsilondtheta_hat, multiply_epsilon_depsilondtheta, NP1, NP2, NPH2P1);
		FORWARD_TRANSFORM(epsilon_epsilon_hat, multiply_epsilon_epsilon, NP1, NP2, NPH2P1);
			
//=========================================================
//	calculate term 1 (phase field phi variable) (substep-3)
//==========================================================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_depsilondtheta, k2, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_depsilondtheta_filtered, fourier_ddy_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(dphidx_filtered, fourier_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_depsilondtheta_dphidx, ddy_epsilon_depsilondtheta_filtered, dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term1_hat, multiply_ddy_epsilon_depsilondtheta_dphidx, NP1, NP2, NPH2P1);
//==============================
//	calculate term 2 (substep-3)
//==============================
		DERIVATIVE_FOURIER(fourier_ddy_dphidx, k2, fourier_dphidx, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter		
		FILTER_INVERSE_TRANSFORM(epsilon_depsilondtheta_filtered, epsilon_depsilondtheta_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddy_dphidx_filtered, fourier_ddy_dphidx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddy_dphidx, epsilon_depsilondtheta_filtered, ddy_dphidx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term2_hat, multiply_epsilon_depsilondtheta_ddy_dphidx, NP1, NP2, NPH2P1);
//==============================
//	calculate term 3 (substep-3)
//==============================
		//DERIVATIVE_FOURIER(fourier_dphidy_shifted, k2, phihat_shifted, NP1, NPH2P1);
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_depsilondtheta, k1, epsilon_depsilondtheta_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter			
		FILTER_INVERSE_TRANSFORM(dphidy_filtered, fourier_dphidy, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_depsilondtheta_filtered, fourier_ddx_epsilon_depsilondtheta, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_depsilondtheta_dphidy, ddx_epsilon_depsilondtheta_filtered, dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term3_hat, multiply_ddx_epsilon_depsilondtheta_dphidy, NP1, NP2, NPH2P1);
//=============================
//	calculate term 4 (substep-3)
//==============================
		DERIVATIVE_FOURIER(fourier_ddx_dphidy, k1, fourier_dphidy, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_dphidy_filtered, fourier_ddx_dphidy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_depsilondtheta_ddx_dphidy, epsilon_depsilondtheta_filtered, ddx_dphidy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(phi_term4_hat, multiply_epsilon_depsilondtheta_ddx_dphidy, NP1, NP2, NPH2P1);
//==============================
//	calculate term 5 (substep-3)
//===============================
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(epsilon_epsilon_filtered, epsilon_epsilon_hat, wk, NP1, NP2, NPH2P1);
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_phi_hat,  kmod, phihat_sub, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(laplace_phi_filtered, laplace_phi_hat, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_epsilon_epsilon_laplace_phi, epsilon_epsilon_filtered, laplace_phi_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term5_hat, multiply_epsilon_epsilon_laplace_phi, NP1, NP2, NPH2P1);
//==============================
//	calculate term 6 (substep-3)
//==============================
		DERIVATIVE_FOURIER(fourier_ddx_epsilon_epsilon, k1, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddx_epsilon_epsilon_filtered, fourier_ddx_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddx_epsilon_epsilon_dphidx, ddx_epsilon_epsilon_filtered, dphidx_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term6_hat, multiply_ddx_epsilon_epsilon_dphidx, NP1, NP2, NPH2P1);
//==============================
//	calculate term 7 (substep-3)
//==============================
		DERIVATIVE_FOURIER(fourier_ddy_epsilon_epsilon, k2, epsilon_epsilon_hat, NP1, NPH2P1);
//......use the filter inverse transform considering the spectral window filter
		FILTER_INVERSE_TRANSFORM(ddy_epsilon_epsilon_filtered, fourier_ddy_epsilon_epsilon, wk, NP1, NP2, NPH2P1);
//	multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_ddy_epsilon_epsilon_dphidy, ddy_epsilon_epsilon_filtered, dphidy_filtered, NP1, NP2);
//	forward transform
		FORWARD_TRANSFORM(phi_term7_hat, multiply_ddy_epsilon_epsilon_dphidy, NP1, NP2, NPH2P1);
//=========================================================
//	calculate term 8 (phase field phi variable) (substep-3)
//=========================================================
//	term8 in the physical space
		CALCULATE_PHI_T8_PHYSICAL(phi_term8, phiphysical, lambda, tphysical, mc_inf, ucphysical, NP1, NP2);
//	term8 in the Fourier space
		FORWARD_TRANSFORM(phi_term8_hat, phi_term8, NP1, NP2, NPH2P1);
//	calculate the RHS of the equation phi and temperature
		EQRHS_PHI(rhs_phi_hat_k3, phi_term1_hat, phi_term2_hat, phi_term3_hat, phi_term4_hat, phi_term5_hat, phi_term6_hat, phi_term7_hat, phi_term8_hat, NP1, NPH2P1);	

		FILTER_INVERSE_TRANSFORM(uc_filtered, uchat_sub, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modified_epsilon_epsilon_filtered[i][j] = (1.0/(tau0*((1.0/Le_numb)+(mc_inf*(1.0+(1.0-part_coeff_const)*uc_filtered[i][j])))))*(1.0/epsilon_epsilon_filtered[i][j]);
			}
		}
		FILTER_INVERSE_TRANSFORM(rhs_phi_filtered, rhs_phi_hat_k3, wk, NP1, NP2, NPH2P1);
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, inv_modified_epsilon_epsilon_filtered, rhs_phi_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_phi_hat_k3, multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered, NP1, NP2, NPH2P1);
//======================================================CONCENTRATION EQUATION========================================================================
//	compute the concentration equation terms (substep 3)
//----------------------------
// compute NU1 in <substep 3>
//----------------------------
		CALCULATE_SPECTRAL_LAPLACIAN(laplace_uc_hat,  kmod, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(laplace_uc_filtered, laplace_uc_hat, wk, NP1, NP2, NPH2P1);
		FILTER_INVERSE_TRANSFORM(phi_filtered, phihat_sub, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				multiply_onemphi_lapuc_filtered[i][j] = (1.0-phi_filtered[i][j])*laplace_uc_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc1_hat, multiply_onemphi_lapuc_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU2 in <substep 3>
//----------------------------
		DERIVATIVE_FOURIER(fourier_ducdx, k1, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdx_filtered, fourier_ducdx, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidx_ducdx_filtered, dphidx_filtered, ducdx_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc2_hat, multiply_dphidx_ducdx_filtered, NP1, NP2, NPH2P1);

//----------------------------
// compute NU3 in <substep 3>
//-----------------------------
		DERIVATIVE_FOURIER(fourier_ducdy, k2, uchat_sub, NP1, NPH2P1);
		FILTER_INVERSE_TRANSFORM(ducdy_filtered, fourier_ducdy, wk, NP1, NP2, NPH2P1);
		MULTIPLY_PHYSICAL_SPACE(multiply_dphidy_ducdy_filtered, dphidy_filtered, ducdy_filtered, NP1, NP2);
		FORWARD_TRANSFORM(nuc3_hat, multiply_dphidy_ducdy_filtered, NP1, NP2, NPH2P1);
//----------------------------
// compute NU4 in <substep 3>
//----------------------------
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				nuc4_filtered[i][j] = 0.5*(1.0+((1.0-part_coeff_const)*uc_filtered[i][j]))*rhs_phi_filtered[i][j];
			}
		}
		FORWARD_TRANSFORM(nuc4_hat, nuc4_filtered, NP1, NP2, NPH2P1);
		
		// calculate RHS for the uc evolution equation
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NPH2P1; j++)
			{
				rhs_uc_hat_k3[i][j] = (0.5*D_uc_liquid*(nuc1_hat[i][j]-nuc2_hat[i][j]-nuc3_hat[i][j]))+nuc4_hat[i][j];
			}
		}
//--------------------------------------------
//	perform the final convolution (substep-3)	
//--------------------------------------------
		FILTER_INVERSE_TRANSFORM(rhs_uc_filtered, rhs_uc_hat_k3, wk, NP1, NP2, NPH2P1);
		for(int i=0; i<NP1 ;i++)
		{
			for(int j=0; j<NP2; j++)
			{
				inv_modcoeff_phi_filtered[i][j] = (1.0/((0.5*(1.0+part_coeff_const))-(0.5*(1.0-part_coeff_const)*phi_filtered[i][j])));
			}
		}
		//multiply
		MULTIPLY_PHYSICAL_SPACE(multiply_rhs_uc_inv_modcoeff_phi_filtered, inv_modcoeff_phi_filtered, rhs_uc_filtered, NP1, NP2);
		//fft
		FORWARD_TRANSFORM(rhs_uc_hat_k3, multiply_rhs_uc_inv_modcoeff_phi_filtered, NP1, NP2, NPH2P1);
//==================================================TEMPERATURE EQUATION=========================================================
//	compute the temperature equation
		EQRHS_TEMPERATURE(rhs_tp_hat_k3, rhs_phi_hat_k3, NP1, NPH2P1);
//=======================================================================================
		SSPRK3_TIME_STEP(phihat1, phihat0, phihat_sub, dt, rhs_phi_hat_k3, NP1, NPH2P1);
		SSPRK3_TIME_STEP(uchat1, uchat0, uchat_sub, dt, rhs_uc_hat_k3, NP1, NPH2P1);
		//SSPRK3_TIME_STEP(that1, that0, that_sub, dt, rhs_tp_hat_k3, NP1, NPH2P1);
		SSPRK3_TIME_STEP_IF(that1, that0, that_sub, kmod, alpha, dt, rhs_tp_hat_k3, NP1, NPH2P1);
		for(int i = 0; i < NP1 ; i++)
		{
			for(int j = 0; j < NPH2P1; j++)
			{
				phihat0[i][j] = phihat1[i][j];
				uchat0[i][j] = uchat1[i][j];
				that0[i][j] = that1[i][j];
			}
		}
	}//end of time loop
	std::cout << "Finishing time steps" << std::endl;

	phiphysicalf_filename << output_dir << fname_prefix << "_phi" << counter << ".dat";
	ofstream phiphysicalf_file(phiphysicalf_filename.str().c_str(), ios::out | ios::trunc);

	ucphysicalf_filename << output_dir << fname_prefix << "_uc" << counter << ".dat";
	ofstream ucphysicalf_file(ucphysicalf_filename.str().c_str(), ios::out | ios::trunc);

	tphysicalf_filename << output_dir << fname_prefix << "_t" << counter << ".dat";
	ofstream tphysicalf_file(tphysicalf_filename.str().c_str(), ios::out | ios::trunc);	

	INVERSE_TRANSFORM(phiphysical_cnt_step, phihat0, NP1, NP2, NPH2P1);
	INVERSE_TRANSFORM(ucphysical_cnt_step, uchat0, NP1, NP2, NPH2P1);
	INVERSE_TRANSFORM(tphysical_cnt_step, that0, NP1, NP2, NPH2P1);
	for (int i = 0; i < NP1; i++)
	{
	//double x = i*dx1;
		for (int j = 0; j < NP2; j++)
		{
			//double y = j*dx2;
			phiphysicalf_file <<std::fixed << std::setprecision(8)<< phiphysical_cnt_step[i][j] << std::endl;
			ucphysicalf_file <<std::fixed << std::setprecision(8)<< ucphysical_cnt_step[i][j] << std::endl;
			tphysicalf_file <<std::fixed << std::setprecision(8)<< tphysical_cnt_step[i][j] << std::endl;
		}
	}
	std::cout << "Deleting the allocatted memory" << std::endl;
	for (int i = 0; i < NP1; i++)
	{
		delete [] k1[i];
		delete [] k2[i];
		delete [] kmod[i];
		delete [] wk[i];		
		delete [] phiphysical[i];
		delete [] ucphysical[i];
		delete [] tphysical[i];
		delete [] phihat0[i];
		delete [] uchat0[i];
		delete [] that0[i];
		delete [] phiphysical_cnt_step[i];
		delete [] ucphysical_cnt_step[i];
		delete [] tphysical_cnt_step[i];
		delete [] fourier_dphidx[i];
		delete [] fourier_dphidy[i];
		delete [] dphidx[i];
		delete [] dphidy[i];
		delete [] theta[i];
		delete [] sigma[i];
		delete [] epsilon[i];
		delete [] depsilondtheta[i];
		delete [] multiply_epsilon_depsilondtheta[i];
		delete [] multiply_epsilon_epsilon[i];
		delete [] epsilon_depsilondtheta_hat[i];
		delete [] epsilon_epsilon_hat[i];
		delete [] fourier_ddy_epsilon_depsilondtheta[i];
		delete [] ddy_epsilon_depsilondtheta_filtered[i];
		delete [] dphidx_filtered[i];
		delete [] multiply_ddy_epsilon_depsilondtheta_dphidx[i];
		delete [] phi_term1_hat[i];
		delete [] fourier_ddy_dphidx[i];
		delete [] epsilon_depsilondtheta_filtered[i];
		delete [] ddy_dphidx_filtered[i];
		delete [] multiply_epsilon_depsilondtheta_ddy_dphidx[i];
		delete [] phi_term2_hat[i];
		delete [] fourier_ddx_epsilon_depsilondtheta[i];
		delete [] dphidy_filtered[i];
		delete [] ddx_epsilon_depsilondtheta_filtered[i];
		delete [] multiply_ddx_epsilon_depsilondtheta_dphidy[i];
		delete [] phi_term3_hat[i];
		delete [] fourier_ddx_dphidy[i];
		delete [] ddx_dphidy_filtered[i];
		delete [] multiply_epsilon_depsilondtheta_ddx_dphidy[i];
		delete [] phi_term4_hat[i];
		delete [] epsilon_epsilon_filtered[i];
		delete [] laplace_phi_hat[i];
		delete [] laplace_phi_filtered[i];
		delete [] multiply_epsilon_epsilon_laplace_phi[i];
		delete [] phi_term5_hat[i];
		delete [] fourier_ddx_epsilon_epsilon[i];
		delete [] ddx_epsilon_epsilon_filtered[i];
		delete [] multiply_ddx_epsilon_epsilon_dphidx[i];
		delete [] phi_term6_hat[i];
		delete [] fourier_ddy_epsilon_epsilon[i];
		delete [] ddy_epsilon_epsilon_filtered[i];
		delete [] multiply_ddy_epsilon_epsilon_dphidy[i];
		delete [] phi_term7_hat[i];
		delete [] phi_term8[i];
		delete [] phi_term8_hat[i];
		delete [] uc_filtered[i];
		delete [] inv_modified_epsilon_epsilon_filtered[i];	
		delete [] rhs_phi_filtered[i];
		delete [] multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered[i];
		delete [] rhs_phi_hat_k1[i];
		delete [] rhs_phi_hat_k2[i];
		delete [] rhs_phi_hat_k3[i];
		delete [] laplace_uc_hat[i];		
		delete [] laplace_uc_filtered[i];
		delete [] phi_filtered[i];
		delete [] multiply_onemphi_lapuc_filtered[i];
		delete [] nuc1_hat[i];	
		delete [] fourier_ducdx[i];		
		delete [] ducdx_filtered[i];
		delete [] multiply_dphidx_ducdx_filtered[i];
		delete [] nuc2_hat[i];
		delete [] fourier_ducdy[i];		
		delete [] ducdy_filtered[i];
		delete [] multiply_dphidy_ducdy_filtered[i];
		delete [] nuc3_hat[i];		
		delete [] nuc4_filtered[i];
		delete [] nuc4_hat[i];			
		delete [] rhs_uc_filtered[i];
		delete [] inv_modcoeff_phi_filtered[i];
		delete [] multiply_rhs_uc_inv_modcoeff_phi_filtered[i];		
		delete [] rhs_uc_hat_k1[i];
		delete [] rhs_uc_hat_k2[i];
		delete [] rhs_uc_hat_k3[i];			
		delete [] rhs_tp_hat_k1[i];
		delete [] rhs_tp_hat_k2[i];
		delete [] rhs_tp_hat_k3[i];
		delete [] phihat_sub[i];
		delete [] uchat_sub[i];
		delete [] that_sub[i];
		delete [] phihat1[i];
		delete [] uchat1[i];
		delete [] that1[i];
	}
		delete [] k1;
		delete [] k2;
		delete [] kmod;
		delete [] wk;
		delete [] phiphysical;
		delete [] ucphysical;
		delete [] tphysical;
		delete [] phihat0;
		delete [] uchat0;
		delete [] that0;
		delete [] phiphysical_cnt_step;
		delete [] ucphysical_cnt_step;
		delete [] tphysical_cnt_step;
		delete [] fourier_dphidx;
		delete [] fourier_dphidy;
		delete [] dphidx;
		delete [] dphidy;
		delete [] theta;
		delete [] sigma;
		delete [] epsilon;
		delete [] depsilondtheta;
		delete [] multiply_epsilon_depsilondtheta;
		delete [] multiply_epsilon_epsilon;
		delete [] epsilon_depsilondtheta_hat;
		delete [] epsilon_epsilon_hat;
		delete [] fourier_ddy_epsilon_depsilondtheta;
		delete [] ddy_epsilon_depsilondtheta_filtered;
		delete [] dphidx_filtered;
		delete [] multiply_ddy_epsilon_depsilondtheta_dphidx;
		delete [] phi_term1_hat;
		delete [] fourier_ddy_dphidx;
		delete [] epsilon_depsilondtheta_filtered;
		delete [] ddy_dphidx_filtered;
		delete [] multiply_epsilon_depsilondtheta_ddy_dphidx;
		delete [] phi_term2_hat;
		delete [] fourier_ddx_epsilon_depsilondtheta;
		delete [] dphidy_filtered;
		delete [] ddx_epsilon_depsilondtheta_filtered;
		delete [] multiply_ddx_epsilon_depsilondtheta_dphidy;
		delete [] phi_term3_hat;
		delete [] fourier_ddx_dphidy;
		delete [] ddx_dphidy_filtered;
		delete [] multiply_epsilon_depsilondtheta_ddx_dphidy;
		delete [] phi_term4_hat;
		delete [] epsilon_epsilon_filtered;
		delete [] laplace_phi_hat;
		delete [] laplace_phi_filtered;
		delete [] multiply_epsilon_epsilon_laplace_phi;
		delete [] phi_term5_hat;
		delete [] fourier_ddx_epsilon_epsilon;
		delete [] ddx_epsilon_epsilon_filtered;
		delete [] multiply_ddx_epsilon_epsilon_dphidx;
		delete [] phi_term6_hat;
		delete [] fourier_ddy_epsilon_epsilon;
		delete [] ddy_epsilon_epsilon_filtered;
		delete [] multiply_ddy_epsilon_epsilon_dphidy;
		delete [] phi_term7_hat;
		delete [] phi_term8;
		delete [] phi_term8_hat;
		delete [] uc_filtered;
		delete [] inv_modified_epsilon_epsilon_filtered;	
		delete [] rhs_phi_filtered;
		delete [] multiply_rhs_phi_inv_modified_epsilon_epsilon_filtered;
		delete [] rhs_phi_hat_k1;
		delete [] rhs_phi_hat_k2;
		delete [] rhs_phi_hat_k3;
		delete [] laplace_uc_hat;		
		delete [] laplace_uc_filtered;
		delete [] phi_filtered;
		delete [] multiply_onemphi_lapuc_filtered;
		delete [] nuc1_hat;	
		delete [] fourier_ducdx;		
		delete [] ducdx_filtered;
		delete [] multiply_dphidx_ducdx_filtered;
		delete [] nuc2_hat;
		delete [] fourier_ducdy;		
		delete [] ducdy_filtered;
		delete [] multiply_dphidy_ducdy_filtered;
		delete [] nuc3_hat;		
		delete [] nuc4_filtered;
		delete [] nuc4_hat;		
		delete [] rhs_uc_filtered;
		delete [] inv_modcoeff_phi_filtered;
		delete [] multiply_rhs_uc_inv_modcoeff_phi_filtered;			
		delete [] rhs_uc_hat_k1;
		delete [] rhs_uc_hat_k2;
		delete [] rhs_uc_hat_k3;				
		delete [] rhs_tp_hat_k1;
		delete [] rhs_tp_hat_k2;
		delete [] rhs_tp_hat_k3;
		delete [] phihat_sub;
		delete [] uchat_sub;
		delete [] that_sub;
		delete [] phihat1;
		delete [] uchat1;
		delete [] that1;
	
		phiphysical0_file.close();
		phiphysicalf_file.close();
		ucphysical0_file.close();
		ucphysicalf_file.close();
		tphysical0_file.close();
		tphysicalf_file.close();
 		runstatus_file.close();
		free(output_dir);
		free(fname_prefix);
		return(0);
	}// end of main program

//----------------------------------------------------------Subroutines----------------------------------------------------//
//	calculate the derivative in spectral space
void DERIVATIVE_FOURIER(std::complex<double>** fourier_ux_temp, double** k1_temp, std::complex<double>** uhat0_temp, int NP1_temp, int NP2_temp)
{

//	calculate the derivative in spectral space in 2D (for u,v,p)
	for (int i=0; i<NP1_temp; i++) 
	{
		for (int j=0; j<NP2_temp; j++)
		{
			fourier_ux_temp[i][j] =std::complex<double>(0.0,1.0)*k1_temp[i][j]*uhat0_temp[i][j];
		}
	}
}
//=============================================================================
//	calculation of u,v,p in physical space (inverse fast fourier transform)
//=============================================================================
void INVERSE_TRANSFORM(double** uout1_final_temp, std::complex<double>** uc_temp, int NP1_temp, int NP2_temp, int NPH2P1_temp)
{
	fftw_complex *uin1;
	double *uout1;
	uin1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NP1_temp*NPH2P1_temp);
	uout1 = (double*)malloc(sizeof(double)*NP1_temp*NP2_temp);
	fftw_plan invfu = fftw_plan_dft_c2r_2d(NP1_temp, NP2_temp, uin1, uout1, FFTW_MEASURE);
	int mm = 0;
	for (int i = 0; i < NP1_temp; i++)
	{
		for(int j = 0; j < NPH2P1_temp; j++)
		{
			uin1[mm][0] = real(uc_temp[i][j]);
			uin1[mm][1] = imag(uc_temp[i][j]);
			mm++;	
		}
	}
	fftw_execute(invfu);
	mm = 0;
	for (int i = 0; i < NP1_temp; i++)
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			uout1_final_temp[i][j] = uout1[mm];
			mm++;	
		}
	}
	fftw_destroy_plan(invfu);
	fftw_free(uin1);
	delete[] uout1;
}
//==========================================================================================
// Filtered Inverse Transform for calculating u,v and p in physical space [W(k) is involved]
//==========================================================================================
void FILTER_INVERSE_TRANSFORM(double** uout1_filt_final_temp, std::complex<double>** uc_filt_temp, std::complex<double>** wk_filt_temp, int NP1_temp, int NP2_temp, int NPHP1_temp)
{
	fftw_complex *uin1_filt;
	double *uout1_filt;
	uin1_filt=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NP1_temp*NPHP1_temp);
	uout1_filt=(double*)malloc(sizeof(double)*NP1_temp*NP2_temp);
	fftw_plan invfu_filt = fftw_plan_dft_c2r_2d(NP1_temp,NP2_temp,uin1_filt,uout1_filt,FFTW_MEASURE);
	int mm=0;
	for (int i=0; i<NP1_temp; i++)
	{
		for(int j=0; j<NPHP1_temp; j++)
		{
			uin1_filt[mm][0]=real(uc_filt_temp[i][j]*wk_filt_temp[i][j]);
			uin1_filt[mm][1]=imag(uc_filt_temp[i][j]*wk_filt_temp[i][j]);
			mm++;	
		}
	}
	fftw_execute(invfu_filt);
	mm=0;
	for (int i=0; i<NP1_temp; i++)
	{
		for(int j=0; j<NP2_temp; j++)
		{
			uout1_filt_final_temp[i][j]= uout1_filt[mm];
			mm++;	
		}
	}
	fftw_destroy_plan(invfu_filt);
	fftw_free(uin1_filt);
	delete [] uout1_filt;
}
//========================================================
//create a function which calculate the forward transform
//========================================================
void FORWARD_TRANSFORM(std::complex<double>** nlt_final_temp, double** uux_temp, int NP1_temp, int NP2_temp, int NPH2P1_temp)
{
	double scale_factor = 1.0/(NP1_temp*NP2_temp);
	double *uuxin1;
	fftw_complex *uuxout1;
	uuxin1 = (double*)malloc(sizeof(double)*NP1_temp*NP2_temp);
	uuxout1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NP1_temp*NPH2P1_temp);
	fftw_plan fftnl = fftw_plan_dft_r2c_2d(NP1_temp,NP2_temp,uuxin1,uuxout1, FFTW_MEASURE);
	int mm = 0;
	for (int i = 0; i < NP1_temp; i++)
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			uuxin1[mm] = uux_temp[i][j];
			mm++;	
		}
	}
	fftw_execute(fftnl);
	mm = 0;
	for (int i = 0; i < NP1_temp; i++)
	{
		for(int j = 0; j < NPH2P1_temp; j++)
		{
			nlt_final_temp[i][j] = std::complex<double>(uuxout1[mm][0], uuxout1[mm][1])*scale_factor;
			mm++;	
		}
	}
	fftw_destroy_plan(fftnl);
	delete[] uuxin1;
	fftw_free(uuxout1);
}
//===========================================
// calculate multiplication in physical space
//===========================================
void MULTIPLY_PHYSICAL_SPACE(double** mulphysical_temp, double** uphysical_temp ,double** uxphysical_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			mulphysical_temp[i][j] = uphysical_temp[i][j]*uxphysical_temp[i][j];
		}
	}
}
//================
// calculate angle
//================
void CALCULATE_ANGLE(double** theta_temp, double** dphidx_temp, double** dphidy_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			if(dphidx_temp[i][j] == 0.0)
			{
				theta_temp[i][j] = (dphidy_temp[i][j]>0 ? 0.5*PI : -0.5*PI);
			}
			else if(dphidx_temp[i][j] > 0.0)
			{
				theta_temp[i][j] = (dphidy_temp[i][j]>0 ? atan(dphidy_temp[i][j]/dphidx_temp[i][j]) : 2.0*PI+atan(dphidy_temp[i][j]/dphidx_temp[i][j]));
			}
			else
			{
				theta_temp[i][j] = 1.0*PI+atan(dphidy_temp[i][j]/dphidx_temp[i][j]);
			}
		}
	}
}
//======================================
// calculate anisotropy in physical space
//======================================
void CALCULATE_ANISOTROPY(double** sigma_temp, double delta_temp, double j_temp, double** theta_temp, double theta0_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			sigma_temp[i][j] = 1.0 + delta_temp*cos(j_temp*(theta_temp[i][j] - theta0_temp));
		}
	}
}
//==================================================================
// calculate anisotropic gradient energy (epsilon) in physical space
//==================================================================
void CALCULATE_ANISO_GRAD_ENERGY(double** epsilon_temp, double epsilon_mean_temp, double** sigma_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			epsilon_temp[i][j] = epsilon_mean_temp*sigma_temp[i][j];
		}
	}
}
//============================================
// calculate depsilon/dtheta in physical space
//============================================
void CALCULATE_DERIVATIVE_ANISO_GRAD_ENERGY(double** depsdtheta_temp, double epsilon_mean_temp, double j_temp, double delta_temp, double** theta_temp, double theta0_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			depsdtheta_temp[i][j] = -1.0*epsilon_mean_temp*j_temp*delta_temp*sin(j_temp*(theta_temp[i][j] - theta0_temp));
		}
	}
}
//=========================================
// calculate TERM 8 (source term like term)
//=========================================
void CALCULATE_PHI_T8_PHYSICAL(double** t8_temp, double** phiphysical_temp, double lambda_temp, double** tphysical_temp, double mc_inf_temp, double** ucphysical_temp, int NP1_temp, int NP2_temp)
{
	for (int i = 0; i < NP1_temp; i++) 
	{
		for(int j = 0; j < NP2_temp; j++)
		{
			t8_temp[i][j] =  phiphysical_temp[i][j]*(1.0 - (phiphysical_temp[i][j]*phiphysical_temp[i][j]))-(lambda_temp*(tphysical_temp[i][j]+(mc_inf_temp*ucphysical_temp[i][j]))*pow((1.0 - (phiphysical_temp[i][j]*phiphysical_temp[i][j])),2));
		}
	}
}
//==================================
// calculation of the laplacian term
//==================================
void CALCULATE_SPECTRAL_LAPLACIAN(std::complex<double>** laplace_phi_hat_temp, double** k1_temp ,std::complex<double>** phihat_temp, int NP1_temp, int NP2_temp)
{
	for (int i=0; i<NP1_temp; i++)
	{
		for(int j=0; j<NP2_temp; j++)
		{
			laplace_phi_hat_temp[i][j] = -1.0*pow(k1_temp[i][j],2)*phihat_temp[i][j];
		}
	}
}
//===========================================
// calculation of the rhs of the phi equation
//===========================================
void EQRHS_PHI(std::complex<double>** rhs_phi_hat_temp, std::complex<double>** t1_hat_temp, std::complex<double>** t2_hat_temp, std::complex<double>** t3_hat_temp, std::complex<double>** t4_hat_temp, std::complex<double>** t5_hat_temp, std::complex<double>** t6_hat_temp, std::complex<double>** t7_hat_temp, std::complex<double>** t8_hat_temp, int NP1_temp, int NP2_temp)
{
	for (int i=0; i<NP1_temp; i++)
	{
		for(int j=0; j<NP2_temp; j++)
		{
			rhs_phi_hat_temp[i][j] = (t1_hat_temp[i][j]+t2_hat_temp[i][j]-t3_hat_temp[i][j]-t4_hat_temp[i][j]+t5_hat_temp[i][j]+t6_hat_temp[i][j]+t7_hat_temp[i][j]+t8_hat_temp[i][j]);
		}
	}
}
//===================================================
// calculation of the rhs of the temperature equation
//===================================================
void EQRHS_TEMPERATURE(std::complex<double>** rhs_t_hat_temp, std::complex<double>** rhs_phi_hat_temp, int NP1_temp, int NP2_temp)
{
	for (int i=0; i<NP1_temp; i++)
	{
		for(int j=0; j<NP2_temp; j++)
		{
			rhs_t_hat_temp[i][j] =  (0.5*rhs_phi_hat_temp[i][j]);
		}
	}
}
// calculation of the final timestep SSPRK3
void SSPRK3_TIME_STEP(std::complex<double>** uhat1_temp, std::complex<double>** uhat0_temp, std::complex<double>** uhat_sub_temp, double dt_temp, std::complex<double>** rhs_hat_k3_temp, int NP1_temp, int NP2_temp)
{
        for (int i=0; i<NP1_temp; i++)
        {
        		for(int j=0; j<NP2_temp; j++)
				{
                         uhat1_temp[i][j]=((1.0/3.0)*uhat0_temp[i][j])+((2.0/3.0)*uhat_sub_temp[i][j])+(2.0/3.0)*dt_temp*rhs_hat_k3_temp[i][j];
				}
         }

}
// calculation of the final timestep SSPRK3
void SSPRK3_TIME_STEP_IF(std::complex<double>** that1_temp, std::complex<double>** that0_temp, std::complex<double>** that_sub_temp, double** kmod_temp, double alpha_temp, double dt_temp, std::complex<double>** rhs_hat_temp, int NP1_temp, int NP2_temp)
{
        for (int i=0; i<NP1_temp; i++)
        {
        		for(int j=0; j<NP2_temp; j++)
				{
                         that1_temp[i][j]=((1.0/3.0)*exp(-1.0*dt_temp*(alpha_temp)*kmod_temp[i][j]*kmod_temp[i][j])*that0_temp[i][j])+((2.0/3.0)*exp(-0.5*dt_temp*alpha_temp*kmod_temp[i][j]*kmod_temp[i][j])*that_sub_temp[i][j])+(2.0/3.0)*exp(-0.5*dt_temp*alpha_temp*kmod_temp[i][j]*kmod_temp[i][j])*dt_temp*rhs_hat_temp[i][j];
				}
         }

}
//============================================================================================================================================================================================================================================================================
