#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth)
{
	// Initialize Vector
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Check for Valid inputs
	if (estimations.size() != ground_truth.size())
	{
		std::cout << " Size Mismatch!" <<std::endl;
		return rmse;
	}

	if ((estimations.size()==0) || (ground_truth.size()==0))
	{
		std::cout << "Empty Vector!" <<std::endl;
		return rmse;
	}

	// Accumulate square residual error
	for(int i=0; i < estimations.size(); ++i){
		// ... your code here

		VectorXd err = ground_truth[i] - estimations[i];
		VectorXd err_sq = err.array()*err.array();

		rmse += err_sq;

	}

	// Divide by size to get mean value
	rmse = rmse/estimations.size();

	// Find square root
	rmse = rmse.array().sqrt();


	return rmse;
}


