#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);


	// Only going to tune the standard deviations for the Process noise since they are model properties
	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 1.0;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = M_PI/6;

	// Leaving these Values alone since they are actually determined by sensor properties
	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;


	// State Variable Length
	n_x_ = 5;
	// Augmented State Variable Length
	n_aug_ = 7;
	// lambda value
	lambda_ = 3 - n_x_;

	// Initialize Laser and Radar reading dimensions
	n_z_laser_ = 2;
	n_z_radar_ = 3;

	// Set is_initialized to false
	is_initialized_ = false;

	// Set first prediction to false
	first_prediction_ = false;


	// Set Weights (constants so do it only once)
	weights_ = VectorXd(2 * n_aug_ + 1);
	for (int i = 0; i< (2*n_aug_ + 1) ; i++)
	{
		if (i==0)
			weights_(i) = lambda_/(lambda_ + n_aug_);
		else
			weights_(i) = 1.0/(2*(lambda_ + n_aug_));
	}

	R_Las_ = MatrixXd(n_z_laser_, n_z_laser_);

	R_Las_ << std_laspx_*std_laspx_, 0,
			0, std_laspy_*std_laspy_;

	R_Rad_ = MatrixXd(n_z_radar_,n_z_radar_);
	R_Rad_ <<    std_radr_*std_radr_, 0, 0,
			0, std_radphi_*std_radphi_, 0,
			0, 0,std_radrd_*std_radrd_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_)
	{
		// Fill X with ones
		x_.fill(1.0);

		//Initialize as Identity
		P_ << 	1, 0, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 0, 1;

		// Set initial readings
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			double rho = meas_package.raw_measurements_[0];
			double psi = meas_package.raw_measurements_[1];
			double rho_d = meas_package.raw_measurements_[2];

			double px = rho*cos(psi);
			double py = rho*sin(psi);
			double vx = rho_d*cos(psi);
			double vy = rho_d*sin(psi);

			double psi_d = atan2(vy, vx);

			//if initial values are zero
			if (px == 0 && py == 0)
			{
				px = py = 0.0001;
			}

			x_ << 	px,
					py,
					0.0,
					0.0,
					0.0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{

			// Extract laser data
			double x = meas_package.raw_measurements_[0];
			double y = meas_package.raw_measurements_[1];
			x_ <<  	x,
					y,
					0.0,
					0.0,
					0.0;
		}

		time_us_ = meas_package.timestamp_;
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}


	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	// Calculate elapsed time
	// Timestamp is in microseconds
	double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	// High Pass filter for timestep
	if (dt > 0.001)
	{
		// Update State Transition Matrix
		Prediction(dt);
		first_prediction_ = true;
	}


	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	if ((use_radar_) && (meas_package.sensor_type_ == MeasurementPackage::RADAR) && (first_prediction_))
	{
		// Radar updates
		UpdateRadar(meas_package);

	}
	else if ( (use_laser_) && (meas_package.sensor_type_ == MeasurementPackage::LASER ) && (first_prediction_))
	{
		// Laser updates
		UpdateLidar(meas_package);
	}

}

/**
 * GenerateAugmentedSigma_points: Generate Augmented state space sigma points
 * @param: Matrix to write the values in by reference
 */
void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_aug)
{

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create augmented mean state
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;

	P_aug.fill(0.0);
	//create augmented covariance matrix
	P_aug.block<5,5>(0,0) = P_;
	P_aug(5,5) = std_a_*std_a_;
	P_aug(6,6) = std_yawdd_*std_yawdd_;

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++)
	{
		Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i);
		Xsig_aug.col(i+1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*A.col(i);
	}

}

/**
 * SigmaPointPrediction: Predict the sigma points for the current values
 * @param Xsig_aug: Augmented state space sigma points
 * @delta_t: Time step
 *
 */
void UKF::PredictSigmaPoints(MatrixXd Xsig_aug, double delta_t)
{

	//predict sigma points
	for(int i = 0; i <2*n_aug_+1; i++)
	{
		// recover values
		const float px = Xsig_aug(0,i);
		const float py = Xsig_aug(1,i);
		const float v = Xsig_aug(2,i);
		const float psi = Xsig_aug(3,i);
		const float psi_d = Xsig_aug(4,i);
		const float a_v = Xsig_aug(5,i);
		const float a_psi = Xsig_aug(6,i);

		/**
		 * Effectively the Sigma point prediction equation is a multivariate form of Newtons Second equation of motion
		 * S = ut + 0.5at^2
		 * In this section we use the two matrices described as vel_term and acc_term
		 */

		VectorXd vel_term = VectorXd(5);
		VectorXd acc_term = VectorXd(5);
		VectorXd out_sig = VectorXd(5);


		// Compute Velocity term
		// Avoid Divide by zero or very small values
		if (psi_d < 0.001)
		{
			vel_term (0) = v*cos(psi)*delta_t;
			vel_term(1) = v*sin(psi)*delta_t;
		}
		else
		{
			vel_term(0) = (v/psi_d)*(sin(psi + psi_d*delta_t) - sin(psi));
			vel_term(1) = (v/psi_d)*(-cos(psi + psi_d*delta_t) + cos(psi));
		}

		vel_term(2) = 0;
		vel_term(3) = psi_d*delta_t;
		vel_term(4) = 0;


		// Calculate Acceleration term
		// Just compute 0.5at^2 once
		float h_dt2 = 0.5*delta_t*delta_t;

		acc_term(0) = h_dt2*cos(psi)*a_v;
		acc_term(1) = h_dt2*sin(psi)*a_v;
		acc_term(2) = delta_t*a_v;
		acc_term(3) = h_dt2*a_psi;
		acc_term(4) = delta_t*a_psi;


		// compute Predicted Sigma Points
		out_sig = Xsig_aug.col(i).head(5) + vel_term + acc_term;
		Xsig_pred_.col(i) = out_sig;
	}

}



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{

	MatrixXd Xsig_aug = MatrixXd(n_aug_,2*n_aug_ + 1);

	// Augment Points
	AugmentedSigmaPoints(Xsig_aug);

	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Predict using Sigma points
	PredictSigmaPoints(Xsig_aug, delta_t);

	// Calculate Mean x
	x_.fill(0.0);
	for (int i = 0; i < (2*n_aug_ + 1); i++)
	{
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	// Calculate Covariance
	P_.fill(0.0);
	for (int i = 0; i < (2*n_aug_ + 1); i++)
	{
		MatrixXd M = Xsig_pred_.col(i) - x_;
		P_ += weights_(i)*M*M.transpose();
	}

}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

	// Lidar values are x,y so n_z = 2
	int n_z = 2;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_ + 1);

	//Transform into Measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		// Recover Values
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		// measurement model
		Zsig(0, i) = px;
		Zsig(1, i) = py;
	}

	MatrixXd z_pred = VectorXd(n_z);
	//mean predicted measurement
	z_pred.fill(0.0);
	for (int i=0; i < 2*n_aug_+1; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z,n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		//residual
		VectorXd zdiff = Zsig.col(i) - z_pred;

		// normalize between -pi and pi
		NormalizeAngle(zdiff(1));

		// Accumulate S
		S += weights_(i) * zdiff * zdiff.transpose();
	}

	S = S + R_Las_;

	// Cross Correlation Matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < (2*n_aug_ + 1); i++)
	{
		VectorXd zdiff = Zsig.col(i) - z_pred;

		// Normalize between -pi and pi
		NormalizeAngle(zdiff(1));

		// compute state difference from prediction and mean
		VectorXd xdiff = Xsig_pred_.col(i) - x_;
		// Normalize between -pi and pi
		NormalizeAngle(xdiff(1));

		Tc = Tc + weights_(i) * xdiff * zdiff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	// True Laser measurement
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];;
	VectorXd zdiff = z - z_pred;

	// Normalize between -pi and pi
	NormalizeAngle(zdiff(1));

	//update state mean and covariance matrix
	x_ = x_ + K * zdiff;
	P_ = P_ - K*S*K.transpose();

	// Compute NIS Value
	NIS_laser_ = zdiff.transpose()*S.inverse()*zdiff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
	// Radar reading dimension is 3 since values are rho,psi,psi_dot
	int n_z = 3;

	//create matrix Radar Sigma Points
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	Zsig.fill(0.0);
	for (int i = 0; i < (2*n_aug_ + 1); i++)
	{
		// Recover values from Predicted X Sigma points
		double px = Xsig_pred_(0,i);
		double py = Xsig_pred_(1,i);
		double v  = Xsig_pred_(2,i);
		double psi = Xsig_pred_(3,i);

		double v1 = cos(psi)*v;
		double v2 = sin(psi)*v;

		// measurement model
		Zsig(0,i) = sqrt(px*px + py*py);

		// avoid atan2 (0,0)
		if(fabs(px)<0.0001 and fabs(py)<0.0001)
			Zsig(1,i) = 0;
		else
			Zsig(1,i) = atan2(py,px);

		//Handle where both are zero to avoid a divide by zero case
		if (Zsig(0,i) == 0)
		{
			Zsig(2,i) = 0;
		}
		else
			Zsig(2,i) = (px*v1 + py*v2 ) / Zsig(0,i);
	}

	MatrixXd z_pred = VectorXd(n_z);
	//mean predicted measurement
	z_pred.fill(0.0);
	for (int i=0; i < 2*n_aug_+1; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z,n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization

		NormalizeAngle(z_diff(1));

		S += weights_(i) * z_diff * z_diff.transpose();
	}


	//add measurement noise covariance matrix
	S += R_Rad_;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < (2*n_aug_ + 1); i++)
	{
		VectorXd zdiff = Zsig.col(i) - z_pred;

		// Normalize between -pi and pi
		NormalizeAngle(zdiff(1));

		// compute state difference from prediction and mean
		VectorXd xdiff = Xsig_pred_.col(i) - x_;
		// Normalize between -pi and pi
		NormalizeAngle(xdiff(1));

		Tc = Tc + weights_(i) * xdiff * zdiff.transpose();
	}

	// Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	// True Sensor Value
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

	VectorXd zdiff = z - z_pred;

	// Normalize between -pi and pi
	NormalizeAngle(zdiff(1));

	//update state mean and covariance matrix
	x_ = x_ + K * zdiff;
	P_ = P_ - K*S*K.transpose();

	// Compute NIS Value
	NIS_radar_ = zdiff.transpose()*S.inverse()*zdiff;
}

/**
 * Normalizeangle: Normalizes the angle passed by reference
 * @param: angle to normalize
 */
void UKF::NormalizeAngle(double& angle)
{
	// Normalize between -pi and pi
	angle = fmod(angle,M_PI);
	if (angle < -M_PI)
		angle += M_PI;
}
