package environment;

public class CovarianceMatrixGenerator {

	// Creates a covariance matrix with the given number of dimensions and a
	// block of correlated dimensions beginning at dimension 0.
	public static double[][] generateCovarianceMatrix(int numberOfDimensions, int blockSize, double covariance) {
		assert (blockSize <= numberOfDimensions);
		double[][] covarianceMatrix = new double[numberOfDimensions][numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			covarianceMatrix[i][i] = 1;
		}

		for (int i = 0; i < blockSize; i++) {
			for (int j = i + 1; j < blockSize; j++) {
				covarianceMatrix[i][j] = covariance;
				covarianceMatrix[j][i] = covariance;
			}
		}

		return covarianceMatrix;
	}
}
