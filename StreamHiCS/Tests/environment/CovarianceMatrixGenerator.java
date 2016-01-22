package environment;

public class CovarianceMatrixGenerator {

	// Creates a covariance matrix with the given number of dimensions and a
	// block of correlated dimensions beginning at dimension 0.
	public static double[][] generateCovarianceMatrix(int numberOfDimensions, int[] blockBeginnings, int[] blockSizes,
			double covariance) {
		double[][] covarianceMatrix = new double[numberOfDimensions][numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			covarianceMatrix[i][i] = 1;
		}
		
		if(blockBeginnings != null && blockSizes != null){
			int l = blockBeginnings.length;
			assert (l == blockSizes.length);
			
			for(int b = 0; b < l; b++){
				for (int i = blockBeginnings[b]; i < blockBeginnings[b] + blockSizes[b]; i++) {
					for (int j = i + 1; j < blockBeginnings[b] + blockSizes[b]; j++) {
						covarianceMatrix[i][j] = covariance;
						covarianceMatrix[j][i] = covariance;
					}
				}
			}
		}
		
		return covarianceMatrix;
	}
}
